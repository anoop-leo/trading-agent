import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urlparse
import unittest

import pandas as pd

from shadow_trading.coinbase_shadow import (
    CoinbaseCandleProvider,
    ShadowPortfolio,
    ShadowSignalCollectionConfig,
    ShadowTradingError,
    ShadowTradingConfig,
    _enriched_false_avoid_analysis,
    _signal_quality_report,
    _validate_signal_quality_or_raise,
    normalize_coinbase_candles,
    collect_enriched_shadow_signals,
    run_coinbase_shadow_trading,
)


PRODUCT_BOOK_PAYLOAD = {
    "pricebook": {
        "product_id": "BTC-USD",
        "time": "2026-06-15T14:00:00Z",
        "bids": [
            {"price": "99990", "size": "0.1"},
            {"price": "99980", "size": "0.1"},
            {"price": "99970", "size": "0.1"},
            {"price": "99960", "size": "0.1"},
            {"price": "99950", "size": "0.1"},
        ],
        "asks": [
            {"price": "100010", "size": "0.1"},
            {"price": "100020", "size": "0.1"},
            {"price": "100030", "size": "0.1"},
            {"price": "100040", "size": "0.1"},
            {"price": "100050", "size": "0.1"},
        ],
    }
}


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class CoinbaseShadowTradingTests(unittest.TestCase):
    def test_normalize_coinbase_candles_sorts_ohlcv(self) -> None:
        payload = {
            "candles": [
                {"start": "1710003600", "open": "101", "high": "104", "low": "100", "close": "103", "volume": "2"},
                {"start": "1710000000", "open": "100", "high": "102", "low": "99", "close": "101", "volume": "1"},
            ]
        }

        frame = normalize_coinbase_candles(payload)

        self.assertEqual(list(frame.columns), ["timestamp", "open", "high", "low", "close", "volume"])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame["timestamp"]))
        self.assertEqual(frame.iloc[0]["close"], 101.0)
        self.assertEqual(frame.iloc[1]["volume"], 2.0)

    def test_candle_provider_uses_public_market_candles_endpoint(self) -> None:
        calls = []

        def opener(request, timeout):
            calls.append((request, timeout))
            return FakeResponse({"candles": _coinbase_candles(3, 3600)})

        provider = CoinbaseCandleProvider(base_url="https://example.test/api/v3/brokerage", timeout_seconds=3.0, opener=opener)
        frame = provider.fetch_ohlcv("BTC-USD", "1h", 3, datetime(2026, 6, 15, tzinfo=UTC))

        request, timeout = calls[0]
        parsed = urlparse(request.full_url)
        query = parse_qs(parsed.query)
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(parsed.path, "/api/v3/brokerage/market/products/BTC-USD/candles")
        self.assertEqual(query["granularity"], ["ONE_HOUR"])
        self.assertEqual(timeout, 3.0)
        self.assertEqual(len(frame), 3)

    def test_shadow_portfolio_opens_and_closes_simulated_trade(self) -> None:
        portfolio = ShadowPortfolio(
            initial_capital=10000,
            intended_order_size_usd=2500,
            max_all_in_cost_per_side=0.0015,
            fee_rate=0.001,
        )
        buy_signal = _signal(final_decision="BUY", price=100.0, target_1=105.0)
        execution_cost = {"all_in_cost_per_side": 0.0012, "price_slippage_pct": 0.0002, "depth_supported": True}

        action, trade = portfolio.process_signal(buy_signal, execution_cost)
        self.assertEqual(action, "BUY")
        self.assertIsNone(trade)
        self.assertIsNotNone(portfolio.open_position)
        self.assertLessEqual(portfolio.open_position.notional, 2500)

        sell_signal = _signal(final_decision="HOLD", price=106.0, target_1=105.0)
        action, trade = portfolio.process_signal(sell_signal, execution_cost)

        self.assertEqual(action, "SELL")
        self.assertIsNotNone(trade)
        self.assertEqual(trade["reason_for_exit"], "TAKE_PROFIT")
        self.assertGreater(trade["net_pnl"], 0)
        self.assertFalse(portfolio.open_position)

    def test_shadow_portfolio_rejects_high_execution_cost(self) -> None:
        portfolio = ShadowPortfolio(10000, 2500, 0.0015, 0.001)
        action, trade = portfolio.process_signal(
            _signal(final_decision="BUY", price=100.0),
            {"all_in_cost_per_side": 0.003, "price_slippage_pct": 0.002, "depth_supported": True},
        )

        self.assertEqual(action, "HOLD")
        self.assertIsNone(trade)
        self.assertIn("high_execution_cost", portfolio.rejected_signals[0]["reasons"])

    def test_run_shadow_trading_writes_artifacts_and_keeps_live_disabled(self) -> None:
        calls = []

        def opener(request, timeout):
            calls.append(request.full_url)
            if "/market/product_book" in request.full_url:
                return FakeResponse(PRODUCT_BOOK_PAYLOAD)
            return FakeResponse({"candles": _coinbase_candles(230, 3600)})

        with TemporaryDirectory() as temp_dir:
            config = ShadowTradingConfig(
                duration_days=1 / 24,
                cycle_interval_seconds=0,
                cycle_limit=1,
                output_dir=Path(temp_dir),
                history_limit=220,
            )
            payload = run_coinbase_shadow_trading(
                config,
                opener=opener,
                sleeper=lambda _seconds: None,
                now_fn=lambda: datetime(2026, 6, 15, 14, tzinfo=UTC),
            )

            summary = payload["shadow_summary_30d"]
            self.assertFalse(summary["live_trading_enabled"])
            self.assertFalse(summary["order_endpoint_calls_allowed"])
            self.assertEqual(summary["final_verdict"], "NEEDS_MORE_SHADOW_TESTING")
            self.assertTrue((Path(temp_dir) / "shadow_trades.csv").exists())
            self.assertTrue((Path(temp_dir) / "shadow_equity_curve.csv").exists())
            self.assertTrue((Path(temp_dir) / "shadow_signals.csv").exists())
            self.assertTrue((Path(temp_dir) / "shadow_signals_v2.csv").exists())
            self.assertTrue((Path(temp_dir) / "signal_journal_v2.json").exists())
            self.assertTrue((Path(temp_dir) / "signal_journal_quality_report.json").exists())
            self.assertTrue((Path(temp_dir) / "enriched_false_avoid_analysis.json").exists())
            self.assertTrue((Path(temp_dir) / "shadow_system_health.json").exists())
            self.assertTrue((Path(temp_dir) / "shadow_summary_30d.json").exists())
            self.assertFalse(any("/orders" in url for url in calls))
            with (Path(temp_dir) / "shadow_equity_curve.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            with (Path(temp_dir) / "shadow_signals.csv").open() as handle:
                signal_rows = list(csv.DictReader(handle))
            self.assertEqual(len(signal_rows), 1)
            self.assertIn("rejection_reasons", signal_rows[0])
            with (Path(temp_dir) / "shadow_signals_v2.csv").open() as handle:
                signal_v2_rows = list(csv.DictReader(handle))
            self.assertEqual(len(signal_v2_rows), 1)
            self.assertNotEqual(signal_v2_rows[0]["ema20"], "")
            self.assertNotEqual(signal_v2_rows[0]["ema50"], "")
            self.assertNotEqual(signal_v2_rows[0]["rsi"], "")
            self.assertNotEqual(signal_v2_rows[0]["macd"], "")
            self.assertNotEqual(signal_v2_rows[0]["volume_ratio"], "")
            self.assertNotEqual(signal_v2_rows[0]["market_regime"], "")
            self.assertNotEqual(signal_v2_rows[0]["multi_timeframe_alignment"], "")
            with (Path(temp_dir) / "signal_journal_quality_report.json").open() as handle:
                quality = json.load(handle)
            self.assertEqual(quality["total_signals"], 1)
            self.assertEqual(quality["incomplete_signal_rows"], 0)
            self.assertEqual(quality["signal_journal_quality_score"], 100.0)
            with (Path(temp_dir) / "signal_journal_v2.json").open() as handle:
                journal = json.load(handle)
            self.assertEqual(len(journal["signals"]), 1)
            with (Path(temp_dir) / "enriched_false_avoid_analysis.json").open() as handle:
                false_avoid = json.load(handle)
            self.assertEqual(false_avoid["total_signals"], 1)
            self.assertFalse(false_avoid["target_reached"])

    def test_collect_enriched_shadow_signals_appends_until_target(self) -> None:
        calls = []
        clock = _FakeClock(datetime(2026, 6, 15, 14, tzinfo=UTC))

        def opener(request, timeout):
            calls.append(request.full_url)
            if "/market/product_book" in request.full_url:
                return FakeResponse(PRODUCT_BOOK_PAYLOAD)
            return FakeResponse({"candles": _coinbase_candles(230, 3600)})

        with TemporaryDirectory() as temp_dir:
            config = ShadowSignalCollectionConfig(
                target_signals=3,
                interval_seconds=0,
                output_dir=Path(temp_dir),
                history_limit=220,
                reset=True,
            )

            payload = collect_enriched_shadow_signals(
                config,
                opener=opener,
                sleeper=lambda _seconds: None,
                now_fn=clock.now,
            )

            report = payload["enriched_false_avoid_analysis"]
            self.assertEqual(report["total_signals"], 3)
            self.assertTrue(report["target_reached"])
            self.assertFalse(report["live_trading_enabled"])
            self.assertFalse(report["order_endpoint_calls_allowed"])
            self.assertFalse(any("/orders" in url for url in calls))
            with (Path(temp_dir) / "shadow_signals_v2.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(len({(row["timestamp"], row["symbol"], row["final_decision"]) for row in rows}), 3)
            with (Path(temp_dir) / "signal_journal_quality_report.json").open() as handle:
                quality = json.load(handle)
            self.assertEqual(quality["total_signals"], 3)
            self.assertEqual(quality["incomplete_signal_rows"], 0)

    def test_signal_quality_report_requires_complete_avoid_reasons(self) -> None:
        row = _complete_signal_row(final_decision="AVOID LONG", rejection_reasons="not_buy_decision")

        report = _signal_quality_report([row])

        self.assertEqual(report["complete_signal_rows"], 1)
        self.assertEqual(report["incomplete_signal_rows"], 0)
        self.assertEqual(report["avoid_long_rows_missing_rejection_reasons"], 0)
        self.assertEqual(report["signal_journal_quality_score"], 100.0)

    def test_signal_quality_validation_rejects_missing_required_fields(self) -> None:
        row = _complete_signal_row()
        row["ema20"] = ""

        with self.assertRaises(ShadowTradingError):
            _validate_signal_quality_or_raise(row)

    def test_enriched_false_avoid_analysis_classifies_false_avoid(self) -> None:
        rows = [
            {
                **_complete_signal_row(final_decision="AVOID LONG", rejection_reasons="not_buy_decision"),
                "timestamp": "2026-01-01T00:00:00+00:00",
                "product_id": "BTC-USD",
                "price": 100.0,
            },
            {
                **_complete_signal_row(),
                "timestamp": "2026-01-01T01:00:00+00:00",
                "product_id": "BTC-USD",
                "price": 103.0,
            },
        ]

        report = _enriched_false_avoid_analysis(rows)

        self.assertEqual(report["total_signals"], 2)
        self.assertEqual(report["total_avoid_long_signals"], 1)
        self.assertEqual(report["false_avoid_count"], 1)
        self.assertEqual(report["most_common_false_avoid_rejection_reasons"], {"not_buy_decision": 1})


def _signal(final_decision: str, price: float, target_1: float | None = None) -> dict[str, object]:
    return {
        "timestamp": "2026-06-15T14:00:00+00:00",
        "price": price,
        "final_decision": final_decision,
        "final_decision_reason": "test signal",
        "alignment": "BULLISH_ALIGNMENT",
        "rr_ratio": 2.0,
        "volume_ratio": 1.2,
        "market_regime": "BULL",
        "daily_setup": "TREND_FOLLOWING",
        "four_hour_price": price,
        "four_hour_ema20": price - 1,
        "trend_score": 8,
        "confidence": 80,
        "alignment_score": 9,
        "stop_loss": price - 10,
        "target_1": target_1,
        "target_2": price + 20,
    }


def _complete_signal_row(
    final_decision: str = "BUY",
    rejection_reasons: str = "",
) -> dict[str, object]:
    return {
        "price": 100.0,
        "setup": "TREND_FOLLOWING",
        "final_decision": final_decision,
        "ema20": 99.0,
        "ema50": 98.0,
        "rsi": 55.0,
        "macd": "bullish",
        "volume_ratio": 1.2,
        "market_regime": "BULL",
        "multi_timeframe_alignment": "BULLISH_ALIGNMENT",
        "rejection_reasons": rejection_reasons,
    }


class _FakeClock:
    def __init__(self, start: datetime) -> None:
        self.current = start

    def now(self) -> datetime:
        value = self.current
        self.current = self.current + pd.Timedelta(hours=1)
        return value


def _coinbase_candles(count: int, step_seconds: int) -> list[dict[str, str]]:
    start = 1_700_000_000
    candles = []
    for index in range(count):
        close = 100.0 + index
        candles.append(
            {
                "start": str(start + index * step_seconds),
                "open": str(close - 0.5),
                "high": str(close + 1.0),
                "low": str(close - 1.0),
                "close": str(close),
                "volume": "1000",
            }
        )
    return list(reversed(candles))


if __name__ == "__main__":
    unittest.main()
