from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from backtesting.backtest_engine import (
    BacktestConfig,
    _precompute_support_resistance,
    _slice_available_history,
    cache_file_path,
    load_or_download_timeframes,
    run_backtest,
)
from scoring.support_resistance_skill import calculate_support_resistance


def ohlcv_frame(start: str, periods: int, freq: str, start_price: float = 100.0) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    close = np.linspace(start_price, start_price + periods, periods)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(periods, 1000.0),
        }
    )


class BacktestEngineTests(unittest.TestCase):
    def test_higher_timeframe_slice_uses_only_completed_candles(self) -> None:
        frame = ohlcv_frame("2024-01-01T00:00:00Z", 3, "4h")

        before_completion = _slice_available_history(
            frame,
            "4h",
            pd.Timestamp("2024-01-01T05:00:00Z"),
            include_current=False,
        )
        at_completion = _slice_available_history(
            frame,
            "4h",
            pd.Timestamp("2024-01-01T08:00:00Z"),
            include_current=False,
        )

        self.assertEqual(len(before_completion), 1)
        self.assertEqual(len(at_completion), 2)
        self.assertLessEqual(before_completion["timestamp"].max(), pd.Timestamp("2024-01-01T00:00:00Z"))

    def test_primary_timeframe_slice_uses_only_completed_candles(self) -> None:
        frame = ohlcv_frame("2024-01-01T00:00:00Z", 3, "1h")

        sliced = _slice_available_history(
            frame,
            "1h",
            pd.Timestamp("2024-01-01T01:00:00Z"),
            include_current=True,
        )

        self.assertEqual(len(sliced), 1)
        self.assertEqual(sliced.iloc[-1]["timestamp"], pd.Timestamp("2024-01-01T00:00:00Z"))

    def test_run_backtest_uses_cached_data_and_returns_result(self) -> None:
        cached_data = {
            "1h": ohlcv_frame("2024-01-01T00:00:00Z", 240, "1h", 100.0),
            "4h": ohlcv_frame("2024-01-01T00:00:00Z", 80, "4h", 100.0),
            "1d": ohlcv_frame("2024-01-01T00:00:00Z", 20, "1D", 100.0),
        }
        config = BacktestConfig(
            symbol="BTCUSDT",
            start="2024-01-01",
            end="2024-01-20",
            initial_capital=10000.0,
        )

        with patch("backtesting.backtest_engine.MIN_HISTORY_ROWS", 5):
            result = run_backtest(config, cached_data=cached_data)

        self.assertEqual(result.symbol, "BTCUSDT")
        self.assertGreater(len(result.decisions), 0)
        self.assertIn("total_return_pct", result.metrics)
        self.assertIn("total_decisions", result.metrics)
        self.assertIn("entry_attempts", result.metrics)
        self.assertIn("accepted_entries", result.metrics)
        self.assertIn("rejected_entries", result.metrics)
        self.assertIn("rejected_entry_reasons", result.metrics)
        self.assertIn("exit_reasons", result.metrics)
        self.assertIn("exits_before_minimum_hold", result.metrics)
        self.assertIn("momentum_exits_blocked", result.metrics)
        self.assertIn("stop_out_count", result.metrics)
        self.assertIn("average_stop_distance_pct", result.metrics)
        self.assertIn("average_stop_distance_atr", result.metrics)
        self.assertIn("average_holding_hours", result.metrics)
        self.assertIn("median_holding_hours", result.metrics)
        self.assertIn("average_r_multiple_by_exit_reason", result.metrics)
        self.assertIn("buy_and_hold_equity", result.equity_curve.columns)
        self.assertIn("stop_type", result.decisions[-1])
        self.assertIn("stop_candidates", result.decisions[-1])

    def test_run_backtest_uses_pre_start_history_for_warmup(self) -> None:
        cached_data = {
            "1h": ohlcv_frame("2024-01-01T00:00:00Z", 120, "1h", 100.0),
            "4h": ohlcv_frame("2023-12-01T00:00:00Z", 300, "4h", 100.0),
            "1d": ohlcv_frame("2023-01-01T00:00:00Z", 500, "1D", 100.0),
        }
        config = BacktestConfig(
            symbol="BTCUSDT",
            start="2024-01-04",
            end="2024-01-04T12:00:00Z",
            initial_capital=10000.0,
        )

        with patch("backtesting.backtest_engine.MIN_HISTORY_ROWS", 20):
            result = run_backtest(config, cached_data=cached_data)

        self.assertGreater(len(result.decisions), 0)
        self.assertGreaterEqual(result.start_date, "2024-01-04")

    def test_precomputed_support_resistance_matches_skill_for_each_slice(self) -> None:
        frame = ohlcv_frame("2024-01-01T00:00:00Z", 30, "1h", 100.0)

        precomputed = _precompute_support_resistance(frame)

        self.assertEqual(len(precomputed), len(frame))
        for index, result in enumerate(precomputed):
            expected = calculate_support_resistance(frame.iloc[: index + 1], float(frame.iloc[index]["close"]))
            self.assertEqual(result.support, expected.support)
            self.assertEqual(result.resistance, expected.resistance)
            self.assertEqual(result.sr_score, expected.sr_score)

    def test_run_backtest_emits_progress_events(self) -> None:
        cached_data = {
            "1h": ohlcv_frame("2024-01-01T00:00:00Z", 240, "1h", 100.0),
            "4h": ohlcv_frame("2024-01-01T00:00:00Z", 80, "4h", 100.0),
            "1d": ohlcv_frame("2024-01-01T00:00:00Z", 20, "1D", 100.0),
        }
        config = BacktestConfig(
            symbol="BTCUSDT",
            start="2024-01-01",
            end="2024-01-20",
            initial_capital=10000.0,
            progress_interval=10,
        )
        events: list[dict] = []

        with patch("backtesting.backtest_engine.MIN_HISTORY_ROWS", 5):
            run_backtest(config, cached_data=cached_data, progress_callback=events.append)

        self.assertEqual(events[0]["phase"], "prepared")
        self.assertTrue(any(event.get("phase") == "replay" for event in events))
        self.assertEqual(events[-1]["processed_rows"], 240)

    def test_cache_prepends_missing_older_range(self) -> None:
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            cache_path = cache_file_path(cache_dir, "BTCUSDT", "1h")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            ohlcv_frame("2024-01-03T00:00:00Z", 3, "1h", 200.0).to_csv(cache_path, index=False)
            config = BacktestConfig(
                symbol="BTCUSDT",
                start="2024-01-01",
                end="2024-01-03T02:00:00Z",
                cache_dir=cache_dir,
            )

            with patch(
                "backtesting.backtest_engine.download_binance_history",
                return_value=ohlcv_frame("2024-01-01T00:00:00Z", 2, "1h", 100.0),
            ) as download:
                frames = load_or_download_timeframes(config, ("1h",))

            self.assertEqual(download.call_count, 1)
            self.assertEqual(frames["1h"]["timestamp"].min(), pd.Timestamp("2024-01-01T00:00:00Z"))
            self.assertEqual(frames["1h"]["timestamp"].max(), pd.Timestamp("2024-01-03T02:00:00Z"))

    def test_cache_appends_missing_newer_range(self) -> None:
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            cache_path = cache_file_path(cache_dir, "BTCUSDT", "1h")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            ohlcv_frame("2024-01-01T00:00:00Z", 2, "1h", 100.0).to_csv(cache_path, index=False)
            config = BacktestConfig(
                symbol="BTCUSDT",
                start="2024-01-01",
                end="2024-01-01T04:00:00Z",
                cache_dir=cache_dir,
            )

            with patch(
                "backtesting.backtest_engine.download_binance_history",
                return_value=ohlcv_frame("2024-01-01T02:00:00Z", 3, "1h", 200.0),
            ) as download:
                frames = load_or_download_timeframes(config, ("1h",))

            self.assertEqual(download.call_count, 1)
            self.assertEqual(len(frames["1h"]), 5)
            self.assertEqual(frames["1h"]["timestamp"].max(), pd.Timestamp("2024-01-01T04:00:00Z"))

    def test_refresh_cache_rebuilds_requested_range(self) -> None:
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            cache_path = cache_file_path(cache_dir, "BTCUSDT", "1h")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            ohlcv_frame("2024-01-01T00:00:00Z", 2, "1h", 100.0).to_csv(cache_path, index=False)
            config = BacktestConfig(
                symbol="BTCUSDT",
                start="2024-01-05",
                end="2024-01-05T02:00:00Z",
                cache_dir=cache_dir,
                refresh_cache=True,
            )

            with patch(
                "backtesting.backtest_engine.download_binance_history",
                return_value=ohlcv_frame("2024-01-05T00:00:00Z", 3, "1h", 300.0),
            ) as download:
                frames = load_or_download_timeframes(config, ("1h",))

            self.assertEqual(download.call_count, 1)
            self.assertEqual(frames["1h"]["timestamp"].min(), pd.Timestamp("2024-01-05T00:00:00Z"))
            self.assertEqual(frames["1h"]["timestamp"].max(), pd.Timestamp("2024-01-05T02:00:00Z"))


if __name__ == "__main__":
    unittest.main()
