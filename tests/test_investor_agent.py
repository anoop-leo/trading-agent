import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from agents.investor_agent import INVESTOR_REPORT_FILENAME, InvestorAgentConfig, run_investor_agent


class FakeMarketDataProvider:
    def fetch_ohlcv(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        del symbol, limit
        if interval == "1d":
            return _frame("2023-01-01", 300, "D", 90.0, 120.0, cycle_high=200.0)
        if interval == "1w":
            return _frame("2020-01-01", 120, "W", 120.0, 80.0)
        if interval == "1M":
            return _frame("2017-01-01", 80, "ME", 20.0, 100.0)
        raise AssertionError(f"Unexpected interval {interval}")


class DeepValueMarketDataProvider:
    def fetch_ohlcv(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        del symbol, limit
        if interval == "1d":
            close = np.concatenate([np.full(250, 160.0), np.full(50, 100.0)])
            high = close * 1.01
            high[10] = 400.0
            return _frame_from_close("2023-01-01", "D", close, high)
        if interval == "1w":
            return _frame("2020-01-01", 120, "W", 140.0, 70.0)
        if interval == "1M":
            return _frame("2017-01-01", 80, "ME", 120.0, 70.0)
        raise AssertionError(f"Unexpected interval {interval}")


class FakeProvider:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def fetch(self, offline: bool = False) -> dict[str, object]:
        del offline
        return self.payload


class InvestorAgentTests(unittest.TestCase):
    def test_investor_agent_outputs_accumulation_report_without_trading_labels(self) -> None:
        with TemporaryDirectory() as temp_dir:
            payload = run_investor_agent(
                InvestorAgentConfig(output_dir=Path(temp_dir)),
                market_data_provider=FakeMarketDataProvider(),
                fear_greed_provider=FakeProvider({"value": 25.0, "missing": False, "fallback": False}),
                mvrv_provider=FakeProvider({"value": 1.2, "missing": False, "fallback": False}),
                liquidity_provider=FakeProvider({"value": "EXPANDING", "missing": False, "fallback": False}),
                thesis_risk_provider=FakeProvider(
                    {"level": "LOW", "flags": [], "missing": False, "fallback": False}
                ),
            )

            self.assertEqual(payload["agent"], "BTC_INVESTOR")
            self.assertEqual(payload["symbol"], "BTC")
            self.assertIn("accumulation_score", payload)
            self.assertIn("margin_of_safety_score", payload)
            self.assertIn("factor_scores", payload)
            self.assertEqual(payload["thesis_risk"], {"level": "LOW", "flags": []})
            self.assertTrue((Path(temp_dir) / INVESTOR_REPORT_FILENAME).exists())

            serialized = json.dumps(payload)
            self.assertNotIn('"BUY"', serialized)
            self.assertNotIn('"SELL"', serialized)
            self.assertNotIn("AVOID LONG", serialized)
            self.assertNotIn("WATCH FOR REVERSAL", serialized)

    def test_investor_agent_offline_marks_missing_market_data(self) -> None:
        with TemporaryDirectory() as temp_dir:
            payload = run_investor_agent(
                InvestorAgentConfig(output_dir=Path(temp_dir), offline=True),
                fear_greed_provider=FakeProvider({"value": None, "missing": True, "fallback": False}),
                mvrv_provider=FakeProvider({"value": None, "missing": True, "fallback": False}),
                liquidity_provider=FakeProvider({"value": "NEUTRAL", "missing": False, "fallback": True}),
                thesis_risk_provider=FakeProvider(
                    {"level": "LOW", "flags": [], "missing": False, "fallback": False}
                ),
            )

            self.assertIn("distance_from_200d_ma", payload["data_quality"]["missing_fields"])
            self.assertIn("mvrv", payload["data_quality"]["missing_fields"])
            self.assertIn("liquidity_condition", payload["data_quality"]["fallback_fields"])
            self.assertEqual(payload["data_quality"]["confidence"], "LOW")

    def test_missing_mvrv_caps_dca_multiplier(self) -> None:
        payload = _deep_value_payload({"value": None, "missing": True, "fallback": False})

        self.assertEqual(payload["accumulation_score"], 90)
        self.assertEqual(payload["accumulation_band"], "AGGRESSIVE_ACCUMULATION")
        self.assertEqual(payload["suggested_dca_multiplier"], "1.5x to 2.5x normal DCA")
        self.assertEqual(payload["confidence_adjusted_dca_multiplier"], "1.0x to 1.25x normal DCA")
        self.assertIn("MVRV missing; capped max DCA multiplier at 1.25x.", payload["confidence_adjustments"])
        self.assertIn(
            "MVRV is unavailable, so aggressive accumulation is capped until valuation is confirmed.",
            payload["rationale"],
        )

    def test_low_mvrv_allows_full_dca_multiplier(self) -> None:
        payload = _deep_value_payload({"value": 1.2, "missing": False, "fallback": False})

        self.assertEqual(payload["accumulation_band"], "AGGRESSIVE_ACCUMULATION")
        self.assertEqual(payload["suggested_dca_multiplier"], "1.5x to 2.5x normal DCA")
        self.assertEqual(payload["confidence_adjusted_dca_multiplier"], "1.5x to 2.5x normal DCA")
        self.assertEqual(payload["confidence_adjustments"], [])

    def test_high_mvrv_caps_accumulation_band(self) -> None:
        payload = _deep_value_payload({"value": 3.5, "missing": False, "fallback": False})

        self.assertEqual(payload["accumulation_band"], "FAIR")
        self.assertEqual(payload["dca_intensity"], "NORMAL_DCA")
        self.assertEqual(payload["suggested_dca_multiplier"], "0.5x to 1.0x normal DCA")
        self.assertEqual(payload["confidence_adjusted_dca_multiplier"], "0.5x to 1.0x normal DCA")
        self.assertIn(
            "MVRV is elevated, so accumulation is capped despite other positive signals.",
            payload["rationale"],
        )

    def test_confidence_adjustments_are_reported(self) -> None:
        payload = _deep_value_payload({"value": None, "missing": True, "fallback": False})

        self.assertIn("confidence_adjustments", payload)
        self.assertEqual(payload["confidence_adjustments"], ["MVRV missing; capped max DCA multiplier at 1.25x."])


def _frame(
    start: str,
    periods: int,
    freq: str,
    first_close: float,
    last_close: float,
    cycle_high: float | None = None,
) -> pd.DataFrame:
    close = np.linspace(first_close, last_close, periods)
    high = close * 1.01
    if cycle_high is not None:
        high[10] = cycle_high
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=periods, freq=freq, tz="UTC"),
            "open": close * 0.99,
            "high": high,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(periods, 1000.0),
        }
    )


def _frame_from_close(start: str, freq: str, close: np.ndarray, high: np.ndarray | None = None) -> pd.DataFrame:
    if high is None:
        high = close * 1.01
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=len(close), freq=freq, tz="UTC"),
            "open": close * 0.99,
            "high": high,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(len(close), 1000.0),
        }
    )


def _deep_value_payload(mvrv_payload: dict[str, object]) -> dict[str, object]:
    with TemporaryDirectory() as temp_dir:
        return run_investor_agent(
            InvestorAgentConfig(output_dir=Path(temp_dir)),
            market_data_provider=DeepValueMarketDataProvider(),
            fear_greed_provider=FakeProvider({"value": 10.0, "missing": False, "fallback": False}),
            mvrv_provider=FakeProvider(mvrv_payload),
            liquidity_provider=FakeProvider({"value": "EXPANDING", "missing": False, "fallback": False}),
            thesis_risk_provider=FakeProvider({"level": "LOW", "flags": [], "missing": False, "fallback": False}),
        )


if __name__ == "__main__":
    unittest.main()
