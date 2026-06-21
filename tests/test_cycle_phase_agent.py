from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from agents.btc.cycle_phase_agent import (
    CyclePhaseInput,
    classify_btc_cycle_phase,
    cycle_adjusted_action,
    cycle_adjusted_dca_multiplier,
)
from agents.investor_agent import InvestorAgentConfig, run_investor_agent


class FakeMarketDataProvider:
    def fetch_ohlcv(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        del symbol, limit
        if interval == "1d":
            return _frame("2023-01-01", 300, "D", 90.0, 120.0, cycle_high=200.0)
        if interval == "1w":
            return _frame("2019-01-01", 300, "W", 120.0, 80.0)
        if interval == "1M":
            return _frame("2017-01-01", 80, "ME", 120.0, 70.0)
        raise AssertionError(f"Unexpected interval {interval}")


class FakeProvider:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def fetch(self, offline: bool = False) -> dict[str, object]:
        del offline
        return self.payload


class CyclePhaseAgentTests(unittest.TestCase):
    def test_late_bear_case_with_current_metrics(self) -> None:
        payload = classify_btc_cycle_phase(
            CyclePhaseInput(
                price=64_500.0,
                ma200=77_000.0,
                monthly_ema20=70_000.0,
                weekly_rsi=36.0,
                mvrv=1.23,
                drawdown_from_cycle_high_pct=-49.0,
                fear_and_greed=15.0,
            )
        )

        self.assertEqual(payload["agent"], "BTC_CYCLE_PHASE")
        self.assertEqual(payload["cycle_phase"], "LATE_BEAR")
        self.assertEqual(payload["accumulation_bias"], "HIGH")
        self.assertEqual(payload["profit_taking_bias"], "LOW")
        self.assertEqual(payload["phase_confidence"], "HIGH")

    def test_euphoria_case(self) -> None:
        payload = classify_btc_cycle_phase(
            CyclePhaseInput(
                price=210_000.0,
                ma200=80_000.0,
                monthly_ema20=120_000.0,
                weekly_rsi=86.0,
                mvrv=4.4,
                drawdown_from_cycle_high_pct=-3.0,
                fear_and_greed=91.0,
            )
        )

        self.assertEqual(payload["cycle_phase"], "EUPHORIA")
        self.assertEqual(payload["phase_score"], 10)
        self.assertEqual(cycle_adjusted_dca_multiplier("1.5x to 2.5x normal DCA", "EUPHORIA"), "0.0x to 0.25x normal DCA")
        self.assertEqual(cycle_adjusted_action("INCREASE_DCA_GRADUALLY", "EUPHORIA"), "PROFIT_TAKING_REVIEW")

    def test_early_bull_case(self) -> None:
        payload = classify_btc_cycle_phase(
            CyclePhaseInput(
                price=70_000.0,
                ma200=65_000.0,
                monthly_ema20=68_000.0,
                weekly_rsi=52.0,
                mvrv=1.8,
                drawdown_from_cycle_high_pct=-25.0,
                fear_and_greed=45.0,
            )
        )

        self.assertEqual(payload["cycle_phase"], "EARLY_BULL")
        self.assertEqual(payload["accumulation_bias"], "HIGH")

    def test_missing_mvrv_fallback_case(self) -> None:
        payload = classify_btc_cycle_phase(
            CyclePhaseInput(
                price=70_000.0,
                ma200=65_000.0,
                monthly_ema20=68_000.0,
                weekly_rsi=52.0,
                mvrv=None,
                drawdown_from_cycle_high_pct=-25.0,
                fear_and_greed=45.0,
            )
        )

        self.assertEqual(payload["cycle_phase"], "UNKNOWN")
        self.assertEqual(payload["phase_confidence"], "LOW")
        self.assertIn("mvrv", payload["inputs"]["missing_fields"])
        self.assertIn("MVRV is unavailable, so cycle phase confidence is limited.", payload["rationale"])

    def test_investor_report_includes_cycle_overlay_fields(self) -> None:
        with TemporaryDirectory() as temp_dir:
            payload = run_investor_agent(
                InvestorAgentConfig(output_dir=Path(temp_dir)),
                market_data_provider=FakeMarketDataProvider(),
                fear_greed_provider=FakeProvider({"value": 15.0, "missing": False, "fallback": False}),
                mvrv_provider=FakeProvider(
                    {
                        "value": 1.23,
                        "mvrv": 1.23,
                        "source": "coinmetrics",
                        "as_of": "2026-06-17T00:00:00Z",
                        "missing": False,
                        "fallback": False,
                    }
                ),
                liquidity_provider=FakeProvider({"value": "NEUTRAL", "missing": False, "fallback": True}),
                thesis_risk_provider=FakeProvider({"level": "LOW", "flags": [], "missing": False, "fallback": False}),
            )

        self.assertIn("cycle_overlay", payload)
        self.assertIn("cycle_adjusted_dca_multiplier", payload)
        self.assertIn("cycle_adjusted_action", payload)
        self.assertEqual(payload["cycle_overlay"]["agent"], "BTC_CYCLE_PHASE")


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


if __name__ == "__main__":
    unittest.main()
