from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from agents.investor_agent import InvestorAgentConfig, run_investor_agent
from scoring.institutional_overlay import (
    InstitutionalOverlayInput,
    calculate_institutional_overlay,
    score_network_adoption,
    score_store_of_value_thesis,
)


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


class InstitutionalOverlayTests(unittest.TestCase):
    def test_unknown_allocation_returns_unknown_allocation_signal(self) -> None:
        overlay = calculate_institutional_overlay(InstitutionalOverlayInput())

        self.assertEqual(overlay.rebalance_signal, "UNKNOWN_ALLOCATION")
        self.assertEqual(overlay.portfolio_risk_budget_score, 50)
        self.assertEqual(overlay.portfolio_discipline_score, 50)
        self.assertIn("Current BTC allocation is unknown; allocation guidance is limited.", overlay.institutional_rationale)

    def test_below_target_allows_calculated_dca(self) -> None:
        payload = _investor_payload(current_allocation=0.0)

        self.assertEqual(payload["institutional_overlay"]["rebalance_signal"], "BELOW_TARGET")
        self.assertEqual(payload["confidence_adjusted_dca_multiplier"], "1.5x to 2.5x normal DCA")
        self.assertEqual(payload["final_investor_action"], "AGGRESSIVE_DCA_ALLOWED")

    def test_above_target_caps_dca(self) -> None:
        payload = _investor_payload(current_allocation=3.0)

        self.assertEqual(payload["institutional_overlay"]["rebalance_signal"], "ABOVE_TARGET")
        self.assertEqual(payload["confidence_adjusted_dca_multiplier"], "0.5x to 1.0x normal DCA")
        self.assertIn("BTC allocation above target; capped max DCA multiplier at 1.0x.", payload["confidence_adjustments"])

    def test_above_max_returns_rebalance_warning(self) -> None:
        payload = _investor_payload(current_allocation=5.0)

        self.assertEqual(payload["institutional_overlay"]["rebalance_signal"], "OVER_ALLOCATED")
        self.assertEqual(payload["confidence_adjusted_dca_multiplier"], "0.0x to 0.5x normal DCA")
        self.assertEqual(payload["final_investor_action"], "REBALANCE_WARNING")

    def test_high_thesis_risk_pauses_extra_dca(self) -> None:
        payload = _investor_payload(current_allocation=0.0, thesis_risk_level="HIGH")

        self.assertEqual(payload["final_investor_action"], "PAUSE_EXTRA_DCA")
        self.assertEqual(payload["confidence_adjusted_dca_multiplier"], "0.0x to 0.5x normal DCA")

    def test_aggressive_dca_requires_accumulation_margin_and_institutional_confirmation(self) -> None:
        unknown_allocation = _investor_payload(current_allocation=None)
        below_target = _investor_payload(current_allocation=0.0)

        self.assertEqual(unknown_allocation["institutional_overlay"]["institutional_score"], 50)
        self.assertNotEqual(unknown_allocation["final_investor_action"], "AGGRESSIVE_DCA_ALLOWED")
        self.assertGreaterEqual(below_target["accumulation_score"], 80)
        self.assertGreaterEqual(below_target["margin_of_safety_score"], 70)
        self.assertGreaterEqual(below_target["institutional_overlay"]["institutional_score"], 70)
        self.assertEqual(below_target["final_investor_action"], "AGGRESSIVE_DCA_ALLOWED")

    def test_missing_adoption_data_uses_neutral_fallback(self) -> None:
        score, fallback_fields, rationale = score_network_adoption(None)

        self.assertEqual(score, 50)
        self.assertEqual(fallback_fields, ["network_adoption"])
        self.assertIn("Network adoption data is unavailable; neutral fallback score used.", rationale)

    def test_missing_store_of_value_data_uses_neutral_fallback(self) -> None:
        score, fallback_fields, rationale = score_store_of_value_thesis(None)

        self.assertEqual(score, 50)
        self.assertEqual(fallback_fields, ["store_of_value_thesis"])
        self.assertIn("Store-of-value thesis data is unavailable; neutral fallback score used.", rationale)


def _investor_payload(current_allocation: float | None, thesis_risk_level: str = "LOW") -> dict[str, object]:
    with TemporaryDirectory() as temp_dir:
        return run_investor_agent(
            InvestorAgentConfig(output_dir=Path(temp_dir), current_btc_allocation_pct=current_allocation),
            market_data_provider=DeepValueMarketDataProvider(),
            fear_greed_provider=FakeProvider({"value": 10.0, "missing": False, "fallback": False}),
            mvrv_provider=FakeProvider({"value": 1.2, "missing": False, "fallback": False}),
            liquidity_provider=FakeProvider({"value": "EXPANDING", "missing": False, "fallback": False}),
            thesis_risk_provider=FakeProvider(
                {"level": thesis_risk_level, "flags": [], "missing": False, "fallback": False}
            ),
        )


def _frame(
    start: str,
    periods: int,
    freq: str,
    first_close: float,
    last_close: float,
) -> pd.DataFrame:
    close = np.linspace(first_close, last_close, periods)
    return _frame_from_close(start, freq, close)


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


if __name__ == "__main__":
    unittest.main()
