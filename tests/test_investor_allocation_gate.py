from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from agents.investor_agent import InvestorAgentConfig, run_investor_agent
from planning.goal_accumulation_plan import GoalAccumulationInput, build_goal_accumulation_plan


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


class InvestorAllocationGateTests(unittest.TestCase):
    def test_unknown_allocation_locks_opportunistic_bucket(self) -> None:
        plan = _plan(current_btc_allocation_pct=None, rebalance_signal="UNKNOWN_ALLOCATION")

        self.assertEqual(_bucket(plan, "OPPORTUNISTIC")["status"], "LOCKED")
        self.assertFalse(_bucket(plan, "OPPORTUNISTIC")["allowed_now"])
        self.assertEqual(
            _bucket(plan, "OPPORTUNISTIC")["reason"],
            "Current BTC allocation is unknown; opportunistic accumulation is locked until portfolio risk budget can be checked.",
        )

    def test_unknown_allocation_disables_lump_sum(self) -> None:
        plan = _plan(current_btc_allocation_pct=None, rebalance_signal="UNKNOWN_ALLOCATION")

        self.assertFalse(plan["lump_sum_plan"]["allowed_now"])
        self.assertEqual(
            plan["lump_sum_plan"]["reason"],
            "Current BTC allocation is unknown; lump-sum planning is disabled until portfolio risk budget is known.",
        )

    def test_unknown_allocation_allows_only_base_dca(self) -> None:
        plan = _plan(current_btc_allocation_pct=None, rebalance_signal="UNKNOWN_ALLOCATION")

        self.assertTrue(_bucket(plan, "BASE_DCA")["allowed_now"])
        self.assertFalse(_bucket(plan, "OPPORTUNISTIC")["allowed_now"])
        self.assertFalse(_bucket(plan, "DIP_RESERVE")["allowed_now"])
        self.assertFalse(_bucket(plan, "DEEP_VALUE_RESERVE")["allowed_now"])

    def test_unknown_allocation_sets_allocation_gate_false(self) -> None:
        payload = _investor_payload(current_allocation=None)

        self.assertEqual(payload["institutional_overlay"]["rebalance_signal"], "UNKNOWN_ALLOCATION")
        self.assertEqual(
            payload["institutional_overlay"]["allocation_guidance"],
            "Current BTC allocation is unknown; only base DCA is allowed until allocation is provided.",
        )
        self.assertEqual(
            payload["allocation_gate"],
            {
                "allocation_known": False,
                "larger_buckets_allowed": False,
                "reason": "Current BTC allocation is unknown.",
            },
        )

    def test_dip_reserve_requires_reference_price(self) -> None:
        plan = _plan(reference_price_for_dip=None)
        dip = _bucket(plan, "DIP_RESERVE")

        self.assertFalse(dip["allowed_now"])
        self.assertEqual(dip["status"], "LOCKED")
        self.assertEqual(dip["reason"], "Dip reserve requires a reference price and confirmed drawdown trigger.")
        self.assertIsNone(plan["reference_price_for_dip"])
        self.assertFalse(plan["dip_trigger_met"])

    def test_dip_reserve_unlocks_only_after_15_percent_drawdown(self) -> None:
        no_trigger = _plan(current_price=90_000.0, reference_price_for_dip=100_000.0)
        triggered = _plan(current_price=85_000.0, reference_price_for_dip=100_000.0)

        self.assertFalse(_bucket(no_trigger, "DIP_RESERVE")["allowed_now"])
        self.assertFalse(no_trigger["dip_trigger_met"])
        self.assertTrue(_bucket(triggered, "DIP_RESERVE")["allowed_now"])
        self.assertTrue(triggered["dip_trigger_met"])
        self.assertEqual(triggered["current_drawdown_from_reference_pct"], -15.0)

    def test_lump_sum_requires_known_allocation(self) -> None:
        unknown = _plan(current_btc_allocation_pct=None, rebalance_signal="UNKNOWN_ALLOCATION")
        known = _plan(current_btc_allocation_pct=0.0, rebalance_signal="BELOW_TARGET", institutional_score=80)

        self.assertFalse(unknown["lump_sum_plan"]["allowed_now"])
        self.assertTrue(known["lump_sum_plan"]["allowed_now"])

    def test_lump_sum_requires_institutional_score_70(self) -> None:
        plan = _plan(institutional_score=69)

        self.assertFalse(plan["lump_sum_plan"]["allowed_now"])
        self.assertEqual(plan["lump_sum_plan"]["reason"], "Institutional score is below 70, so lump-sum planning remains disabled.")

    def test_below_target_allows_opportunistic_when_all_conditions_pass(self) -> None:
        plan = _plan(current_btc_allocation_pct=0.0, rebalance_signal="BELOW_TARGET")

        self.assertEqual(_bucket(plan, "OPPORTUNISTIC")["status"], "ACTIVE")
        self.assertTrue(_bucket(plan, "OPPORTUNISTIC")["allowed_now"])

    def test_above_max_allocation_returns_rebalance_warning(self) -> None:
        payload = _investor_payload(current_allocation=5.0)

        self.assertEqual(payload["institutional_overlay"]["rebalance_signal"], "OVER_ALLOCATED")
        self.assertEqual(payload["final_investor_action"], "REBALANCE_WARNING")
        self.assertEqual(_bucket(payload["goal_plan"], "BASE_DCA")["status"], "PAUSED")
        self.assertFalse(_bucket(payload["goal_plan"], "OPPORTUNISTIC")["allowed_now"])


def _plan(**overrides: object) -> dict[str, object]:
    defaults = {
        "current_btc": 1.13059494,
        "target_btc": 2.0,
        "current_price": 100000.0,
        "accumulation_score": 85,
        "accumulation_band": "AGGRESSIVE_ACCUMULATION",
        "margin_of_safety_score": 75,
        "mvrv_value": 1.2,
        "mvrv_missing": False,
        "fear_and_greed_value": 20.0,
        "thesis_risk_level": "LOW",
        "final_investor_action": "AGGRESSIVE_DCA_ALLOWED",
        "rebalance_signal": "BELOW_TARGET",
        "current_btc_allocation_pct": 0.0,
        "target_btc_allocation_pct": 2.0,
        "max_btc_allocation_pct": 5.0,
        "institutional_score": 80,
        "reference_price_for_dip": None,
        "distance_from_200d_ma_pct": -30.0,
        "drawdown_from_cycle_high_pct": -55.0,
    }
    defaults.update(overrides)
    return build_goal_accumulation_plan(GoalAccumulationInput(**defaults))


def _investor_payload(current_allocation: float | None) -> dict[str, object]:
    with TemporaryDirectory() as temp_dir:
        return run_investor_agent(
            InvestorAgentConfig(output_dir=Path(temp_dir), current_btc_allocation_pct=current_allocation),
            market_data_provider=DeepValueMarketDataProvider(),
            fear_greed_provider=FakeProvider({"value": 10.0, "missing": False, "fallback": False}),
            mvrv_provider=FakeProvider(
                {"value": 1.2, "mvrv": 1.2, "source": "coinmetrics", "as_of": "2026-06-17T00:00:00Z", "missing": False, "fallback": False}
            ),
            liquidity_provider=FakeProvider({"value": "EXPANDING", "missing": False, "fallback": False}),
            thesis_risk_provider=FakeProvider({"level": "LOW", "flags": [], "missing": False, "fallback": False}),
        )


def _bucket(plan: dict[str, object], name: str) -> dict[str, object]:
    buckets = {item["bucket_name"]: item for item in plan["tranche_plan"]}
    return buckets[name]


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
