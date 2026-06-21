import json
import unittest

from planning.goal_accumulation_plan import GoalAccumulationInput, TAX_WARNING, build_goal_accumulation_plan


class GoalAccumulationPlanTests(unittest.TestCase):
    def test_remaining_btc_needed_is_calculated_correctly(self) -> None:
        plan = _plan()

        self.assertEqual(plan["remaining_btc_needed"], 0.86940506)

    def test_usd_needed_uses_current_price(self) -> None:
        plan = _plan(current_price=100000.0)

        self.assertEqual(plan["estimated_usd_needed_at_current_price"], 86940.51)

    def test_base_dca_bucket_is_10_percent(self) -> None:
        base = _bucket(_plan(), "BASE_DCA")

        self.assertEqual(base["bucket_pct"], 10)
        self.assertEqual(base["target_btc"], 0.08694051)

    def test_opportunistic_bucket_is_35_percent(self) -> None:
        opportunistic = _bucket(_plan(), "OPPORTUNISTIC")

        self.assertEqual(opportunistic["bucket_pct"], 35)
        self.assertEqual(opportunistic["target_btc"], 0.30429177)

    def test_dip_reserve_bucket_is_35_percent(self) -> None:
        dip_reserve = _bucket(_plan(), "DIP_RESERVE")

        self.assertEqual(dip_reserve["bucket_pct"], 35)
        self.assertEqual(dip_reserve["target_btc"], 0.30429177)

    def test_deep_value_bucket_is_20_percent(self) -> None:
        deep_value = _bucket(_plan(), "DEEP_VALUE_RESERVE")

        self.assertEqual(deep_value["bucket_pct"], 20)
        self.assertEqual(deep_value["target_btc"], 0.17388101)

    def test_tranche_plan_sums_to_remaining_btc(self) -> None:
        plan = _plan()
        tranche_sum = round(sum(item["target_btc"] for item in plan["tranche_plan"]), 8)

        self.assertEqual(tranche_sum, plan["remaining_btc_needed"])

    def test_mvrv_missing_locks_opportunistic_and_deep_value_buckets(self) -> None:
        plan = _plan(accumulation_score=85, margin_of_safety_score=80, mvrv_missing=True, mvrv_value=None)

        self.assertEqual(_bucket(plan, "OPPORTUNISTIC")["status"], "LOCKED")
        self.assertFalse(_bucket(plan, "OPPORTUNISTIC")["allowed_now"])
        self.assertEqual(_bucket(plan, "DEEP_VALUE_RESERVE")["status"], "LOCKED")
        self.assertFalse(_bucket(plan, "DEEP_VALUE_RESERVE")["allowed_now"])
        self.assertIn("MVRV is missing, so larger accumulation buckets remain locked.", plan["goal_rationale"])

    def test_insufficient_data_allows_only_base_dca(self) -> None:
        plan = _plan(final_investor_action="INSUFFICIENT_DATA", mvrv_missing=False, mvrv_value=1.2)

        self.assertTrue(_bucket(plan, "BASE_DCA")["allowed_now"])
        self.assertFalse(_bucket(plan, "OPPORTUNISTIC")["allowed_now"])
        self.assertFalse(_bucket(plan, "DIP_RESERVE")["allowed_now"])
        self.assertFalse(_bucket(plan, "DEEP_VALUE_RESERVE")["allowed_now"])
        self.assertIn("Insufficient data; only small base DCA is allowed.", plan["goal_rationale"])

    def test_good_accumulation_medium_confidence_allows_base_dca_only(self) -> None:
        plan = _plan(
            accumulation_score=67,
            margin_of_safety_score=57,
            final_investor_action="INSUFFICIENT_DATA",
            mvrv_missing=True,
            mvrv_value=None,
        )

        self.assertTrue(_bucket(plan, "BASE_DCA")["allowed_now"])
        self.assertFalse(_bucket(plan, "OPPORTUNISTIC")["allowed_now"])
        self.assertFalse(_bucket(plan, "DEEP_VALUE_RESERVE")["allowed_now"])

    def test_opportunistic_bucket_requires_score_75_margin_65_and_mvrv_available(self) -> None:
        allowed = _plan(
            accumulation_score=76,
            margin_of_safety_score=65,
            mvrv_missing=False,
            mvrv_value=2.0,
            fear_and_greed_value=20.0,
            final_investor_action="INCREASE_DCA_GRADUALLY",
        )
        locked = _plan(
            accumulation_score=76,
            margin_of_safety_score=64,
            mvrv_missing=False,
            mvrv_value=2.0,
            fear_and_greed_value=20.0,
            final_investor_action="INCREASE_DCA_GRADUALLY",
        )

        self.assertEqual(_bucket(allowed, "OPPORTUNISTIC")["status"], "PARTIALLY_AVAILABLE")
        self.assertTrue(_bucket(allowed, "OPPORTUNISTIC")["allowed_now"])
        self.assertEqual(_bucket(locked, "OPPORTUNISTIC")["status"], "LOCKED")

    def test_deep_value_bucket_requires_score_80_margin_70_and_cheap_mvrv(self) -> None:
        allowed = _plan(
            accumulation_score=85,
            margin_of_safety_score=75,
            mvrv_missing=False,
            mvrv_value=1.2,
            fear_and_greed_value=20.0,
            distance_from_200d_ma_pct=-30.0,
            final_investor_action="AGGRESSIVE_DCA_ALLOWED",
        )
        locked = _plan(
            accumulation_score=85,
            margin_of_safety_score=75,
            mvrv_missing=False,
            mvrv_value=2.0,
            fear_and_greed_value=20.0,
            distance_from_200d_ma_pct=-30.0,
            final_investor_action="AGGRESSIVE_DCA_ALLOWED",
        )

        self.assertEqual(_bucket(allowed, "DEEP_VALUE_RESERVE")["status"], "ACTIVE")
        self.assertTrue(_bucket(allowed, "DEEP_VALUE_RESERVE")["allowed_now"])
        self.assertEqual(_bucket(locked, "DEEP_VALUE_RESERVE")["status"], "LOCKED")

    def test_overallocated_portfolio_pauses_or_caps_accumulation(self) -> None:
        plan = _plan(rebalance_signal="OVER_ALLOCATED")

        self.assertEqual(_bucket(plan, "BASE_DCA")["status"], "PAUSED")
        self.assertFalse(_bucket(plan, "BASE_DCA")["allowed_now"])
        self.assertFalse(_bucket(plan, "OPPORTUNISTIC")["allowed_now"])
        self.assertIn("BTC allocation is already above risk budget. Accumulation should pause or slow.", plan["goal_rationale"])

    def test_staged_sell_plan_sums_to_planned_sell_btc(self) -> None:
        plan = _plan()
        planned_sum = round(sum(item["sell_btc"] for item in plan["sell_plan"]), 8)

        self.assertEqual(planned_sum, plan["planned_sell_btc"])
        self.assertEqual([item["trigger_price"] for item in plan["sell_plan"]], [450000.0, 500000.0, 550000.0, 600000.0])

    def test_tax_warning_is_present(self) -> None:
        self.assertEqual(_plan()["tax_warning"], TAX_WARNING)

    def test_no_broker_order_logic_added(self) -> None:
        serialized = json.dumps(_plan())

        self.assertNotIn('"BUY"', serialized)
        self.assertNotIn('"SELL"', serialized)
        self.assertNotIn("AVOID LONG", serialized)
        self.assertNotIn("WATCH FOR REVERSAL", serialized)
        self.assertNotIn("order_endpoint", serialized)


def _plan(**overrides: object) -> dict[str, object]:
    defaults = {
        "current_btc": 1.13059494,
        "target_btc": 2.0,
        "current_price": 100000.0,
        "monthly_dca_usd": None,
        "target_sell_price": 500000.0,
        "planned_sell_btc": 1.0,
        "retain_btc": 1.0,
        "accumulation_score": 67,
        "accumulation_band": "GOOD_ACCUMULATION",
        "margin_of_safety_score": 57,
        "mvrv_value": None,
        "mvrv_missing": True,
        "fear_and_greed_value": 20.0,
        "thesis_risk_level": "LOW",
        "final_investor_action": "INCREASE_DCA_GRADUALLY",
        "rebalance_signal": "BELOW_TARGET",
        "current_btc_allocation_pct": 0.0,
        "target_btc_allocation_pct": 2.0,
        "max_btc_allocation_pct": 5.0,
        "institutional_score": 80,
        "distance_from_200d_ma_pct": -16.0,
        "drawdown_from_cycle_high_pct": -20.0,
    }
    defaults.update(overrides)
    return build_goal_accumulation_plan(GoalAccumulationInput(**defaults))


def _bucket(plan: dict[str, object], name: str) -> dict[str, object]:
    buckets = {item["bucket_name"]: item for item in plan["tranche_plan"]}
    return buckets[name]


if __name__ == "__main__":
    unittest.main()
