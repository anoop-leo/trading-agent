import unittest

from trading_agent.entry_gates import apply_entry_quality_gates


def _payload(**overrides) -> dict:
    base = {
        "symbol": "AAVEUSDT",
        "final_decision": "BUY",
        "final_decision_reason": "Multi-timeframe alignment supports long-side setups.",
        "rr_ratio": 2.0,
        "rr_score": 5,
        "volume_score": 6,
        "rsi": 60.0,
        "volume_ratio": 1.5,
        "setup": "TREND_FOLLOWING",
        "regime_score": 9,
        "multi_timeframe": {"alignment": "BULLISH_ALIGNMENT"},
        "support": 90.0,
        "resistance": 110.0,
        "ema20": 95.0,
        "stop_loss": 88.0,
    }
    base.update(overrides)
    return base


class GateFiringTests(unittest.TestCase):
    def test_clean_buy_passes_all_gates(self) -> None:
        p = apply_entry_quality_gates(_payload())
        self.assertEqual(p["pre_gate_decision"], "BUY")
        self.assertEqual(p["final_decision"], "BUY")
        self.assertEqual(p["entry_decision"], "BUY")
        self.assertIsNone(p["gate_triggered"])
        self.assertIsNone(p["watch_levels"])
        self.assertIsNone(p["alert_summary"])

    def test_low_rr_gates_to_wait_for_pullback(self) -> None:
        p = apply_entry_quality_gates(_payload(rr_ratio=0.8, setup="TREND_FOLLOWING"))
        self.assertEqual(p["pre_gate_decision"], "BUY")
        self.assertEqual(p["final_decision"], "WAIT")
        self.assertEqual(p["gate_triggered"], "RR_BELOW_1R")
        self.assertEqual(p["entry_decision"], "WAIT_FOR_PULLBACK")
        self.assertEqual(p["watch_levels"]["pullback_entry"], {"low": 90.0, "high": 95.0})
        self.assertEqual(p["watch_levels"]["breakout_entry"], {"level": 110.0, "volume_requirement": 1.3})
        self.assertEqual(p["watch_levels"]["invalidation"], 88.0)
        self.assertIn("Watch pullback 90.0–95.0 or breakout above 110.0", p["alert_summary"])

    def test_missing_rr_score_fires_null_path_not_silent_pass(self) -> None:
        p = apply_entry_quality_gates(_payload(rr_ratio=None, rr_score=None, volume_score=5, setup="BREAKOUT"))
        self.assertEqual(p["final_decision"], "WAIT")
        self.assertEqual(p["gate_triggered"], "MISSING_ENTRY_SCORE")
        self.assertEqual(p["entry_decision"], "WAIT")  # not TREND_FOLLOWING -> plain WAIT
        self.assertIn("rr_score=None", p["final_decision_reason"])

    def test_missing_volume_score_also_fires(self) -> None:
        p = apply_entry_quality_gates(_payload(volume_score=None))
        self.assertEqual(p["gate_triggered"], "MISSING_ENTRY_SCORE")
        self.assertEqual(p["final_decision"], "WAIT")

    def test_zero_rr_and_volume_gates(self) -> None:
        p = apply_entry_quality_gates(_payload(rr_score=0, volume_score=0))
        self.assertEqual(p["gate_triggered"], "ZERO_RR_AND_VOLUME")

    def test_overbought_low_volume_gates(self) -> None:
        p = apply_entry_quality_gates(_payload(rsi=80.0, volume_ratio=0.7))
        self.assertEqual(p["gate_triggered"], "OVERBOUGHT_LOW_VOLUME")

    def test_first_matching_gate_wins(self) -> None:
        # rr_ratio<1 AND scores zero -> RR gate fires first.
        p = apply_entry_quality_gates(_payload(rr_ratio=0.5, rr_score=0, volume_score=0))
        self.assertEqual(p["gate_triggered"], "RR_BELOW_1R")


class NonBuyUntouchedTests(unittest.TestCase):
    def test_non_buy_decisions_are_not_gated(self) -> None:
        for decision in ("WAIT", "SELL", "HOLD", "AVOID LONG"):
            p = apply_entry_quality_gates(_payload(final_decision=decision, rr_ratio=0.1, rr_score=0, volume_score=0))
            self.assertEqual(p["final_decision"], decision)
            self.assertEqual(p["entry_decision"], decision)
            self.assertIsNone(p["gate_triggered"])

    def test_existing_reason_preserved_when_no_gate_fires(self) -> None:
        p = apply_entry_quality_gates(_payload(final_decision="WAIT", final_decision_reason="original reason"))
        self.assertEqual(p["final_decision_reason"], "original reason")


class TrendBiasTests(unittest.TestCase):
    def test_bearish_alignment_overrides_high_regime_score(self) -> None:
        p = apply_entry_quality_gates(
            _payload(final_decision="WAIT", multi_timeframe={"alignment": "BEARISH_ALIGNMENT"}, regime_score=10)
        )
        self.assertEqual(p["trend_bias"], "BEARISH")

    def test_regime_score_used_when_alignment_neutral(self) -> None:
        p = apply_entry_quality_gates(_payload(multi_timeframe={"alignment": "MIXED_ALIGNMENT"}, regime_score=8))
        self.assertEqual(p["trend_bias"], "BULLISH")

    def test_missing_regime_score_does_not_crash(self) -> None:
        p = apply_entry_quality_gates(_payload(multi_timeframe=None, regime_score=None))
        self.assertEqual(p["trend_bias"], "NEUTRAL")


class NullSafetyTests(unittest.TestCase):
    def test_watch_levels_null_safe_in_alert_summary(self) -> None:
        p = apply_entry_quality_gates(_payload(rr_ratio=0.8, support=None, ema20=None, resistance=None))
        self.assertEqual(p["entry_decision"], "WAIT_FOR_PULLBACK")
        self.assertEqual(p["watch_levels"]["pullback_entry"], {"low": None, "high": None})
        self.assertIn("Watch pullback n/a–n/a or breakout above n/a", p["alert_summary"])


if __name__ == "__main__":
    unittest.main()
