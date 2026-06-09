import unittest

from decision.decision_engine import Decision, DecisionInput, make_decision


def make_input(**overrides) -> DecisionInput:
    values = {
        "symbol": "BTCUSDT",
        "trend_score": 5,
        "momentum_score": 5,
        "volume_score": 5,
        "bottom_score": 5,
        "sr_score": 5,
        "rr_score": 4,
        "regime_score": 5,
        "current_price": 100.0,
        "ema20": 98.0,
        "ema50": 95.0,
        "ema200": 90.0,
        "recent_swing_high": 130.0,
        "recent_swing_low": 90.0,
        "rr_ratio": 1.5,
        "market_regime": "NEUTRAL",
        "support": 90.0,
        "volume_ratio": 1.0,
        "rsi": 55.0,
        "macd": "bullish",
    }
    values.update(overrides)
    return DecisionInput(**values)


class DecisionEngineV2RuleTests(unittest.TestCase):
    def test_no_position_high_score_rr_and_bull_regime_returns_buy(self) -> None:
        result = make_decision(
            make_input(
                trend_score=10,
                momentum_score=10,
                volume_score=10,
                bottom_score=10,
                sr_score=10,
                rr_score=10,
                regime_score=10,
                rr_ratio=3.0,
                market_regime="BULL",
            )
        )

        self.assertEqual(result.decision, Decision.BUY)

    def test_buy_requires_overall_score_at_least_70(self) -> None:
        result = make_decision(
            make_input(
                trend_score=7,
                momentum_score=7,
                volume_score=7,
                bottom_score=7,
                sr_score=7,
                rr_score=7,
                regime_score=7,
            )
        )

        self.assertEqual(result.decision, Decision.BUY)

    def test_watch_for_reversal_uses_bottom_score(self) -> None:
        result = make_decision(make_input(bottom_score=8, trend_score=4, momentum_score=4, volume_score=4))

        self.assertEqual(result.decision, Decision.WATCH_FOR_REVERSAL)

    def test_no_position_middle_overall_score_waits(self) -> None:
        result = make_decision(make_input())

        self.assertEqual(result.decision, Decision.WAIT)

    def test_no_position_overall_score_below_40_avoids_long(self) -> None:
        result = make_decision(
            make_input(
                trend_score=3,
                momentum_score=3,
                volume_score=3,
                bottom_score=3,
                sr_score=3,
                rr_score=4,
                regime_score=5,
            )
        )

        self.assertEqual(result.decision, Decision.AVOID_LONG)

    def test_no_position_bear_regime_and_weak_trend_avoids_long(self) -> None:
        result = make_decision(
            make_input(
                trend_score=3,
                momentum_score=3,
                volume_score=5,
                bottom_score=4,
                market_regime="BEAR",
                regime_score=0,
            )
        )

        self.assertEqual(result.decision, Decision.AVOID_LONG)

    def test_bottom_score_10_bear_regime_watches_for_reversal(self) -> None:
        result = make_decision(
            make_input(
                trend_score=3,
                momentum_score=3,
                volume_score=7,
                bottom_score=10,
                market_regime="BEAR",
                regime_score=0,
            )
        )

        self.assertEqual(result.decision, Decision.WATCH_FOR_REVERSAL)
        self.assertNotEqual(result.decision, Decision.AVOID_LONG)
        self.assertGreaterEqual(result.confidence, 55)
        self.assertLessEqual(result.confidence, 65)
        self.assertIn(
            "Market is bearish, but bottom detection is strong. Wait for confirmation before buying.",
            result.rationale,
        )

    def test_bottom_score_at_or_below_4_bear_regime_avoids_long_without_position(self) -> None:
        result = make_decision(
            make_input(
                trend_score=3,
                momentum_score=3,
                volume_score=5,
                bottom_score=4,
                market_regime="BEAR",
                regime_score=0,
            )
        )

        self.assertEqual(result.decision, Decision.AVOID_LONG)

    def test_bottom_score_at_or_below_4_bear_regime_can_exit_when_holding(self) -> None:
        result = make_decision(
            make_input(
                position_mode="HOLDING",
                trend_score=3,
                momentum_score=3,
                volume_score=5,
                bottom_score=4,
                market_regime="BEAR",
                regime_score=0,
            )
        )

        self.assertEqual(result.decision, Decision.EXIT)

    def test_bottom_score_8_returns_watch_for_reversal_unless_buy_conditions_are_met(self) -> None:
        reversal = make_decision(
            make_input(
                trend_score=4,
                momentum_score=4,
                volume_score=4,
                bottom_score=8,
            )
        )
        buy = make_decision(
            make_input(
                trend_score=8,
                momentum_score=8,
                volume_score=8,
                bottom_score=8,
                sr_score=8,
                rr_score=8,
                regime_score=8,
            )
        )

        self.assertEqual(reversal.decision, Decision.WATCH_FOR_REVERSAL)
        self.assertEqual(buy.decision, Decision.BUY)

    def test_no_position_bearish_trend_without_bottom_signal_avoids_long(self) -> None:
        result = make_decision(
            make_input(
                trend_score=4,
                momentum_score=4,
                volume_score=3,
                bottom_score=5,
                market_regime="BEAR",
                regime_score=0,
            )
        )

        self.assertEqual(result.decision, Decision.AVOID_LONG)

    def test_holding_bearish_trend_without_bottom_signal_reduces(self) -> None:
        result = make_decision(
            make_input(
                position_mode="HOLDING",
                trend_score=4,
                momentum_score=4,
                volume_score=3,
                bottom_score=5,
                market_regime="BEAR",
                regime_score=0,
            )
        )

        self.assertEqual(result.decision, Decision.REDUCE)

    def test_setup_breakout_maps_to_buy(self) -> None:
        self.assertEqual(make_decision(make_input(setup="BREAKOUT")).decision, Decision.BUY)

    def test_setup_trend_following_maps_to_buy(self) -> None:
        self.assertEqual(make_decision(make_input(setup="TREND_FOLLOWING")).decision, Decision.BUY)

    def test_setup_bottoming_maps_to_watch_for_reversal(self) -> None:
        self.assertEqual(make_decision(make_input(setup="BOTTOMING")).decision, Decision.WATCH_FOR_REVERSAL)

    def test_setup_pullback_maps_to_buy_watch(self) -> None:
        self.assertEqual(make_decision(make_input(setup="PULLBACK")).decision, Decision.BUY_WATCH)

    def test_setup_range_bound_maps_to_wait_without_position(self) -> None:
        self.assertEqual(make_decision(make_input(setup="RANGE_BOUND")).decision, Decision.WAIT)

    def test_setup_bear_trend_maps_to_avoid_long_without_position(self) -> None:
        self.assertEqual(make_decision(make_input(setup="BEAR_TREND")).decision, Decision.AVOID_LONG)

    def test_no_position_bear_trend_avoids_long(self) -> None:
        result = make_decision(make_input(setup="BEAR_TREND", market_regime="BEAR", trend_score=2))

        self.assertEqual(result.decision, Decision.AVOID_LONG)

    def test_holding_bear_trend_reduces_or_exits(self) -> None:
        result = make_decision(
            make_input(setup="BEAR_TREND", position_mode="HOLDING", market_regime="BEAR", trend_score=2)
        )

        self.assertIn(result.decision, {Decision.REDUCE, Decision.EXIT})

    def test_no_position_bottoming_watches_for_reversal(self) -> None:
        result = make_decision(make_input(setup="BOTTOMING", bottom_score=8, market_regime="BEAR"))

        self.assertEqual(result.decision, Decision.WATCH_FOR_REVERSAL)

    def test_holding_bullish_trend_holds_or_adds(self) -> None:
        result = make_decision(
            make_input(
                position_mode="HOLDING",
                setup="TREND_FOLLOWING",
                trend_score=8,
                momentum_score=7,
                volume_score=7,
                regime_score=10,
                market_regime="BULL",
            )
        )

        self.assertIn(result.decision, {Decision.HOLD, Decision.ADD})

    def test_rationale_notes_bottom_signal_lacks_volume_confirmation(self) -> None:
        result = make_decision(make_input(bottom_score=6, volume_ratio=0.79))

        self.assertIn("Bottom signal lacks volume confirmation.", result.rationale)


if __name__ == "__main__":
    unittest.main()
