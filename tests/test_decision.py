import unittest

from decision.decision_engine import Decision, DecisionError, DecisionInput, PriceZone, calculate_confidence, make_decision


def decision_input(**overrides) -> DecisionInput:
    values = {
        "symbol": "BTCUSDT",
        "trend_score": 5,
        "momentum_score": 5,
        "volume_score": 5,
        "bottom_score": 5,
        "sr_score": 5,
        "rr_score": 4,
        "regime_score": 5,
        "current_price": 63500.0,
        "ema20": 62800.0,
        "ema50": 65000.0,
        "ema200": 68000.0,
        "recent_swing_high": 70000.0,
        "recent_swing_low": 61800.0,
        "rr_ratio": 1.5,
        "market_regime": "NEUTRAL",
        "support": 61800.0,
        "rsi": 58.0,
        "macd": "bearish",
    }
    values.update(overrides)
    return DecisionInput(**values)


class DecisionEngineV2Tests(unittest.TestCase):
    def test_confidence_uses_four_scores(self) -> None:
        self.assertEqual(
            calculate_confidence(
                decision_input(
                    trend_score=9,
                    momentum_score=8,
                    volume_score=8,
                    bottom_score=7,
                    sr_score=8,
                    rr_score=7,
                    regime_score=10,
                )
            ),
            81,
        )

    def test_rejects_invalid_scores(self) -> None:
        with self.assertRaises(DecisionError):
            calculate_confidence(decision_input(trend_score=11))

    def test_rejects_invalid_position_mode(self) -> None:
        with self.assertRaises(DecisionError):
            make_decision(decision_input(position_mode="SHORTING"))

    def test_high_quality_no_position_signal_returns_buy(self) -> None:
        result = make_decision(
            decision_input(
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
        self.assertEqual(result.decision_meaning, "Potential long entry setup detected.")
        self.assertEqual(result.confidence, 100)
        self.assertEqual(result.entry_zone, PriceZone(low=62800.0, high=63500.0))
        self.assertEqual(result.stop_loss, 61800.0)
        self.assertEqual(result.target_1, 68000.0)
        self.assertEqual(result.target_2, 70000.0)

    def test_buy(self) -> None:
        result = make_decision(
            decision_input(
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
        self.assertEqual(result.confidence, 70)
        self.assertEqual(result.entry_zone, PriceZone(low=62800.0, high=63500.0))

    def test_watch_for_reversal(self) -> None:
        result = make_decision(decision_input(trend_score=4, momentum_score=4, volume_score=4, bottom_score=8))

        self.assertEqual(result.decision, Decision.WATCH_FOR_REVERSAL)
        self.assertEqual(result.entry_zone, PriceZone(low=61800.0, high=62800.0))
        self.assertEqual(result.stop_loss, 60873.0)
        self.assertEqual(result.target_1, 65000.0)
        self.assertEqual(result.target_2, 68000.0)
        self.assertIn("Bottom detection suggests early reversal formation", result.rationale)

    def test_no_position_mixed_signal_waits(self) -> None:
        result = make_decision(decision_input(trend_score=5, momentum_score=5, volume_score=5, bottom_score=5))

        self.assertEqual(result.decision, Decision.WAIT)
        self.assertIsNone(result.entry_zone)
        self.assertIsNone(result.stop_loss)
        self.assertIsNone(result.target_1)
        self.assertIsNone(result.target_2)

    def test_no_position_weak_signal_avoids_long(self) -> None:
        result = make_decision(
            decision_input(
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
        self.assertEqual(result.decision_meaning, "Do not open a new long position here.")
        self.assertIsNone(result.entry_zone)
        self.assertIsNone(result.stop_loss)

    def test_holding_strong_bearish_conditions_exit(self) -> None:
        result = make_decision(
            decision_input(
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
        self.assertIsNone(result.entry_zone)
        self.assertIsNone(result.stop_loss)

    def test_rationale_mentions_ema_alignment_and_momentum(self) -> None:
        result = make_decision(decision_input())

        self.assertIn("BTCUSDT is above EMA20 but below longer trend averages", result.rationale)
        self.assertIn("RSI is healthy and MACD is bearish", result.rationale)


if __name__ == "__main__":
    unittest.main()
