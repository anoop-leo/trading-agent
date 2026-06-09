import unittest

from scoring.setup_detection_skill import Setup, SetupInput, detect_setup


def setup_input(**overrides) -> SetupInput:
    values = {
        "price": 100.0,
        "ema20": 98.0,
        "ema50": 95.0,
        "ema200": 90.0,
        "support": 90.0,
        "resistance": 120.0,
        "volume_ratio": 1.0,
        "trend_score": 5,
        "momentum_score": 5,
        "bottom_score": 5,
        "market_regime": "NEUTRAL",
        "rsi": 55.0,
    }
    values.update(overrides)
    return SetupInput(**values)


class SetupDetectionTests(unittest.TestCase):
    def test_bottoming(self) -> None:
        result = detect_setup(
            setup_input(
                price=91.0,
                ema20=100.0,
                ema50=105.0,
                ema200=110.0,
                support=90.0,
                resistance=120.0,
                trend_score=5,
                momentum_score=4,
                bottom_score=8,
                market_regime="BEAR",
                rsi=42.0,
            )
        )

        self.assertEqual(result.setup, Setup.BOTTOMING)
        self.assertEqual(result.setup_score, 8)
        self.assertEqual(result.setup_confidence, 72)
        self.assertIn("Price near support", result.setup_reason)

    def test_breakout_has_highest_priority(self) -> None:
        result = detect_setup(
            setup_input(
                price=125.0,
                resistance=120.0,
                volume_ratio=1.8,
                momentum_score=7,
                trend_score=9,
                bottom_score=8,
                market_regime="BULL",
            )
        )

        self.assertEqual(result.setup, Setup.BREAKOUT)

    def test_trend_following(self) -> None:
        result = detect_setup(
            setup_input(
                price=130.0,
                ema20=120.0,
                ema50=110.0,
                ema200=100.0,
                trend_score=9,
                momentum_score=6,
                market_regime="BULL",
            )
        )

        self.assertEqual(result.setup, Setup.TREND_FOLLOWING)

    def test_pullback(self) -> None:
        result = detect_setup(
            setup_input(
                price=112.0,
                ema20=115.0,
                ema50=110.0,
                ema200=90.0,
                trend_score=7,
                momentum_score=6,
                market_regime="BULL",
            )
        )

        self.assertEqual(result.setup, Setup.PULLBACK)

    def test_range_bound(self) -> None:
        result = detect_setup(setup_input(trend_score=5, momentum_score=5, market_regime="NEUTRAL"))

        self.assertEqual(result.setup, Setup.RANGE_BOUND)

    def test_bear_trend(self) -> None:
        result = detect_setup(
            setup_input(
                price=80.0,
                ema20=90.0,
                ema50=95.0,
                ema200=100.0,
                trend_score=3,
                momentum_score=3,
                market_regime="BEAR",
            )
        )

        self.assertEqual(result.setup, Setup.BEAR_TREND)


if __name__ == "__main__":
    unittest.main()
