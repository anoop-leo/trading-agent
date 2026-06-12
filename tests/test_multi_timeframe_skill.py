import unittest

from decision.decision_engine import Decision, apply_multi_timeframe_alignment
from scoring.multi_timeframe_skill import Alignment, TimeframeSignal, analyze_multi_timeframe, chart_annotation


def signal(timeframe: str, setup: str, **overrides) -> TimeframeSignal:
    values = {
        "timeframe": timeframe,
        "trend_score": 5,
        "momentum_score": 5,
        "volume_score": 5,
        "bottom_score": 5,
        "sr_score": 5,
        "rr_score": 4,
        "regime_score": 5,
        "setup": setup,
        "setup_confidence": 60,
        "decision": "WAIT",
        "price": 100.0,
        "rsi": 50.0,
        "macd": "neutral",
        "ema20": 99.0,
        "ema50": 98.0,
        "ema200": 97.0,
        "market_regime": "NEUTRAL",
    }
    values.update(overrides)
    return TimeframeSignal(**values)


class MultiTimeframeSkillTests(unittest.TestCase):
    def test_bullish_alignment(self) -> None:
        result = analyze_multi_timeframe(
            {
                "1h": signal("1h", "TREND_FOLLOWING", market_regime="BULL", volume_score=8),
                "4h": signal("4h", "BREAKOUT", market_regime="BULL", volume_score=8),
                "1d": signal("1d", "RANGE_BOUND", market_regime="NEUTRAL", volume_score=8),
            }
        )

        self.assertEqual(result.alignment, Alignment.BULLISH_ALIGNMENT)
        self.assertGreaterEqual(result.alignment_score, 80)
        self.assertIn("Long setups are allowed", result.summary)

    def test_bearish_alignment(self) -> None:
        result = analyze_multi_timeframe(
            {
                "1h": signal("1h", "BEAR_TREND", market_regime="BEAR"),
                "4h": signal("4h", "BEAR_TREND", market_regime="BEAR"),
                "1d": signal("1d", "BEAR_TREND", market_regime="BEAR"),
            }
        )

        self.assertEqual(result.alignment, Alignment.BEARISH_ALIGNMENT)
        self.assertLessEqual(result.alignment_score, 30)
        self.assertEqual(result.summary, "All major timeframes are bearish. Avoid long trades.")

    def test_reversal_forming(self) -> None:
        result = analyze_multi_timeframe(
            {
                "1h": signal("1h", "BOTTOMING", bottom_score=8, setup_confidence=72, market_regime="BEAR"),
                "4h": signal("4h", "BEAR_TREND", market_regime="BEAR"),
                "1d": signal("1d", "BEAR_TREND", market_regime="BEAR"),
            }
        )

        self.assertEqual(result.alignment, Alignment.REVERSAL_FORMING)
        self.assertGreaterEqual(result.alignment_score, 40)
        self.assertLessEqual(result.alignment_score, 60)
        self.assertEqual(result.summary, "1h is bottoming, but 4h and 1d remain bearish.")

    def test_mixed_alignment(self) -> None:
        result = analyze_multi_timeframe(
            {
                "1h": signal("1h", "TREND_FOLLOWING", market_regime="BULL"),
                "4h": signal("4h", "RANGE_BOUND", market_regime="NEUTRAL"),
                "1d": signal("1d", "PULLBACK", market_regime="BULL"),
            }
        )

        self.assertEqual(result.alignment, Alignment.MIXED_ALIGNMENT)
        self.assertGreaterEqual(result.alignment_score, 40)
        self.assertLessEqual(result.alignment_score, 60)

    def test_pullback_in_uptrend(self) -> None:
        result = analyze_multi_timeframe(
            {
                "1h": signal("1h", "RANGE_BOUND", market_regime="NEUTRAL", momentum_score=4, price=96.0),
                "4h": signal("4h", "PULLBACK", market_regime="BULL", trend_score=7),
                "1d": signal("1d", "TREND_FOLLOWING", market_regime="BULL", trend_score=9),
            }
        )

        self.assertEqual(result.alignment, Alignment.PULLBACK_IN_UPTREND)
        self.assertGreaterEqual(result.alignment_score, 60)
        self.assertLessEqual(result.alignment_score, 80)
        self.assertEqual(
            result.summary,
            "Daily trend is bullish, 4h is pulling back, and 1h shows weakness. Watch for continuation.",
        )

    def test_pullback_in_uptrend_requires_4h_pullback(self) -> None:
        result = analyze_multi_timeframe(
            {
                "1h": signal("1h", "RANGE_BOUND", market_regime="NEUTRAL", momentum_score=4, price=96.0),
                "4h": signal("4h", "RANGE_BOUND", market_regime="BULL", trend_score=6),
                "1d": signal("1d", "TREND_FOLLOWING", market_regime="BULL", trend_score=9),
            }
        )

        self.assertNotEqual(result.alignment, Alignment.PULLBACK_IN_UPTREND)

    def test_pullback_in_downtrend(self) -> None:
        result = analyze_multi_timeframe(
            {
                "1h": signal("1h", "TREND_FOLLOWING", market_regime="BULL", trend_score=8),
                "4h": signal("4h", "RANGE_BOUND", market_regime="NEUTRAL"),
                "1d": signal("1d", "BEAR_TREND", market_regime="BEAR", trend_score=2),
            }
        )

        self.assertEqual(result.alignment, Alignment.PULLBACK_IN_DOWNTREND)
        self.assertGreaterEqual(result.alignment_score, 25)
        self.assertLessEqual(result.alignment_score, 45)
        self.assertEqual(
            result.summary,
            "Daily trend is bearish while the short-term timeframe is bouncing. Avoid chasing long entries.",
        )

    def test_range_alignment(self) -> None:
        result = analyze_multi_timeframe(
            {
                "1h": signal("1h", "RANGE_BOUND"),
                "4h": signal("4h", "RANGE_BOUND"),
                "1d": signal("1d", "PULLBACK", market_regime="BULL"),
            }
        )

        self.assertEqual(result.alignment, Alignment.RANGE_ALIGNMENT)
        self.assertIn("range-bound", result.summary)

    def test_daily_timeframe_vetoes_weak_bullish_setup(self) -> None:
        result = analyze_multi_timeframe(
            {
                "1h": signal("1h", "TREND_FOLLOWING", market_regime="BULL"),
                "4h": signal("4h", "BREAKOUT", market_regime="BULL"),
                "1d": signal("1d", "RANGE_BOUND", market_regime="BEAR"),
            }
        )

        self.assertNotEqual(result.alignment, Alignment.BULLISH_ALIGNMENT)
        self.assertEqual(result.alignment, Alignment.PULLBACK_IN_DOWNTREND)

    def test_bearish_alignment_blocks_buy(self) -> None:
        result = apply_multi_timeframe_alignment(
            Decision.BUY,
            Alignment.BEARISH_ALIGNMENT.value,
            position_mode="NO_POSITION",
        )

        self.assertEqual(result.decision, Decision.AVOID_LONG)
        self.assertEqual(result.reason, "Major timeframes are bearish, so new long entries are blocked.")

    def test_reversal_forming_produces_watch_for_reversal(self) -> None:
        result = apply_multi_timeframe_alignment(
            Decision.BUY,
            Alignment.REVERSAL_FORMING.value,
            position_mode="NO_POSITION",
        )

        self.assertEqual(result.decision, Decision.WATCH_FOR_REVERSAL)
        self.assertEqual(result.reason, "Short-term bottoming detected, but higher timeframes remain bearish.")

    def test_pullback_in_uptrend_produces_buy_watch(self) -> None:
        result = apply_multi_timeframe_alignment(
            Decision.WAIT,
            Alignment.PULLBACK_IN_UPTREND.value,
            position_mode="NO_POSITION",
        )

        self.assertEqual(result.decision, Decision.BUY_WATCH)
        self.assertEqual(
            result.reason,
            "Higher timeframe uptrend remains intact while the short-term timeframe pulls back.",
        )

    def test_pullback_in_downtrend_avoids_long(self) -> None:
        result = apply_multi_timeframe_alignment(
            Decision.BUY,
            Alignment.PULLBACK_IN_DOWNTREND.value,
            position_mode="NO_POSITION",
        )

        self.assertEqual(result.decision, Decision.AVOID_LONG)
        self.assertEqual(result.reason, "Short-term bounce is occurring inside a higher-timeframe downtrend.")

    def test_chart_annotation_lists_timeframes_and_alignment(self) -> None:
        result = analyze_multi_timeframe(
            {
                "1h": signal("1h", "BOTTOMING", bottom_score=8, market_regime="BEAR"),
                "4h": signal("4h", "BEAR_TREND", market_regime="BEAR"),
                "1d": signal("1d", "BEAR_TREND", market_regime="BEAR"),
            }
        )

        self.assertEqual(
            chart_annotation(result),
            "1h: BOTTOMING\n4h: BEAR_TREND\n1d: BEAR_TREND\nAlignment: REVERSAL_FORMING",
        )


if __name__ == "__main__":
    unittest.main()
