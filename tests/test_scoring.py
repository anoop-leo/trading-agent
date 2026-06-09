import unittest

import pandas as pd

from trading_agent.scoring import (
    ScoringError,
    calculate_recent_swing_high,
    calculate_recent_swing_low,
    calculate_scores,
    calculate_volume_ratio,
    score_bottom,
    score_momentum,
    score_trend,
    score_volume,
)


class ScoringTests(unittest.TestCase):
    def test_score_trend_uses_price_above_emas(self) -> None:
        row = pd.Series({"close": 110.0, "ema_20": 100.0, "ema_50": 105.0, "ema_200": 120.0})

        self.assertEqual(score_trend(row), 5)

    def test_score_momentum_uses_rsi_and_macd(self) -> None:
        row = pd.Series({"rsi_14": 62.0, "macd": 3.0, "macd_signal": 2.0})

        self.assertEqual(score_momentum(row), 8)

    def test_score_momentum_rewards_strengthening_macd(self) -> None:
        row = pd.Series({"rsi_14": 62.0, "macd": 3.0, "macd_signal": 2.0, "macd_histogram": 1.0})
        previous_row = pd.Series({"macd_histogram": 0.5})

        self.assertEqual(score_momentum(row, previous_row), 10)

    def test_score_volume_scales_with_volume_ratio(self) -> None:
        self.assertEqual(calculate_volume_ratio(pd.Series({"volume": 180.0, "volume_ma_20": 100.0})), 1.8)
        self.assertEqual(score_volume(pd.Series({"volume": 180.0, "volume_ma_20": 100.0})), 7)
        self.assertEqual(score_volume(pd.Series({"volume": 201.0, "volume_ma_20": 100.0})), 10)
        self.assertEqual(score_volume(pd.Series({"volume": 100.0, "volume_ma_20": 100.0})), 0)

    def test_score_bottom_detects_early_reversal(self) -> None:
        row = pd.Series(
            {
                "open": 98.0,
                "close": 101.0,
                "rsi_14": 39.0,
                "macd_histogram": -0.1,
                "volume": 100.0,
                "volume_ma_20": 100.0,
            }
        )
        previous_row = pd.Series({"macd_histogram": -0.5})

        self.assertEqual(score_bottom(row, previous_row, recent_swing_low=100.0), 10)

    def test_score_bottom_caps_when_volume_lacks_participation(self) -> None:
        row = pd.Series(
            {
                "open": 98.0,
                "close": 101.0,
                "rsi_14": 39.0,
                "macd_histogram": -0.1,
                "volume": 79.0,
                "volume_ma_20": 100.0,
            }
        )
        previous_row = pd.Series({"macd_histogram": -0.5})

        self.assertEqual(score_bottom(row, previous_row, recent_swing_low=100.0), 6)

    def test_score_bottom_caps_more_aggressively_when_volume_is_very_low(self) -> None:
        row = pd.Series(
            {
                "open": 98.0,
                "close": 101.0,
                "rsi_14": 39.0,
                "macd_histogram": -0.1,
                "volume": 49.0,
                "volume_ma_20": 100.0,
            }
        )
        previous_row = pd.Series({"macd_histogram": -0.5})

        self.assertEqual(score_bottom(row, previous_row, recent_swing_low=100.0), 4)

    def test_recent_swing_levels_use_latest_window(self) -> None:
        frame = pd.DataFrame({"high": [10.0, 11.0, 12.0], "low": [8.0, 7.0, 9.0]})

        self.assertEqual(calculate_recent_swing_high(frame, window=2), 12.0)
        self.assertEqual(calculate_recent_swing_low(frame, window=2), 7.0)

    def test_calculate_scores_uses_latest_row(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "open": 51.0,
                    "high": 55.0,
                    "low": 49.0,
                    "close": 50.0,
                    "ema_20": 100.0,
                    "ema_50": 100.0,
                    "ema_200": 100.0,
                    "rsi_14": 20.0,
                    "macd": 1.0,
                    "macd_signal": 2.0,
                    "macd_histogram": -1.0,
                    "volume": 100.0,
                    "volume_ma_20": 100.0,
                },
                {
                    "open": 129.0,
                    "high": 132.0,
                    "low": 128.0,
                    "close": 130.0,
                    "ema_20": 100.0,
                    "ema_50": 110.0,
                    "ema_200": 120.0,
                    "rsi_14": 60.0,
                    "macd": 3.0,
                    "macd_signal": 2.0,
                    "macd_histogram": 1.0,
                    "volume": 180.0,
                    "volume_ma_20": 100.0,
                },
            ]
        )

        scores = calculate_scores(frame)

        self.assertEqual(scores.trend_score, 10)
        self.assertEqual(scores.momentum_score, 10)
        self.assertEqual(scores.volume_score, 7)
        self.assertEqual(scores.bottom_score, 3)

    def test_calculate_scores_requires_columns(self) -> None:
        with self.assertRaises(ScoringError):
            calculate_scores(pd.DataFrame([{"close": 1.0}]))


if __name__ == "__main__":
    unittest.main()
