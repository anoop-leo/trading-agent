import unittest

from scoring.accumulation_score import (
    band_for_accumulation_score,
    calculate_accumulation_score,
    score_distance_from_200d_ma,
    score_drawdown_from_cycle_high,
    score_fear_and_greed,
    score_liquidity_condition,
    score_monthly_trend,
    score_mvrv,
    score_thesis_risk,
    score_weekly_rsi,
)


class AccumulationScoreTests(unittest.TestCase):
    def test_scores_distance_from_200d_ma(self) -> None:
        self.assertEqual(score_distance_from_200d_ma(-30), 20)
        self.assertEqual(score_distance_from_200d_ma(-15), 16)
        self.assertEqual(score_distance_from_200d_ma(0), 12)
        self.assertEqual(score_distance_from_200d_ma(20), 6)
        self.assertEqual(score_distance_from_200d_ma(40), 2)
        self.assertEqual(score_distance_from_200d_ma(60), 0)

    def test_scores_external_factors(self) -> None:
        self.assertEqual(score_mvrv(0.9), 20)
        self.assertEqual(score_mvrv(None), 10)
        self.assertEqual(score_mvrv(1.5), 12)
        self.assertEqual(score_mvrv(1.7), 12)
        self.assertEqual(score_mvrv(3.5), 0)
        self.assertEqual(score_weekly_rsi(35), 12)
        self.assertEqual(score_fear_and_greed(15), 15)
        self.assertEqual(score_fear_and_greed(85), 0)
        self.assertEqual(score_monthly_trend("BELOW_EMA20"), 10)
        self.assertEqual(score_drawdown_from_cycle_high(-55), 8)
        self.assertEqual(score_liquidity_condition("NEUTRAL"), 3)
        self.assertEqual(score_thesis_risk("HIGH"), 0)

    def test_accumulation_band_and_high_risk_cap(self) -> None:
        band = band_for_accumulation_score(85, thesis_risk_level="LOW")

        self.assertEqual(band.accumulation_band, "AGGRESSIVE_ACCUMULATION")
        self.assertEqual(band.dca_intensity, "AGGRESSIVE_DCA")

        capped = band_for_accumulation_score(85, thesis_risk_level="HIGH")

        self.assertEqual(capped.accumulation_band, "FAIR")
        self.assertEqual(capped.suggested_dca_multiplier, "0.5x to 1.0x normal DCA")

    def test_calculates_total_score(self) -> None:
        self.assertEqual(
            calculate_accumulation_score(
                {
                    "a": {"score": 10},
                    "b": {"score": 5.4},
                }
            ),
            15,
        )


if __name__ == "__main__":
    unittest.main()
