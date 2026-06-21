import unittest

from scoring.equity_investor_score import (
    EQUITY_INVESTOR_WEIGHTS,
    band_for_equity_investor_score,
    calculate_equity_investor_score,
    factor_payload,
    score_distance_from_200d_ma,
    score_fcf_yield,
    score_growth_consistency,
    score_quality_roe,
    score_valuation_pb,
    score_valuation_pe,
    score_valuation_peg,
)


class ScoreFunctionTests(unittest.TestCase):
    def test_score_valuation_pe_rewards_cheap_earnings(self) -> None:
        self.assertEqual(score_valuation_pe(10.0), 20)
        self.assertEqual(score_valuation_pe(40.0), 0)
        self.assertEqual(score_valuation_pe(None), 0)
        self.assertEqual(score_valuation_pe(-5.0), 0)

    def test_score_valuation_peg_rewards_below_one(self) -> None:
        self.assertEqual(score_valuation_peg(0.8), 15)
        self.assertEqual(score_valuation_peg(3.0), 2)
        self.assertEqual(score_valuation_peg(None), 0)

    def test_score_valuation_pb(self) -> None:
        self.assertEqual(score_valuation_pb(0.5), 10)
        self.assertEqual(score_valuation_pb(10.0), 1)

    def test_score_fcf_yield_rewards_high_yield(self) -> None:
        self.assertEqual(score_fcf_yield(9.0), 20)
        self.assertEqual(score_fcf_yield(-2.0), 0)
        self.assertEqual(score_fcf_yield(None), 0)

    def test_score_quality_roe(self) -> None:
        self.assertEqual(score_quality_roe(0.25), 10)
        self.assertEqual(score_quality_roe(-0.1), 0)

    def test_score_growth_consistency_averages_available_values(self) -> None:
        self.assertEqual(score_growth_consistency(0.2, 0.2), 10)
        self.assertEqual(score_growth_consistency(None, None), 0)
        self.assertEqual(score_growth_consistency(-0.1, -0.1), 0)

    def test_score_distance_from_200d_ma_rewards_discount(self) -> None:
        self.assertEqual(score_distance_from_200d_ma(-40.0), 15)
        self.assertEqual(score_distance_from_200d_ma(50.0), 0)


class BandTests(unittest.TestCase):
    def test_band_thresholds(self) -> None:
        self.assertEqual(band_for_equity_investor_score(90).investor_band, "STRONG_ACCUMULATION_ZONE")
        self.assertEqual(band_for_equity_investor_score(75).investor_band, "ACCUMULATION_ZONE")
        self.assertEqual(band_for_equity_investor_score(60).investor_band, "NEUTRAL_WATCH_ZONE")
        self.assertEqual(band_for_equity_investor_score(45).investor_band, "WEAK_ACCUMULATION_ZONE")
        self.assertEqual(band_for_equity_investor_score(10).investor_band, "AVOID_ZONE")
        self.assertEqual(band_for_equity_investor_score(10).final_investor_action, "DO_NOT_ACCUMULATE")


class AggregationTests(unittest.TestCase):
    def test_calculate_equity_investor_score_sums_factor_scores(self) -> None:
        factor_scores = {key: factor_payload(None, weight, weight) for key, weight in EQUITY_INVESTOR_WEIGHTS.items()}
        self.assertEqual(calculate_equity_investor_score(factor_scores), sum(EQUITY_INVESTOR_WEIGHTS.values()))

    def test_weights_sum_to_100(self) -> None:
        self.assertEqual(sum(EQUITY_INVESTOR_WEIGHTS.values()), 100)


if __name__ == "__main__":
    unittest.main()
