import unittest

from scoring.margin_of_safety import calculate_margin_of_safety_score, margin_of_safety_band


class MarginOfSafetyTests(unittest.TestCase):
    def test_normalizes_margin_of_safety_factors(self) -> None:
        factor_scores = {
            "distance_from_200d_ma": {"score": 20, "weight": 20},
            "mvrv": {"score": 10, "weight": 20},
            "drawdown_from_cycle_high": {"score": 5, "weight": 10},
            "fear_and_greed": {"score": 15, "weight": 15},
        }

        self.assertEqual(calculate_margin_of_safety_score(factor_scores), 77)

    def test_margin_bands(self) -> None:
        self.assertEqual(margin_of_safety_band(10), "NO_MARGIN_OF_SAFETY")
        self.assertEqual(margin_of_safety_band(40), "SOME_MARGIN_OF_SAFETY")
        self.assertEqual(margin_of_safety_band(70), "GOOD_MARGIN_OF_SAFETY")
        self.assertEqual(margin_of_safety_band(90), "RARE_MARGIN_OF_SAFETY")


if __name__ == "__main__":
    unittest.main()
