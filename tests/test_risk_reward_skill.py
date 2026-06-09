import unittest

from scoring.risk_reward_skill import calculate_risk_reward


class RiskRewardSkillTests(unittest.TestCase):
    def test_calculates_risk_reward_and_ratio(self) -> None:
        result = calculate_risk_reward(current_price=100.0, support=90.0, resistance=130.0)

        self.assertEqual(result.risk, 10.0)
        self.assertEqual(result.reward, 30.0)
        self.assertEqual(result.rr_ratio, 3.0)
        self.assertEqual(result.rr_score, 10)

    def test_scores_ratio_buckets(self) -> None:
        self.assertEqual(calculate_risk_reward(100.0, 90.0, 95.0).rr_score, 0)
        self.assertEqual(calculate_risk_reward(100.0, 90.0, 115.0).rr_score, 4)
        self.assertEqual(calculate_risk_reward(100.0, 90.0, 120.0).rr_score, 7)
        self.assertEqual(calculate_risk_reward(100.0, 90.0, 130.0).rr_score, 10)


if __name__ == "__main__":
    unittest.main()
