import unittest

import pandas as pd

from scoring.support_resistance_skill import calculate_support_resistance


class SupportResistanceSkillTests(unittest.TestCase):
    def test_uses_most_recent_meaningful_swing_levels(self) -> None:
        frame = pd.DataFrame(
            {
                "high": [105, 108, 112, 106, 110, 107, 111],
                "low": [100, 98, 95, 99, 97, 94, 96],
            }
        )

        result = calculate_support_resistance(frame, current_price=100.0, swing_window=1)

        self.assertEqual(result.support, 94.0)
        self.assertEqual(result.resistance, 110.0)
        self.assertEqual(result.distance_to_support, 6.0)
        self.assertEqual(result.distance_to_resistance, 10.0)
        self.assertEqual(result.sr_score, 5)

    def test_scores_near_support_highly(self) -> None:
        frame = pd.DataFrame({"high": [130, 125, 130], "low": [90, 92, 91]})

        result = calculate_support_resistance(frame, current_price=92.0, swing_window=1)

        self.assertEqual(result.sr_score, 10)

    def test_scores_near_resistance_as_zero(self) -> None:
        frame = pd.DataFrame({"high": [130, 125, 130], "low": [90, 92, 91]})

        result = calculate_support_resistance(frame, current_price=128.0, swing_window=1)

        self.assertEqual(result.sr_score, 0)


if __name__ == "__main__":
    unittest.main()
