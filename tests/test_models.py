import unittest

from trading_agent.models import SignalScores


class ModelTests(unittest.TestCase):
    def test_signal_scores_total(self) -> None:
        self.assertEqual(SignalScores(1, 2, 3, 4, 5, 6, 7).total, 28)


if __name__ == "__main__":
    unittest.main()
