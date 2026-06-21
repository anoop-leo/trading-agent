import unittest

from decision.recommendation import PositionRecommendation, RiskDecision


def _recommendation(**overrides: object) -> PositionRecommendation:
    defaults = dict(
        symbol="BTC",
        asset_class="crypto",
        bucket="core",
        action="buy",
        conviction_score=0.8,
        suggested_size_usd=1000.0,
        rationale="test",
        source_agent="test_agent",
    )
    defaults.update(overrides)
    return PositionRecommendation(**defaults)


class PositionRecommendationTests(unittest.TestCase):
    def test_round_trips_through_dict(self) -> None:
        recommendation = _recommendation()
        self.assertEqual(PositionRecommendation.from_dict(recommendation.to_dict()), recommendation)

    def test_rejects_unsupported_asset_class(self) -> None:
        with self.assertRaises(ValueError):
            _recommendation(asset_class="commodity")

    def test_rejects_unsupported_bucket(self) -> None:
        with self.assertRaises(ValueError):
            _recommendation(bucket="satellite")

    def test_rejects_conviction_score_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            _recommendation(conviction_score=1.5)

    def test_rejects_negative_suggested_size(self) -> None:
        with self.assertRaises(ValueError):
            _recommendation(suggested_size_usd=-1.0)


class RiskDecisionTests(unittest.TestCase):
    def test_approved_decision_does_not_require_reason(self) -> None:
        decision = RiskDecision(_recommendation(), "approved", 1000.0)
        self.assertEqual(decision.reason, "")

    def test_adjusted_decision_requires_reason(self) -> None:
        with self.assertRaises(ValueError):
            RiskDecision(_recommendation(), "adjusted", 500.0)

    def test_blocked_decision_requires_reason(self) -> None:
        with self.assertRaises(ValueError):
            RiskDecision(_recommendation(), "blocked", 0.0)

    def test_round_trips_through_dict(self) -> None:
        decision = RiskDecision(_recommendation(), "adjusted", 500.0, "trimmed for cap")
        restored = RiskDecision.from_dict(decision.to_dict())
        self.assertEqual(restored.status, "adjusted")
        self.assertEqual(restored.approved_size_usd, 500.0)
        self.assertEqual(restored.recommendation, decision.recommendation)


if __name__ == "__main__":
    unittest.main()
