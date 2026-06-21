from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from decision.recommendation import PositionRecommendation, RiskDecision
from risk.risk_decision_log import append_risk_decision, read_risk_decision_log


class RiskDecisionLogTests(unittest.TestCase):
    def test_read_missing_log_returns_empty_list(self) -> None:
        with TemporaryDirectory() as temp_dir:
            self.assertEqual(read_risk_decision_log(Path(temp_dir) / "missing.jsonl"), [])

    def test_append_and_read_round_trips_multiple_entries(self) -> None:
        recommendation = PositionRecommendation(
            symbol="BTC", asset_class="crypto", bucket="core", action="buy",
            conviction_score=0.8, suggested_size_usd=1_000.0, rationale="test", source_agent="test",
        )
        decision_one = RiskDecision(recommendation, "approved", 1_000.0)
        decision_two = RiskDecision(recommendation, "blocked", 0.0, "drawdown breach")

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "risk_decision_log.jsonl"
            append_risk_decision(decision_one, path, generated_at="2026-01-01T00:00:00Z")
            append_risk_decision(decision_two, path, generated_at="2026-01-02T00:00:00Z")

            entries = read_risk_decision_log(path)

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["status"], "approved")
        self.assertEqual(entries[0]["generated_at"], "2026-01-01T00:00:00Z")
        self.assertEqual(entries[1]["status"], "blocked")
        self.assertEqual(entries[1]["reason"], "drawdown breach")

    def test_append_creates_parent_directory(self) -> None:
        recommendation = PositionRecommendation(
            symbol="ETH", asset_class="crypto", bucket="growth", action="hold",
            conviction_score=0.1, suggested_size_usd=0.0, rationale="test", source_agent="test",
        )
        decision = RiskDecision(recommendation, "approved", 0.0)
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "log.jsonl"
            append_risk_decision(decision, path)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
