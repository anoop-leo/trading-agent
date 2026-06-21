import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from risk.portfolio_state import (
    PortfolioState,
    apply_decision_to_state,
    load_portfolio_state,
    save_portfolio_state,
)
from risk.risk_config import RiskEngineConfig


class PortfolioStateTests(unittest.TestCase):
    def test_missing_path_defaults_from_config_targets(self) -> None:
        config = RiskEngineConfig(total_portfolio_value_usd=100_000.0)
        with TemporaryDirectory() as temp_dir:
            state = load_portfolio_state(Path(temp_dir) / "missing.json", config)

        self.assertEqual(state.total_value_usd, 100_000.0)
        self.assertEqual(state.core_usd, 45_000.0)
        self.assertEqual(state.growth_usd, 30_000.0)
        self.assertEqual(state.speculative_usd, 12_000.0)
        self.assertEqual(state.cash_usd, 13_000.0)
        self.assertEqual(state.drawdown_pct, 0.0)

    def test_save_and_load_round_trip(self) -> None:
        state = PortfolioState(
            total_value_usd=330_000, peak_value_usd=400_000,
            cash_usd=42_900, core_usd=148_500, growth_usd=99_000, speculative_usd=39_600,
        )
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "portfolio_state.json"
            save_portfolio_state(state, path)
            loaded = load_portfolio_state(path)

        self.assertEqual(loaded, state)
        self.assertAlmostEqual(loaded.drawdown_pct, 17.5)

    def test_rejects_negative_values(self) -> None:
        with self.assertRaises(ValueError):
            PortfolioState(total_value_usd=-1, peak_value_usd=0, cash_usd=0, core_usd=0, growth_usd=0, speculative_usd=0)

    def test_bucket_usd_returns_requested_bucket(self) -> None:
        state = PortfolioState(
            total_value_usd=10_000, peak_value_usd=10_000,
            cash_usd=1_000, core_usd=5_000, growth_usd=3_000, speculative_usd=1_000,
        )
        self.assertEqual(state.bucket_usd("growth"), 3_000)
        with self.assertRaises(ValueError):
            state.bucket_usd("cash")

    def test_apply_decision_buy_moves_cash_into_bucket(self) -> None:
        state = PortfolioState(
            total_value_usd=10_000, peak_value_usd=10_000,
            cash_usd=2_000, core_usd=5_000, growth_usd=2_000, speculative_usd=1_000,
        )
        decision_dict = {
            "recommendation": {"bucket": "growth", "action": "buy"},
            "status": "approved",
            "approved_size_usd": 500.0,
        }
        updated = apply_decision_to_state(state, decision_dict)
        self.assertEqual(updated.growth_usd, 2_500.0)
        self.assertEqual(updated.cash_usd, 1_500.0)

    def test_apply_decision_blocked_is_noop(self) -> None:
        state = PortfolioState(
            total_value_usd=10_000, peak_value_usd=10_000,
            cash_usd=2_000, core_usd=5_000, growth_usd=2_000, speculative_usd=1_000,
        )
        decision_dict = {
            "recommendation": {"bucket": "growth", "action": "buy"},
            "status": "blocked",
            "approved_size_usd": 0.0,
        }
        self.assertEqual(apply_decision_to_state(state, decision_dict), state)

    def test_apply_decision_sell_returns_cash_to_buffer(self) -> None:
        state = PortfolioState(
            total_value_usd=10_000, peak_value_usd=10_000,
            cash_usd=2_000, core_usd=5_000, growth_usd=2_000, speculative_usd=1_000,
        )
        decision_dict = {
            "recommendation": {"bucket": "speculative", "action": "sell"},
            "status": "approved",
            "approved_size_usd": 300.0,
        }
        updated = apply_decision_to_state(state, decision_dict)
        self.assertEqual(updated.speculative_usd, 700.0)
        self.assertEqual(updated.cash_usd, 2_300.0)


if __name__ == "__main__":
    unittest.main()
