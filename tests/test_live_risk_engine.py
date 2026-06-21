import unittest

from decision.recommendation import PositionRecommendation
from risk.live_risk_engine import LiveRiskEngine
from risk.portfolio_state import PortfolioState
from risk.risk_config import RiskEngineConfig


def _config(**overrides: object) -> RiskEngineConfig:
    defaults = dict(
        total_portfolio_value_usd=330_000.0,
        speculative_max_pct=12.0,
        cash_buffer_min_pct=10.0,
        growth_position_max_pct=5.0,
        speculative_position_max_pct=2.0,
        portfolio_drawdown_circuit_breaker_pct=25.0,
        portfolio_drawdown_recovery_pct=15.0,
    )
    defaults.update(overrides)
    return RiskEngineConfig(**defaults)


def _state(**overrides: object) -> PortfolioState:
    defaults = dict(
        total_value_usd=330_000.0, peak_value_usd=330_000.0,
        cash_usd=42_900.0, core_usd=148_500.0, growth_usd=99_000.0, speculative_usd=39_600.0,
    )
    defaults.update(overrides)
    return PortfolioState(**defaults)


def _buy(bucket: str, suggested_size_usd: float, symbol: str = "TEST") -> PositionRecommendation:
    return PositionRecommendation(
        symbol=symbol, asset_class="crypto", bucket=bucket, action="buy",
        conviction_score=0.8, suggested_size_usd=suggested_size_usd,
        rationale="test", source_agent="test",
    )


class LiveRiskEngineHoldSellTrimTests(unittest.TestCase):
    def test_hold_is_always_approved_at_zero(self) -> None:
        engine = LiveRiskEngine(_config())
        rec = PositionRecommendation(
            symbol="BTC", asset_class="crypto", bucket="core", action="hold",
            conviction_score=0.5, suggested_size_usd=0.0, rationale="test", source_agent="test",
        )
        decision = engine.evaluate(rec, _state())
        self.assertEqual(decision.status, "approved")
        self.assertEqual(decision.approved_size_usd, 0.0)

    def test_sell_passes_through_at_suggested_size(self) -> None:
        engine = LiveRiskEngine(_config())
        rec = PositionRecommendation(
            symbol="BTC", asset_class="crypto", bucket="core", action="sell",
            conviction_score=0.5, suggested_size_usd=2_000.0, rationale="test", source_agent="test",
        )
        decision = engine.evaluate(rec, _state())
        self.assertEqual(decision.status, "approved")
        self.assertEqual(decision.approved_size_usd, 2_000.0)


class LiveRiskEngineBuyCapTests(unittest.TestCase):
    def test_growth_buy_under_all_caps_is_approved_in_full(self) -> None:
        engine = LiveRiskEngine(_config())
        decision = engine.evaluate(_buy("growth", 1_000.0), _state())
        self.assertEqual(decision.status, "approved")
        self.assertEqual(decision.approved_size_usd, 1_000.0)

    def test_growth_buy_above_single_position_cap_is_trimmed(self) -> None:
        engine = LiveRiskEngine(_config(growth_position_max_pct=5.0))
        state = _state(cash_usd=200_000.0)
        decision = engine.evaluate(_buy("growth", 50_000.0), state)
        self.assertEqual(decision.status, "adjusted")
        self.assertEqual(decision.approved_size_usd, 16_500.0)
        self.assertIn("single-position cap", decision.reason)

    def test_speculative_buy_blocked_when_bucket_already_at_cap(self) -> None:
        engine = LiveRiskEngine(_config())
        state = _state(speculative_usd=39_600.0)
        decision = engine.evaluate(_buy("speculative", 1_000.0), state)
        self.assertEqual(decision.status, "blocked")
        self.assertEqual(decision.approved_size_usd, 0.0)

    def test_speculative_buy_partially_trimmed_to_remaining_bucket_room(self) -> None:
        engine = LiveRiskEngine(_config())
        state = _state(speculative_usd=35_000.0)
        decision = engine.evaluate(_buy("speculative", 5_000.0), state)
        self.assertEqual(decision.status, "adjusted")
        self.assertEqual(decision.approved_size_usd, 4_600.0)

    def test_buy_trimmed_to_preserve_cash_buffer(self) -> None:
        engine = LiveRiskEngine(_config(growth_position_max_pct=100.0))
        state = _state(cash_usd=35_000.0)
        decision = engine.evaluate(_buy("growth", 10_000.0), state)
        self.assertEqual(decision.status, "adjusted")
        self.assertEqual(decision.approved_size_usd, 2_000.0)
        self.assertIn("cash buffer", decision.reason)

    def test_buy_blocked_when_no_cash_room_remains(self) -> None:
        engine = LiveRiskEngine(_config(growth_position_max_pct=100.0))
        state = _state(cash_usd=33_000.0)
        decision = engine.evaluate(_buy("growth", 10_000.0), state)
        self.assertEqual(decision.status, "blocked")
        self.assertEqual(decision.approved_size_usd, 0.0)


class LiveRiskEngineDrawdownCircuitBreakerTests(unittest.TestCase):
    def test_non_core_buy_blocked_above_drawdown_threshold(self) -> None:
        engine = LiveRiskEngine(_config())
        state = _state(total_value_usd=240_000.0, peak_value_usd=330_000.0)
        decision = engine.evaluate(_buy("growth", 1_000.0), state)
        self.assertEqual(decision.status, "blocked")
        self.assertIn("circuit breaker", decision.reason)

    def test_core_buy_still_allowed_above_drawdown_threshold(self) -> None:
        engine = LiveRiskEngine(_config())
        state = _state(total_value_usd=240_000.0, peak_value_usd=330_000.0)
        decision = engine.evaluate(_buy("core", 1_000.0), state)
        self.assertEqual(decision.status, "approved")
        self.assertEqual(decision.approved_size_usd, 1_000.0)

    def test_buy_allowed_below_drawdown_threshold(self) -> None:
        engine = LiveRiskEngine(_config())
        state = _state(total_value_usd=260_000.0, peak_value_usd=330_000.0)
        decision = engine.evaluate(_buy("growth", 1_000.0), state)
        self.assertEqual(decision.status, "approved")


class LiveRiskEnginePositionSizeForTradeTests(unittest.TestCase):
    def test_equity_gap_buffer_reduces_size_versus_crypto(self) -> None:
        engine = LiveRiskEngine(_config(equity_gap_buffer_pct=10.0))
        crypto_result = engine.position_size_for_trade(
            entry_price=100.0, stop_price=90.0, available_cash=50_000.0, asset_class="crypto",
        )
        equity_result = engine.position_size_for_trade(
            entry_price=100.0, stop_price=90.0, available_cash=50_000.0, asset_class="equity",
        )
        self.assertLess(equity_result.position_size, crypto_result.position_size)

    def test_no_stop_price_returns_zero_position(self) -> None:
        engine = LiveRiskEngine(_config())
        result = engine.position_size_for_trade(
            entry_price=100.0, stop_price=None, available_cash=50_000.0, asset_class="crypto",
        )
        self.assertEqual(result.position_size, 0.0)


if __name__ == "__main__":
    unittest.main()
