import unittest

from backtesting.portfolio_governor_simulator import PortfolioGovernorSimulator
from risk.portfolio_risk_governor import (
    PortfolioRiskGovernor,
    RiskState,
    calculate_atr_moving_average,
    calculate_drawdown_pct,
)


def signal(**overrides):
    values = {
        "timestamp": "2024-01-01T00:00:00+00:00",
        "price": 100.0,
        "ema20": 95.0,
        "final_decision": "WAIT",
        "stop_loss": 90.0,
        "market_regime": "BULL",
        "trend_score": 8,
        "position_mode": "NO_POSITION",
        "alignment": "BULLISH_ALIGNMENT",
        "rr_ratio": 2.5,
        "volume_ratio": 1.2,
        "daily_setup": "TREND_FOLLOWING",
        "four_hour_price": 105.0,
        "four_hour_ema20": 100.0,
        "four_hour_macd": "bullish",
        "daily_price": 110.0,
        "daily_ema20": 105.0,
        "daily_ema50": 100.0,
        "daily_ema200": 90.0,
        "daily_rsi": 60.0,
        "daily_macd": "bullish",
        "weekly_price": 120.0,
        "weekly_ema20": 100.0,
        "target_1": None,
        "macd": "bullish",
        "atr": 4.0,
        "atr_ma": 4.0,
    }
    values.update(overrides)
    return values


class PortfolioRiskGovernorTests(unittest.TestCase):
    def test_drawdown_calculation(self) -> None:
        self.assertEqual(calculate_drawdown_pct(9000.0, 10000.0), 10.0)
        self.assertEqual(calculate_drawdown_pct(11000.0, 10000.0), 0.0)

    def test_risk_state_transitions(self) -> None:
        governor = PortfolioRiskGovernor(initial_equity=10000)

        self.assertEqual(governor.update_equity(9500).risk_state, RiskState.NORMAL.value)
        self.assertEqual(governor.update_equity(9000).risk_state, RiskState.CAUTION.value)
        self.assertEqual(governor.update_equity(8500).risk_state, RiskState.DEFENSIVE.value)
        self.assertEqual(governor.update_equity(8000).risk_state, RiskState.CAPITAL_PRESERVATION.value)

    def test_portfolio_stop_activation(self) -> None:
        governor = PortfolioRiskGovernor(initial_equity=10000)

        snapshot = governor.update_equity(7400)

        self.assertTrue(snapshot.portfolio_stop_active)
        self.assertTrue(snapshot.portfolio_stop_triggered)
        self.assertEqual(governor.portfolio_stop_count, 1)
        self.assertFalse(snapshot.runner_enabled)

    def test_volatility_sizing(self) -> None:
        governor = PortfolioRiskGovernor(initial_equity=10000)

        self.assertEqual(governor.volatility_factor(atr=100, atr_ma=200), 1.0)
        self.assertEqual(governor.volatility_factor(atr=200, atr_ma=100), 0.5)
        self.assertEqual(governor.volatility_factor(atr=1000, atr_ma=100), 0.25)

    def test_position_sizing_uses_one_percent_risk(self) -> None:
        governor = PortfolioRiskGovernor(initial_equity=10000)

        details = governor.position_size_details(
            entry_price=100.0,
            stop_price=90.0,
            available_cash=10000.0,
        )

        self.assertEqual(details.risk_amount, 100.0)
        self.assertEqual(details.position_size, 10.0)
        self.assertEqual(details.notional, 1000.0)
        self.assertEqual(details.position_size_fraction, 0.1)

    def test_defensive_mode_recovery(self) -> None:
        governor = PortfolioRiskGovernor(initial_equity=10000)
        governor.update_equity(7400)
        self.assertTrue(governor.portfolio_stop_active)

        snapshot = governor.update_equity(8600)

        self.assertFalse(snapshot.portfolio_stop_active)
        self.assertEqual(snapshot.risk_state, RiskState.CAUTION.value)

    def test_atr_moving_average_uses_recent_candles(self) -> None:
        import pandas as pd

        frame = pd.DataFrame(
            {
                "high": [11, 12, 13, 14, 15, 16],
                "low": [9, 10, 11, 12, 13, 14],
                "close": [10, 11, 12, 13, 14, 15],
            }
        )

        self.assertEqual(calculate_atr_moving_average(frame, atr_period=2, ma_period=2), 2.0)


class PortfolioGovernorSimulatorTests(unittest.TestCase):
    def test_defensive_state_disables_runner_activation(self) -> None:
        simulator = PortfolioGovernorSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.governor.peak_equity = 12000.0

        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(signal(timestamp="2024-01-04T00:00:00+00:00", price=140.0))

        self.assertFalse(simulator.trend_holding_allowed)
        self.assertEqual(simulator.runner_activation_count, 0)
        self.assertGreaterEqual(simulator.runner_disabled_count, 1)

    def test_portfolio_stop_closes_active_runner(self) -> None:
        simulator = PortfolioGovernorSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(signal(timestamp="2024-01-02T00:00:00+00:00", price=120.0))
        simulator.process_signal(signal(timestamp="2024-01-03T00:00:00+00:00", price=140.0))
        self.assertTrue(simulator.runner_active)

        simulator.governor.peak_equity = 50000.0
        simulator.process_signal(signal(timestamp="2024-01-04T00:00:00+00:00", price=130.0))

        self.assertEqual(len(simulator.trades), 1)
        self.assertEqual(simulator.trades[0].exit_reason, "PORTFOLIO_STOP")
        self.assertEqual(simulator.governor.portfolio_stop_count, 1)


if __name__ == "__main__":
    unittest.main()
