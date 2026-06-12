import unittest

from backtesting.regime_gated_trend_holding_simulator import RegimeGatedTrendHoldingSimulator
from strategy.trend_holding_engine import MacroRegime, determine_macro_regime


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
    }
    values.update(overrides)
    return values


class RegimeGatedTrendHoldingTests(unittest.TestCase):
    def test_macro_regime_detection(self) -> None:
        self.assertEqual(determine_macro_regime(signal()), MacroRegime.STRONG_BULL)
        self.assertEqual(determine_macro_regime(signal(weekly_price=95.0, weekly_ema20=100.0)), MacroRegime.BEAR)
        self.assertEqual(determine_macro_regime(signal(weekly_price=None, weekly_ema20=None)), MacroRegime.BULL)
        self.assertEqual(determine_macro_regime(signal(daily_rsi=50.0)), MacroRegime.RANGE)
        self.assertEqual(
            determine_macro_regime(signal(daily_ema20=80.0, daily_ema50=90.0, daily_ema200=100.0)),
            MacroRegime.BEAR,
        )

    def test_runner_activation_only_in_strong_bull(self) -> None:
        simulator = RegimeGatedTrendHoldingSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(signal(timestamp="2024-01-02T00:00:00+00:00", price=120.0))
        simulator.process_signal(signal(timestamp="2024-01-03T00:00:00+00:00", price=140.0))

        self.assertEqual(simulator.runner_activation_count, 1)
        self.assertTrue(simulator.runner_active)

        non_strong = RegimeGatedTrendHoldingSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        non_strong.process_signal(
            signal(final_decision="BUY", price=100.0, stop_loss=90.0, weekly_price=95.0, weekly_ema20=100.0)
        )
        non_strong.process_signal(signal(timestamp="2024-01-03T00:00:00+00:00", price=140.0, weekly_price=95.0))

        self.assertEqual(non_strong.runner_activation_count, 0)
        self.assertFalse(non_strong.runner_active)

    def test_runner_disabled_after_twenty_percent_portfolio_drawdown(self) -> None:
        simulator = RegimeGatedTrendHoldingSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.portfolio_peak_equity = 13000.0

        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(signal(timestamp="2024-01-03T00:00:00+00:00", price=140.0))

        self.assertFalse(simulator.trend_holding_allowed)
        self.assertEqual(simulator.runner_activation_count, 0)
        self.assertEqual(simulator.runner_disabled_count, 1)

    def test_runner_size_reduced_after_fifteen_percent_portfolio_drawdown(self) -> None:
        simulator = RegimeGatedTrendHoldingSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.portfolio_peak_equity = 12000.0

        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(signal(timestamp="2024-01-02T00:00:00+00:00", price=120.0))
        simulator.process_signal(signal(timestamp="2024-01-03T00:00:00+00:00", price=140.0))

        self.assertEqual(simulator.runner_activation_count, 1)
        self.assertAlmostEqual(simulator.active_runner_fraction, 0.125)
        self.assertAlmostEqual(simulator.position_size, simulator.initial_position_size * 0.125)

    def test_standard_exits_outside_strong_bull(self) -> None:
        simulator = RegimeGatedTrendHoldingSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.process_signal(
            signal(
                final_decision="BUY",
                price=100.0,
                stop_loss=90.0,
                target_1=110.0,
                weekly_price=95.0,
                weekly_ema20=100.0,
            )
        )
        simulator.process_signal(
            signal(
                timestamp="2024-01-04T01:00:00+00:00",
                price=111.0,
                target_1=110.0,
                weekly_price=95.0,
                weekly_ema20=100.0,
            )
        )

        self.assertEqual(len(simulator.trades), 1)
        self.assertEqual(simulator.trades[0].exit_reason, "TAKE_PROFIT")
        self.assertEqual(simulator.runner_activation_count, 0)


if __name__ == "__main__":
    unittest.main()

