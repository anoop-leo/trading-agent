import unittest

from backtesting.trend_holding_simulator import TrendHoldingSimulator
from strategy.trend_holding_engine import TrendState, determine_trend_state


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
        "target_1": None,
        "macd": "bullish",
    }
    values.update(overrides)
    return values


class TrendHoldingEngineTests(unittest.TestCase):
    def test_trend_state_detection(self) -> None:
        self.assertEqual(determine_trend_state(signal()), TrendState.STRONG_BULL)
        self.assertEqual(
            determine_trend_state(signal(daily_ema20=101.0, daily_ema50=100.0, daily_rsi=50.0)),
            TrendState.BULL,
        )
        self.assertEqual(determine_trend_state(signal(daily_price=99.0, daily_ema50=100.0)), TrendState.BEAR)
        self.assertEqual(determine_trend_state({"daily_price": 100.0}), TrendState.RANGE)

    def test_tp1_sells_50_percent_and_moves_stop_to_breakeven(self) -> None:
        simulator = TrendHoldingSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)

        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(signal(timestamp="2024-01-02T00:00:00+00:00", price=120.0))

        self.assertTrue(simulator.tp1_done)
        self.assertEqual(simulator.tp1_hit_count, 1)
        self.assertAlmostEqual(simulator.position_size, simulator.initial_position_size * 0.5)
        self.assertEqual(simulator.entry_stop_loss, simulator.entry_price)

    def test_tp2_sells_25_percent_and_activates_runner(self) -> None:
        simulator = TrendHoldingSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)

        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(signal(timestamp="2024-01-02T00:00:00+00:00", price=120.0))
        simulator.process_signal(signal(timestamp="2024-01-03T00:00:00+00:00", price=140.0))

        self.assertTrue(simulator.tp2_done)
        self.assertTrue(simulator.runner_active)
        self.assertEqual(simulator.tp2_hit_count, 1)
        self.assertEqual(simulator.runner_activation_count, 1)
        self.assertAlmostEqual(simulator.position_size, simulator.initial_position_size * 0.25)

    def test_runner_exits_on_daily_close_below_ema50(self) -> None:
        simulator = TrendHoldingSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(signal(timestamp="2024-01-02T00:00:00+00:00", price=120.0))
        simulator.process_signal(signal(timestamp="2024-01-03T00:00:00+00:00", price=140.0))
        simulator.process_signal(
            signal(timestamp="2024-01-04T00:00:00+00:00", price=145.0, daily_price=99.0, daily_ema50=100.0)
        )

        self.assertEqual(len(simulator.trades), 1)
        self.assertEqual(simulator.trades[0].exit_reason, "RUNNER_DAILY_EMA50_EXIT")

    def test_runner_exits_on_daily_macd_bearish(self) -> None:
        simulator = TrendHoldingSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(signal(timestamp="2024-01-02T00:00:00+00:00", price=120.0))
        simulator.process_signal(signal(timestamp="2024-01-03T00:00:00+00:00", price=140.0))
        simulator.process_signal(signal(timestamp="2024-01-04T00:00:00+00:00", price=145.0, daily_macd="bearish"))

        self.assertEqual(len(simulator.trades), 1)
        self.assertEqual(simulator.trades[0].exit_reason, "RUNNER_MACD_EXIT")

    def test_runner_exits_on_twenty_percent_trailing_stop(self) -> None:
        simulator = TrendHoldingSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(signal(timestamp="2024-01-02T00:00:00+00:00", price=120.0))
        simulator.process_signal(signal(timestamp="2024-01-03T00:00:00+00:00", price=150.0))
        simulator.process_signal(
            signal(
                timestamp="2024-01-04T00:00:00+00:00",
                price=119.0,
                daily_price=140.0,
                daily_ema50=100.0,
                daily_macd="bullish",
            )
        )

        self.assertEqual(len(simulator.trades), 1)
        self.assertEqual(simulator.trades[0].exit_reason, "RUNNER_TRAILING_STOP")


if __name__ == "__main__":
    unittest.main()

