import unittest
from dataclasses import replace

from backtesting.hybrid_trend_rider_simulator import HybridTrendRiderSimulator
from backtesting.trade_simulator import TradeSimulator
from backtesting.trend_rider_simulator import TrendRiderSimulator
from backtesting.profiles import get_strategy_profile


def signal(**overrides):
    values = {
        "timestamp": "2024-01-01T00:00:00+00:00",
        "price": 100.0,
        "ema20": 95.0,
        "final_decision": "WAIT",
        "stop_loss": None,
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
        "daily_ema20": 100.0,
        "daily_ema200": 90.0,
        "daily_rsi": 60.0,
        "target_1": None,
        "macd": "bullish",
    }
    values.update(overrides)
    return values


class TradeSimulatorTests(unittest.TestCase):
    def test_buy_opens_position_with_fee_and_slippage(self) -> None:
        simulator = TradeSimulator(initial_capital=10000, fee_rate=0.001, slippage_rate=0.0005)

        simulator.process_signal(signal(final_decision="BUY"))

        self.assertAlmostEqual(simulator.cash, 7500.0)
        self.assertGreater(simulator.position_size, 0.0)
        self.assertAlmostEqual(simulator.entry_price, 100.05)
        self.assertAlmostEqual(simulator.entry_fee, 2.4975024975024978)

    def test_hold_does_not_open_trade(self) -> None:
        simulator = TradeSimulator(initial_capital=10000)

        simulator.process_signal(signal(final_decision="HOLD"))

        self.assertEqual(simulator.position_size, 0.0)
        self.assertEqual(simulator.rejected_entry_reasons["not_buy_decision"], 1)

    def test_wait_does_not_open_trade(self) -> None:
        simulator = TradeSimulator(initial_capital=10000)

        simulator.process_signal(signal(final_decision="WAIT"))

        self.assertEqual(simulator.position_size, 0.0)
        self.assertEqual(simulator.rejected_entry_reasons["not_buy_decision"], 1)

    def test_add_does_not_open_trade_without_position(self) -> None:
        simulator = TradeSimulator(initial_capital=10000)

        simulator.process_signal(signal(final_decision="ADD"))

        self.assertEqual(simulator.position_size, 0.0)
        self.assertEqual(simulator.rejected_entry_reasons["not_buy_decision"], 1)

    def test_buy_watch_with_bearish_alignment_does_not_open_trade(self) -> None:
        simulator = TradeSimulator(initial_capital=10000)

        simulator.process_signal(signal(final_decision="BUY WATCH", alignment="BEARISH_ALIGNMENT"))

        self.assertEqual(simulator.position_size, 0.0)
        self.assertEqual(simulator.rejected_entry_reasons["bearish_alignment"], 1)

    def test_buy_watch_with_low_volume_does_not_open_trade(self) -> None:
        simulator = TradeSimulator(initial_capital=10000)

        simulator.process_signal(signal(final_decision="BUY WATCH", volume_ratio=0.9))

        self.assertEqual(simulator.position_size, 0.0)
        self.assertEqual(simulator.rejected_entry_reasons["low_volume_ratio"], 1)

    def test_buy_watch_with_low_rr_ratio_does_not_open_trade(self) -> None:
        simulator = TradeSimulator(initial_capital=10000)

        simulator.process_signal(signal(final_decision="BUY WATCH", rr_ratio=1.99))

        self.assertEqual(simulator.position_size, 0.0)
        self.assertEqual(simulator.rejected_entry_reasons["low_rr_ratio"], 1)

    def test_conservative_profile_requires_higher_rr_and_volume(self) -> None:
        simulator = TradeSimulator(
            initial_capital=10000,
            strategy_profile=get_strategy_profile("conservative"),
        )

        simulator.process_signal(signal(final_decision="BUY WATCH", rr_ratio=2.4, volume_ratio=1.1))

        self.assertEqual(simulator.position_size, 0.0)
        self.assertEqual(simulator.rejected_entry_reasons["low_rr_ratio"], 1)
        self.assertEqual(simulator.rejected_entry_reasons["low_volume_ratio"], 1)

    def test_balanced_profile_requires_4h_macd_bullish(self) -> None:
        simulator = TradeSimulator(initial_capital=10000)

        simulator.process_signal(signal(final_decision="BUY WATCH", four_hour_macd="bearish"))

        self.assertEqual(simulator.position_size, 0.0)
        self.assertEqual(simulator.rejected_entry_reasons["four_hour_macd_not_bullish"], 1)

    def test_aggressive_profile_does_not_require_4h_macd_bullish_or_1h_ema20(self) -> None:
        simulator = TradeSimulator(
            initial_capital=10000,
            strategy_profile=get_strategy_profile("aggressive"),
        )

        simulator.process_signal(
            signal(
                final_decision="BUY WATCH",
                rr_ratio=1.5,
                volume_ratio=0.8,
                four_hour_macd="bearish",
                ema20=105.0,
            )
        )

        self.assertGreater(simulator.position_size, 0.0)

    def test_balanced_profile_requires_price_above_1h_ema20(self) -> None:
        simulator = TradeSimulator(initial_capital=10000)

        simulator.process_signal(signal(final_decision="BUY WATCH", ema20=101.0))

        self.assertEqual(simulator.position_size, 0.0)
        self.assertEqual(simulator.rejected_entry_reasons["below_1h_ema20"], 1)

    def test_buy_watch_with_all_gates_true_opens_trade(self) -> None:
        simulator = TradeSimulator(initial_capital=10000)

        simulator.process_signal(signal(final_decision="BUY WATCH"))

        self.assertGreater(simulator.position_size, 0.0)
        self.assertEqual(simulator.accepted_entries, 1)
        self.assertEqual(simulator.rejected_entries, 0)

    def test_aggressive_profile_uses_configured_allocation(self) -> None:
        simulator = TradeSimulator(
            initial_capital=10000,
            fee_rate=0.0,
            slippage_rate=0.0,
            strategy_profile=get_strategy_profile("aggressive"),
        )

        simulator.process_signal(signal(final_decision="BUY WATCH"))

        self.assertEqual(simulator.cash, 7000.0)
        self.assertEqual(simulator.entry_total_cost, 3000.0)

    def test_profile_can_disable_attribution_filters(self) -> None:
        profile = replace(
            get_strategy_profile("aggressive"),
            require_alignment=False,
            require_rr_ratio=False,
            require_volume_ratio=False,
        )
        simulator = TradeSimulator(initial_capital=10000, strategy_profile=profile)

        simulator.process_signal(
            signal(
                final_decision="BUY WATCH",
                alignment="MIXED_ALIGNMENT",
                rr_ratio=0.1,
                volume_ratio=0.1,
            )
        )

        self.assertGreater(simulator.position_size, 0.0)

    def test_bull_mode_allows_pullback_entry_with_relaxed_rr_and_volume(self) -> None:
        profile = replace(get_strategy_profile("aggressive"), enable_bull_market_mode=True)
        simulator = TradeSimulator(initial_capital=10000, strategy_profile=profile)

        simulator.process_signal(
            signal(
                final_decision="BUY WATCH",
                alignment="PULLBACK_IN_UPTREND",
                rr_ratio=1.2,
                volume_ratio=0.5,
                daily_rsi=60.0,
                daily_price=120.0,
                daily_ema200=100.0,
            )
        )

        self.assertGreater(simulator.position_size, 0.0)
        self.assertTrue(simulator.entry_audit["entry_bull_mode_active"])
        self.assertEqual(simulator.entry_audit["entry_regime_tag"], "bull")

    def test_bull_mode_inactive_uses_current_aggressive_gates(self) -> None:
        profile = replace(get_strategy_profile("aggressive"), enable_bull_market_mode=True)
        simulator = TradeSimulator(initial_capital=10000, strategy_profile=profile)

        simulator.process_signal(
            signal(
                final_decision="BUY WATCH",
                alignment="PULLBACK_IN_UPTREND",
                rr_ratio=1.2,
                volume_ratio=0.5,
                daily_rsi=50.0,
                daily_price=120.0,
                daily_ema200=100.0,
            )
        )

        self.assertEqual(simulator.position_size, 0.0)
        self.assertEqual(simulator.rejected_entry_reasons["bearish_alignment"], 1)
        self.assertEqual(simulator.rejected_entry_reasons["low_rr_ratio"], 1)
        self.assertEqual(simulator.rejected_entry_reasons["low_volume_ratio"], 1)

    def test_trade_records_entry_and_exit_regime_tags(self) -> None:
        profile = replace(get_strategy_profile("aggressive"), enable_bull_market_mode=True)
        simulator = TradeSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0, strategy_profile=profile)
        simulator.process_signal(
            signal(
                final_decision="BUY WATCH",
                alignment="PULLBACK_IN_UPTREND",
                rr_ratio=1.2,
                volume_ratio=0.5,
                daily_rsi=60.0,
                daily_price=120.0,
                daily_ema200=100.0,
                stop_loss=95.0,
                target_1=101.0,
            )
        )
        simulator.process_signal(
            signal(
                timestamp="2024-01-03T01:00:00+00:00",
                price=102.0,
                daily_rsi=40.0,
                daily_price=80.0,
                daily_ema200=100.0,
            )
        )

        self.assertEqual(len(simulator.trades), 1)
        self.assertEqual(simulator.trades[0].entry_regime_tag, "bull")
        self.assertEqual(simulator.trades[0].exit_regime_tag, "bear")
        self.assertTrue(simulator.trades[0].entry_bull_mode_active)

    def test_trend_rider_takes_partials_and_exits_runner_on_daily_rsi(self) -> None:
        simulator = TrendRiderSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(signal(timestamp="2024-01-02T00:00:00+00:00", price=120.0))
        self.assertAlmostEqual(simulator.position_size, simulator.initial_position_size * 0.5)
        simulator.process_signal(signal(timestamp="2024-01-03T00:00:00+00:00", price=140.0))
        self.assertAlmostEqual(simulator.position_size, simulator.initial_position_size * 0.25)
        simulator.process_signal(
            signal(
                timestamp="2024-01-04T00:00:00+00:00",
                price=150.0,
                daily_rsi=49.0,
            )
        )

        self.assertEqual(simulator.position_size, 0.0)
        self.assertEqual(len(simulator.trades), 1)
        trade = simulator.trades[0]
        self.assertEqual(trade.exit_reason, "RUNNER_RSI_EXIT")
        self.assertEqual(len(trade.partial_exits), 2)
        self.assertGreater(trade.runner_return_pct, 0.0)
        self.assertEqual(trade.runner_holding_hours, 72.0)
        self.assertGreater(trade.profit_capture_ratio, 0.0)

    def test_trend_rider_runner_exit_waits_until_tp2_creates_runner(self) -> None:
        simulator = TrendRiderSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(
            signal(
                timestamp="2024-01-02T00:00:00+00:00",
                price=120.0,
                daily_rsi=49.0,
            )
        )

        self.assertEqual(len(simulator.trades), 0)
        self.assertTrue(simulator.tp1_done)
        self.assertFalse(simulator.tp2_done)
        self.assertAlmostEqual(simulator.position_size, simulator.initial_position_size * 0.5)

    def test_trend_rider_stop_before_tp2_has_no_runner_metrics(self) -> None:
        simulator = TrendRiderSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(signal(timestamp="2024-01-02T00:00:00+00:00", price=89.0))

        self.assertEqual(len(simulator.trades), 1)
        trade = simulator.trades[0]
        self.assertEqual(trade.exit_reason, "STOP_LOSS")
        self.assertIsNone(trade.runner_return_pct)
        self.assertIsNone(trade.runner_holding_hours)

    def test_trend_rider_runner_exits_on_ten_percent_trailing_stop(self) -> None:
        simulator = TrendRiderSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(signal(timestamp="2024-01-02T00:00:00+00:00", price=150.0))
        simulator.process_signal(signal(timestamp="2024-01-03T00:00:00+00:00", price=134.0))

        self.assertEqual(len(simulator.trades), 1)
        self.assertEqual(simulator.trades[0].exit_reason, "RUNNER_TRAILING_STOP")

    def test_hybrid_tp1_tp2_runner_and_breakeven_stop(self) -> None:
        simulator = HybridTrendRiderSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(signal(timestamp="2024-01-02T00:00:00+00:00", price=120.0))

        self.assertAlmostEqual(simulator.position_size, simulator.initial_position_size * 0.5)
        self.assertTrue(simulator.tp1_done)
        self.assertEqual(simulator.tp1_hit_count, 1)
        self.assertEqual(simulator.entry_stop_loss, simulator.entry_price)

        simulator.process_signal(signal(timestamp="2024-01-03T00:00:00+00:00", price=140.0))

        self.assertAlmostEqual(simulator.position_size, simulator.initial_position_size * 0.25)
        self.assertTrue(simulator.tp2_done)
        self.assertEqual(simulator.tp2_hit_count, 1)
        self.assertEqual(simulator.runner_activation_count, 1)

    def test_hybrid_uses_support_as_stop_when_decision_stop_is_missing(self) -> None:
        simulator = HybridTrendRiderSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)

        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=None, support=95.0))

        self.assertEqual(simulator.entry_stop_loss, 95.0)
        self.assertEqual(simulator.initial_risk, 5.0)
        simulator.process_signal(signal(timestamp="2024-01-02T00:00:00+00:00", price=110.0))
        self.assertTrue(simulator.tp1_done)

    def test_hybrid_runner_exits_on_daily_rsi(self) -> None:
        simulator = HybridTrendRiderSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(signal(timestamp="2024-01-02T00:00:00+00:00", price=120.0))
        simulator.process_signal(signal(timestamp="2024-01-03T00:00:00+00:00", price=140.0))
        simulator.process_signal(signal(timestamp="2024-01-04T00:00:00+00:00", price=145.0, daily_rsi=49.0))

        self.assertEqual(len(simulator.trades), 1)
        self.assertEqual(simulator.trades[0].exit_reason, "RUNNER_RSI_EXIT")

    def test_hybrid_runner_exits_on_daily_ema_cross(self) -> None:
        simulator = HybridTrendRiderSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(signal(timestamp="2024-01-02T00:00:00+00:00", price=120.0))
        simulator.process_signal(signal(timestamp="2024-01-03T00:00:00+00:00", price=140.0))
        simulator.process_signal(
            signal(
                timestamp="2024-01-04T00:00:00+00:00",
                price=145.0,
                daily_rsi=55.0,
                daily_ema20=99.0,
                daily_ema50=100.0,
            )
        )

        self.assertEqual(len(simulator.trades), 1)
        self.assertEqual(simulator.trades[0].exit_reason, "RUNNER_EMA_EXIT")

    def test_hybrid_runner_exits_when_daily_price_below_ema50(self) -> None:
        simulator = HybridTrendRiderSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(signal(timestamp="2024-01-02T00:00:00+00:00", price=120.0))
        simulator.process_signal(signal(timestamp="2024-01-03T00:00:00+00:00", price=140.0))
        simulator.process_signal(
            signal(
                timestamp="2024-01-04T00:00:00+00:00",
                price=145.0,
                daily_rsi=55.0,
                daily_price=99.0,
                daily_ema20=105.0,
                daily_ema50=100.0,
            )
        )

        self.assertEqual(len(simulator.trades), 1)
        self.assertEqual(simulator.trades[0].exit_reason, "RUNNER_DAILY_EMA50_EXIT")

    def test_hybrid_runner_exits_on_fifteen_percent_trailing_stop(self) -> None:
        simulator = HybridTrendRiderSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(signal(timestamp="2024-01-02T00:00:00+00:00", price=120.0))
        simulator.process_signal(signal(timestamp="2024-01-03T00:00:00+00:00", price=150.0))
        simulator.process_signal(
            signal(
                timestamp="2024-01-04T00:00:00+00:00",
                price=126.0,
                daily_rsi=55.0,
                daily_price=140.0,
                daily_ema20=120.0,
                daily_ema50=100.0,
            )
        )

        self.assertEqual(len(simulator.trades), 1)
        self.assertEqual(simulator.trades[0].exit_reason, "RUNNER_TRAILING_STOP")

    def test_hybrid_blocks_new_entry_while_runner_is_active(self) -> None:
        simulator = HybridTrendRiderSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(signal(timestamp="2024-01-02T00:00:00+00:00", price=120.0))
        simulator.process_signal(signal(timestamp="2024-01-03T00:00:00+00:00", price=140.0))
        simulator.process_signal(
            signal(
                timestamp="2024-01-04T00:00:00+00:00",
                final_decision="BUY",
                price=142.0,
                daily_rsi=55.0,
                daily_price=142.0,
                daily_ema20=120.0,
                daily_ema50=100.0,
            )
        )

        self.assertEqual(simulator.accepted_entries, 1)
        self.assertEqual(simulator.rejected_entry_reasons["already_holding"], 1)
        self.assertAlmostEqual(simulator.position_size, simulator.initial_position_size * 0.25)

    def test_hybrid_max_runner_drawdown_rule_exits_position(self) -> None:
        simulator = HybridTrendRiderSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(signal(timestamp="2024-01-02T00:00:00+00:00", price=120.0))
        simulator.process_signal(signal(timestamp="2024-01-03T00:00:00+00:00", price=150.0))
        simulator.process_signal(
            signal(
                timestamp="2024-01-04T00:00:00+00:00",
                price=110.0,
                daily_rsi=55.0,
                daily_price=160.0,
                daily_ema20=120.0,
                daily_ema50=100.0,
            )
        )

        self.assertEqual(len(simulator.trades), 1)
        self.assertEqual(simulator.trades[0].exit_reason, "RUNNER_MAX_DRAWDOWN_EXIT")
        self.assertGreater(simulator.trades[0].runner_max_drawdown_pct, 25.0)

    def test_hybrid_balanced_uses_40_percent_runner_and_slower_daily_exits(self) -> None:
        simulator = HybridTrendRiderSimulator(
            initial_capital=10000,
            fee_rate=0.0,
            slippage_rate=0.0,
            hybrid_profile_name="hybrid_balanced",
        )
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(signal(timestamp="2024-01-02T00:00:00+00:00", price=120.0, daily_rsi=49.0))
        self.assertAlmostEqual(simulator.position_size, simulator.initial_position_size * 0.6)
        simulator.process_signal(signal(timestamp="2024-01-03T00:00:00+00:00", price=140.0, daily_rsi=49.0))
        self.assertAlmostEqual(simulator.position_size, simulator.initial_position_size * 0.4)
        simulator.process_signal(signal(timestamp="2024-01-04T00:00:00+00:00", price=145.0, daily_rsi=44.0))

        self.assertEqual(len(simulator.trades), 1)
        self.assertEqual(simulator.trades[0].exit_reason, "RUNNER_RSI_EXIT")

    def test_hybrid_balanced_exits_on_daily_ema50_below_ema100(self) -> None:
        simulator = HybridTrendRiderSimulator(
            initial_capital=10000,
            fee_rate=0.0,
            slippage_rate=0.0,
            hybrid_profile_name="hybrid_balanced",
        )
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(signal(timestamp="2024-01-02T00:00:00+00:00", price=120.0, daily_rsi=50.0))
        simulator.process_signal(signal(timestamp="2024-01-03T00:00:00+00:00", price=140.0, daily_rsi=50.0))
        simulator.process_signal(
            signal(
                timestamp="2024-01-04T00:00:00+00:00",
                price=145.0,
                daily_rsi=50.0,
                daily_ema50=99.0,
                daily_ema100=100.0,
            )
        )

        self.assertEqual(len(simulator.trades), 1)
        self.assertEqual(simulator.trades[0].exit_reason, "RUNNER_EMA_EXIT")

    def test_hybrid_aggressive_uses_50_percent_runner_and_weekly_exits(self) -> None:
        simulator = HybridTrendRiderSimulator(
            initial_capital=10000,
            fee_rate=0.0,
            slippage_rate=0.0,
            hybrid_profile_name="hybrid_aggressive",
        )
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(signal(timestamp="2024-01-02T00:00:00+00:00", price=120.0, weekly_rsi=50.0))
        self.assertAlmostEqual(simulator.position_size, simulator.initial_position_size * 0.75)
        simulator.process_signal(signal(timestamp="2024-01-03T00:00:00+00:00", price=140.0, weekly_rsi=50.0))
        self.assertAlmostEqual(simulator.position_size, simulator.initial_position_size * 0.5)
        simulator.process_signal(signal(timestamp="2024-01-04T00:00:00+00:00", price=145.0, weekly_rsi=44.0))

        self.assertEqual(len(simulator.trades), 1)
        self.assertEqual(simulator.trades[0].exit_reason, "RUNNER_RSI_EXIT")

    def test_hybrid_aggressive_exits_on_weekly_ema_cross(self) -> None:
        simulator = HybridTrendRiderSimulator(
            initial_capital=10000,
            fee_rate=0.0,
            slippage_rate=0.0,
            hybrid_profile_name="hybrid_aggressive",
        )
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(signal(timestamp="2024-01-02T00:00:00+00:00", price=120.0, weekly_rsi=50.0))
        simulator.process_signal(signal(timestamp="2024-01-03T00:00:00+00:00", price=140.0, weekly_rsi=50.0))
        simulator.process_signal(
            signal(
                timestamp="2024-01-04T00:00:00+00:00",
                price=145.0,
                weekly_rsi=50.0,
                weekly_ema20=99.0,
                weekly_ema50=100.0,
            )
        )

        self.assertEqual(len(simulator.trades), 1)
        self.assertEqual(simulator.trades[0].exit_reason, "RUNNER_EMA_EXIT")

    def test_hybrid_aggressive_uses_twenty_five_percent_trailing_stop(self) -> None:
        simulator = HybridTrendRiderSimulator(
            initial_capital=10000,
            fee_rate=0.0,
            slippage_rate=0.0,
            hybrid_profile_name="hybrid_aggressive",
        )
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(signal(timestamp="2024-01-02T00:00:00+00:00", price=120.0, weekly_rsi=50.0))
        simulator.process_signal(signal(timestamp="2024-01-03T00:00:00+00:00", price=160.0, weekly_rsi=50.0))
        simulator.process_signal(signal(timestamp="2024-01-04T00:00:00+00:00", price=119.0, weekly_rsi=50.0))

        self.assertEqual(len(simulator.trades), 1)
        self.assertEqual(simulator.trades[0].exit_reason, "RUNNER_TRAILING_STOP")

    def test_avoid_long_blocks_new_position(self) -> None:
        simulator = TradeSimulator(initial_capital=10000)

        simulator.process_signal(signal(final_decision="AVOID LONG"))

        self.assertEqual(simulator.position_size, 0.0)
        self.assertEqual(simulator.cash, 10000.0)

    def test_close_open_position_marks_end_of_backtest(self) -> None:
        simulator = TradeSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))

        snapshot = simulator.close_open_position(
            "2024-01-02T00:00:00+00:00",
            110.0,
            signal(timestamp="2024-01-02T00:00:00+00:00", price=110.0),
        )

        self.assertEqual(simulator.position_size, 0.0)
        self.assertEqual(len(simulator.trades), 1)
        self.assertEqual(simulator.trades[0].exit_reason, "END_OF_BACKTEST")
        self.assertEqual(snapshot.position_size, 0.0)

    def test_take_profit_closes_position(self) -> None:
        simulator = TradeSimulator(initial_capital=10000)
        simulator.process_signal(signal(final_decision="BUY", stop_loss=95.0, target_1=105.0))
        simulator.process_signal(signal(timestamp="2024-01-03T01:00:00+00:00", price=106.0))

        self.assertEqual(simulator.position_size, 0.0)
        self.assertEqual(len(simulator.trades), 1)
        self.assertEqual(simulator.trades[0].exit_reason, "TAKE_PROFIT")
        self.assertEqual(simulator.trades[0].open_reason, "all_entry_gates_passed")
        self.assertEqual(simulator.trades[0].close_reason, "TAKE_PROFIT")
        self.assertEqual(simulator.trades[0].entry_decision, "BUY")
        self.assertEqual(simulator.trades[0].exit_decision, "WAIT")
        self.assertEqual(simulator.trades[0].entry_alignment, "BULLISH_ALIGNMENT")
        self.assertEqual(simulator.trades[0].entry_rr_ratio, 2.5)
        self.assertEqual(simulator.trades[0].entry_volume_ratio, 1.2)
        self.assertGreater(simulator.trades[0].r_multiple, 0.0)
        self.assertGreater(simulator.trades[0].pnl, 0.0)

    def test_take_profit_uses_two_r_when_no_target_exists(self) -> None:
        simulator = TradeSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0, target_1=None))
        simulator.process_signal(signal(timestamp="2024-01-03T01:00:00+00:00", price=120.0))

        self.assertEqual(len(simulator.trades), 1)
        self.assertEqual(simulator.trades[0].exit_reason, "TAKE_PROFIT")
        self.assertEqual(simulator.trades[0].entry_target_1, 120.0)
        self.assertEqual(simulator.trades[0].r_multiple, 2.0)

    def test_stop_loss_closes_position(self) -> None:
        simulator = TradeSimulator(initial_capital=10000)
        simulator.process_signal(signal(final_decision="BUY", stop_loss=95.0))
        simulator.process_signal(signal(timestamp="2024-01-01T01:00:00+00:00", price=94.0, stop_loss=95.0))

        self.assertEqual(len(simulator.trades), 1)
        self.assertEqual(simulator.trades[0].exit_reason, "STOP_LOSS")

    def test_trailing_stop_after_two_r_gain(self) -> None:
        simulator = TradeSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0, target_1=200.0))
        simulator.process_signal(signal(timestamp="2024-01-02T00:00:00+00:00", price=121.0))
        simulator.process_signal(signal(timestamp="2024-01-03T00:00:00+00:00", price=110.0))

        self.assertEqual(len(simulator.trades), 1)
        self.assertEqual(simulator.trades[0].exit_reason, "TRAILING_STOP")
        self.assertEqual(simulator.trades[0].trailing_stop, 111.0)

    def test_momentum_exit_cannot_fire_before_24_hours(self) -> None:
        simulator = TradeSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(
            signal(
                timestamp="2024-01-01T23:00:00+00:00",
                price=105.0,
                macd="bearish",
                four_hour_macd="bearish",
                four_hour_price=95.0,
                four_hour_ema20=100.0,
            )
        )

        self.assertEqual(len(simulator.trades), 0)
        self.assertGreater(simulator.momentum_exits_blocked, 0)

    def test_momentum_exit_cannot_fire_when_trade_is_losing(self) -> None:
        simulator = TradeSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(
            signal(
                timestamp="2024-01-03T01:00:00+00:00",
                price=99.0,
                macd="bearish",
                four_hour_macd="bearish",
                four_hour_price=95.0,
                four_hour_ema20=100.0,
            )
        )

        self.assertEqual(len(simulator.trades), 0)

    def test_momentum_exit_requires_4h_bearish_macd(self) -> None:
        simulator = TradeSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(
            signal(
                timestamp="2024-01-03T01:00:00+00:00",
                price=105.0,
                macd="bearish",
                four_hour_macd="bullish",
                four_hour_price=95.0,
                four_hour_ema20=100.0,
            )
        )

        self.assertEqual(len(simulator.trades), 0)
        self.assertEqual(simulator.momentum_exits_blocked, 1)

    def test_momentum_exit_requires_4h_close_below_ema20(self) -> None:
        simulator = TradeSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(
            signal(
                timestamp="2024-01-03T01:00:00+00:00",
                price=105.0,
                macd="bearish",
                four_hour_macd="bearish",
                four_hour_price=101.0,
                four_hour_ema20=100.0,
            )
        )

        self.assertEqual(len(simulator.trades), 0)
        self.assertEqual(simulator.momentum_exits_blocked, 1)

    def test_momentum_exit_fires_after_4h_confirmation(self) -> None:
        simulator = TradeSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0, target_1=200.0))
        simulator.process_signal(
            signal(
                timestamp="2024-01-03T01:00:00+00:00",
                price=105.0,
                four_hour_macd="bearish",
                four_hour_price=95.0,
                four_hour_ema20=100.0,
            )
        )

        self.assertEqual(len(simulator.trades), 1)
        self.assertEqual(simulator.trades[0].exit_reason, "MOMENTUM_EXIT")

    def test_time_exit_after_seven_days_when_not_profitable(self) -> None:
        simulator = TradeSimulator(initial_capital=10000, fee_rate=0.0, slippage_rate=0.0)
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0))
        simulator.process_signal(signal(timestamp="2024-01-09T01:00:00+00:00", price=99.0))

        self.assertEqual(len(simulator.trades), 1)
        self.assertEqual(simulator.trades[0].exit_reason, "TIME_EXIT")

    def test_bearish_regime_exit_rule(self) -> None:
        simulator = TradeSimulator(initial_capital=10000)
        simulator.process_signal(signal(final_decision="BUY"))
        simulator.process_signal(
            signal(
                timestamp="2024-01-03T01:00:00+00:00",
                price=98.0,
                market_regime="BEAR",
                trend_score=3,
            )
        )

        self.assertEqual(len(simulator.trades), 1)
        self.assertEqual(simulator.trades[0].exit_reason, "BEAR_TREND")

    def test_bear_trend_daily_breakdown_can_exit_before_minimum_hold(self) -> None:
        simulator = TradeSimulator(initial_capital=10000)
        simulator.process_signal(signal(final_decision="BUY"))
        simulator.process_signal(
            signal(
                timestamp="2024-01-01T12:00:00+00:00",
                price=98.0,
                market_regime="BEAR",
                trend_score=3,
                daily_setup="BEAR_TREND",
                daily_price=95.0,
                daily_ema20=100.0,
            )
        )

        self.assertEqual(len(simulator.trades), 1)
        self.assertEqual(simulator.trades[0].exit_reason, "BEAR_TREND")


if __name__ == "__main__":
    unittest.main()
