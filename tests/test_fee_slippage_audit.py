import unittest

import pandas as pd

from backtesting.fee_slippage_audit import build_fee_slippage_audit_report
from backtesting.trade_simulator import TradeSimulator
from backtesting.profiles import get_strategy_profile

from test_trade_simulator import signal


class FeeSlippageAuditTests(unittest.TestCase):
    def test_final_equity_reconciles_to_sum_of_net_pnl(self) -> None:
        simulator = TradeSimulator(
            initial_capital=10000,
            fee_rate=0.001,
            slippage_rate=0.0005,
            strategy_profile=get_strategy_profile("aggressive"),
        )
        simulator.process_signal(signal(final_decision="BUY", price=100.0, stop_loss=90.0, target_1=105.0))
        simulator.process_signal(signal(timestamp="2024-01-03T01:00:00+00:00", price=110.0))
        equity_curve = simulator.equity_curve_frame()

        result = _fake_result(
            trades=simulator.trades_as_dicts(),
            equity_curve=equity_curve,
            initial_capital=simulator.initial_capital,
        )
        report = build_fee_slippage_audit_report(
            result,
            fee_rate=simulator.fee_rate,
            slippage_rate=simulator.slippage_rate,
        )

        self.assertTrue(report["net_pnl_equals_pnl"])
        self.assertTrue(report["final_equity_reconciles_to_net_pnl"])
        self.assertGreater(report["total_fees"], 0.0)
        self.assertGreater(report["total_slippage_cost"], 0.0)
        self.assertAlmostEqual(
            report["total_net_pnl"],
            sum(float(trade["net_pnl"]) for trade in result.trades),
            places=6,
        )
        self.assertEqual(report["missing_trade_fields"], [])


def _fake_result(trades: list[dict], equity_curve: pd.DataFrame, initial_capital: float):
    from backtesting.backtest_engine import BacktestResult

    final_equity = float(equity_curve.iloc[-1]["current_equity"])
    return BacktestResult(
        symbol="BTCUSDT",
        profile="aggressive",
        strategy_profile={},
        start_date="2024-01-01",
        end_date="2024-01-03",
        initial_capital=initial_capital,
        final_equity=final_equity,
        metrics={"final_equity": final_equity},
        decisions=[],
        trades=trades,
        equity_curve=equity_curve,
    )


if __name__ == "__main__":
    unittest.main()
