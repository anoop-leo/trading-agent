import unittest

import pandas as pd

from backtesting.performance_metrics import calculate_performance_metrics


class PerformanceMetricsTests(unittest.TestCase):
    def test_calculates_trade_and_equity_metrics(self) -> None:
        equity_curve = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01T00:00:00Z",
                        "2024-01-02T00:00:00Z",
                        "2024-01-03T00:00:00Z",
                        "2024-01-04T00:00:00Z",
                    ]
                ),
                "current_equity": [10000.0, 11000.0, 9000.0, 12000.0],
            }
        )
        trades = [
            {
                "entry_timestamp": "2024-01-01T00:00:00Z",
                "exit_timestamp": "2024-01-02T00:00:00Z",
                "pnl": 1000.0,
                "return_pct": 10.0,
                "r_multiple": 2.0,
                "exit_reason": "TAKE_PROFIT",
            },
            {
                "entry_timestamp": "2024-01-02T00:00:00Z",
                "exit_timestamp": "2024-01-03T12:00:00Z",
                "pnl": -500.0,
                "return_pct": -5.0,
                "r_multiple": -1.0,
                "exit_reason": "STOP_LOSS",
            },
            {
                "entry_timestamp": "2024-01-03T00:00:00Z",
                "exit_timestamp": "2024-01-04T00:00:00Z",
                "pnl": 250.0,
                "return_pct": 2.5,
                "r_multiple": 0.5,
                "exit_reason": "MOMENTUM_EXIT",
            },
        ]

        metrics = calculate_performance_metrics(
            equity_curve=equity_curve,
            trades=trades,
            initial_capital=10000.0,
            start_price=100.0,
            end_price=130.0,
        )

        self.assertEqual(metrics["total_return_pct"], 20.0)
        self.assertEqual(metrics["total_trades"], 3)
        self.assertEqual(metrics["winning_trades"], 2)
        self.assertEqual(metrics["losing_trades"], 1)
        self.assertEqual(metrics["win_rate"], 66.67)
        self.assertEqual(metrics["loss_rate"], 33.33)
        self.assertEqual(metrics["average_win_pct"], 6.25)
        self.assertEqual(metrics["average_loss_pct"], -5.0)
        self.assertEqual(metrics["profit_factor"], 2.5)
        self.assertEqual(metrics["max_drawdown_pct"], 18.18)
        self.assertEqual(metrics["expectancy"], 2.5)
        self.assertEqual(metrics["buy_and_hold_return_pct"], 30.0)
        self.assertEqual(metrics["average_r_multiple"], 0.5)
        self.assertEqual(
            metrics["average_r_multiple_by_exit_reason"],
            {"MOMENTUM_EXIT": 0.5, "STOP_LOSS": -1.0, "TAKE_PROFIT": 2.0},
        )
        self.assertEqual(metrics["best_trade"]["exit_reason"], "TAKE_PROFIT")
        self.assertEqual(metrics["worst_trade"]["exit_reason"], "STOP_LOSS")
        self.assertEqual(metrics["average_holding_hours"], 28.0)
        self.assertEqual(metrics["median_holding_hours"], 24.0)
        self.assertGreater(metrics["cagr"], 0)

    def test_empty_equity_curve_raises(self) -> None:
        with self.assertRaises(ValueError):
            calculate_performance_metrics(pd.DataFrame(), [], 10000.0, 100.0, 120.0)


if __name__ == "__main__":
    unittest.main()
