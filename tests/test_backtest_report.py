from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import pandas as pd

from backtesting.backtest_report import write_backtest_report, write_benchmark_comparison, write_profile_comparison


@dataclass(frozen=True)
class FakeBacktestResult:
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    metrics: dict
    trades: list
    equity_curve: pd.DataFrame


class BacktestReportTests(unittest.TestCase):
    def test_write_backtest_report_outputs_json_and_csv_files(self) -> None:
        result = FakeBacktestResult(
            symbol="BTCUSDT",
            start_date="2024-01-01",
            end_date="2024-01-03",
            initial_capital=10000.0,
            metrics={
                "final_equity": 11000.0,
                "total_return_pct": 10.0,
                "profit_factor": float("inf"),
            },
            trades=[
                {
                    "entry_timestamp": "2024-01-01T00:00:00+00:00",
                    "exit_timestamp": "2024-01-02T00:00:00+00:00",
                    "pnl": 1000.0,
                    "return_pct": 10.0,
                }
            ],
            equity_curve=pd.DataFrame(
                {
                    "timestamp": ["2024-01-01T00:00:00+00:00", "2024-01-02T00:00:00+00:00"],
                    "current_equity": [10000.0, 11000.0],
                    "buy_and_hold_equity": [10000.0, 10500.0],
                }
            ),
        )

        with TemporaryDirectory() as temp_dir:
            paths = write_backtest_report(result, Path(temp_dir), write_chart=False)

            self.assertTrue(paths["report"].exists())
            self.assertTrue(paths["trades"].exists())
            self.assertTrue(paths["equity_curve"].exists())
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            self.assertEqual(report["symbol"], "BTCUSDT")
            self.assertIsNone(report["profit_factor"])

    def test_write_backtest_report_keeps_trade_audit_headers_when_no_trades(self) -> None:
        result = FakeBacktestResult(
            symbol="BTCUSDT",
            start_date="2024-01-01",
            end_date="2024-01-03",
            initial_capital=10000.0,
            metrics={"final_equity": 10000.0, "total_return_pct": 0.0},
            trades=[],
            equity_curve=pd.DataFrame(
                {
                    "timestamp": ["2024-01-01T00:00:00+00:00"],
                    "current_equity": [10000.0],
                }
            ),
        )

        with TemporaryDirectory() as temp_dir:
            paths = write_backtest_report(result, Path(temp_dir), write_chart=False)

            trades_csv = paths["trades"].read_text(encoding="utf-8")
            self.assertIn("open_reason", trades_csv)
            self.assertIn("entry_alignment", trades_csv)
            self.assertIn("rejected_entry_reasons", trades_csv)

    def test_write_profile_comparison_ranks_by_risk_adjusted_return(self) -> None:
        def result(metrics: dict) -> FakeBacktestResult:
            return FakeBacktestResult(
                symbol="BTCUSDT",
                start_date="2024-01-01",
                end_date="2024-01-03",
                initial_capital=10000.0,
                metrics=metrics,
                trades=[],
                equity_curve=pd.DataFrame({"timestamp": [], "current_equity": []}),
            )

        results = {
            "conservative": result(
                {
                    "total_return_pct": 5.0,
                    "cagr": 4.0,
                    "max_drawdown_pct": 4.0,
                    "profit_factor": 1.4,
                    "sharpe_ratio": 0.8,
                    "win_rate": 45.0,
                    "expectancy": 0.2,
                    "total_trades": 10,
                }
            ),
            "balanced": result(
                {
                    "total_return_pct": 12.0,
                    "cagr": 8.0,
                    "max_drawdown_pct": 12.0,
                    "profit_factor": 1.2,
                    "sharpe_ratio": 0.6,
                    "win_rate": 42.0,
                    "expectancy": 0.1,
                    "total_trades": 20,
                }
            ),
            "aggressive": result(
                {
                    "total_return_pct": 20.0,
                    "cagr": 10.0,
                    "max_drawdown_pct": 25.0,
                    "profit_factor": 1.1,
                    "sharpe_ratio": 0.4,
                    "win_rate": 40.0,
                    "expectancy": 0.05,
                    "total_trades": 30,
                }
            ),
        }

        with TemporaryDirectory() as temp_dir:
            path = write_profile_comparison(results, Path(temp_dir))

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["best_profile_by_risk_adjusted_return"], "conservative")
            self.assertEqual(payload["ranking"][0], "conservative")
            self.assertEqual(payload["profiles"]["balanced"]["total_return_pct"], 12.0)

    def test_write_benchmark_comparison_outputs_required_metrics(self) -> None:
        def result(metrics: dict) -> FakeBacktestResult:
            return FakeBacktestResult(
                symbol="BTCUSDT",
                start_date="2024-01-01",
                end_date="2024-01-03",
                initial_capital=10000.0,
                metrics=metrics,
                trades=[],
                equity_curve=pd.DataFrame({"timestamp": [], "current_equity": []}),
            )

        results = {
            "buy_and_hold": result(
                {
                    "total_return_pct": 10.0,
                    "cagr": 8.0,
                    "max_drawdown_pct": 20.0,
                    "sharpe_ratio": 0.5,
                    "profit_factor": 0.0,
                    "total_trades": 0,
                }
            ),
            "daily_ema200": result(
                {
                    "total_return_pct": 8.0,
                    "cagr": 7.0,
                    "max_drawdown_pct": 8.0,
                    "sharpe_ratio": 0.8,
                    "profit_factor": 1.3,
                    "total_trades": 4,
                }
            ),
        }

        with TemporaryDirectory() as temp_dir:
            path = write_benchmark_comparison(results, Path(temp_dir))

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["best_strategy_by_risk_adjusted_return"], "daily_ema200")
            self.assertEqual(payload["ranking"][0], "daily_ema200")
            self.assertEqual(payload["strategies"]["buy_and_hold"]["total_return_pct"], 10.0)
            self.assertEqual(payload["strategies"]["daily_ema200"]["total_trades"], 4)


if __name__ == "__main__":
    unittest.main()
