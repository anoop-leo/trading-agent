from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from backtesting.backtest_engine import normalize_yahoo_chart
from backtesting.cross_asset_validation import (
    asset_class_average,
    cross_asset_validation_payload,
    equity_success_criteria,
    equity_validation_report_payload,
    failure_analysis,
    recommended_production_assets,
    robustness_score,
    success_criteria,
    universal_trend_following_assessment,
    write_cross_asset_validation,
    write_equity_validation_report,
)


class CrossAssetValidationTests(unittest.TestCase):
    def test_robustness_score_awards_twenty_points_per_threshold(self) -> None:
        row = {
            "total_return_pct": 12.0,
            "sharpe_ratio": 0.9,
            "max_drawdown_pct": 20.0,
            "profit_factor": 1.2,
            "win_rate": 45.0,
        }

        self.assertEqual(robustness_score(row), 100)
        self.assertEqual(robustness_score({**row, "profit_factor": 0.8}), 80)

    def test_asset_class_average_uses_successful_rows_only(self) -> None:
        assets = {
            "BTCUSDT": {"status": "OK", "total_return_pct": 10.0, "sharpe_ratio": 1.0, "max_drawdown_pct": 12.0},
            "ETHUSDT": {"status": "OK", "total_return_pct": 20.0, "sharpe_ratio": 0.5, "max_drawdown_pct": 18.0},
            "SOLUSDT": {"status": "FAILED", "error": "missing"},
        }

        average = asset_class_average(assets, ("BTCUSDT", "ETHUSDT", "SOLUSDT"))

        self.assertEqual(average["assets"], ["BTCUSDT", "ETHUSDT"])
        self.assertEqual(average["failed_assets"], ["SOLUSDT"])
        self.assertEqual(average["average_return"], 15.0)
        self.assertEqual(average["average_sharpe"], 0.75)
        self.assertEqual(average["average_drawdown"], 15.0)

    def test_success_criteria_and_recommended_assets(self) -> None:
        good = {
            "status": "OK",
            "total_return_pct": 10.0,
            "sharpe_ratio": 1.0,
            "max_drawdown_pct": 12.0,
            "profit_factor": 1.2,
            "win_rate": 45.0,
            "robustness_score": 100,
        }
        weak = {
            "status": "OK",
            "total_return_pct": -2.0,
            "sharpe_ratio": -0.2,
            "max_drawdown_pct": 30.0,
            "profit_factor": 0.8,
            "win_rate": 35.0,
            "robustness_score": 0,
        }
        assets = {
            "BTCUSDT": good,
            "ETHUSDT": good,
            "SOLUSDT": good,
            "SPY": good,
            "QQQ": weak,
        }

        criteria = success_criteria(assets)

        self.assertTrue(criteria["crypto_sharpe_gt_0_8_average"])
        self.assertFalse(criteria["equity_positive_return_all_assets"])
        self.assertTrue(criteria["at_least_4_of_5_profitable"])
        self.assertEqual(recommended_production_assets(assets), ["BTCUSDT", "ETHUSDT", "SOLUSDT", "SPY"])

    def test_failure_analysis_reports_filters_and_loss_exits(self) -> None:
        row = {
            "status": "OK",
            "total_return_pct": -5.0,
            "sharpe_ratio": -0.4,
            "robustness_score": 20,
            "total_trades": 3,
            "rejected_entry_reasons": {"low_rr_ratio": 10},
            "loss_exit_reasons": {"STOP_LOSS": 2},
        }

        analysis = failure_analysis("QQQ", row)

        self.assertEqual(analysis["failure_reason"], "losses concentrated in STOP_LOSS exits")
        self.assertFalse(analysis["trend_following_worked"])

    def test_payload_ranks_assets_and_writes_report(self) -> None:
        assets = {
            "BTCUSDT": {
                "status": "OK",
                "asset_class": "crypto",
                "total_return_pct": 20.0,
                "sharpe_ratio": 1.2,
                "max_drawdown_pct": 10.0,
                "profit_factor": 1.5,
                "win_rate": 50.0,
                "total_trades": 4,
                "profit_capture_ratio": 0.12,
                "robustness_score": 100,
            },
            "ETHUSDT": {
                "status": "OK",
                "asset_class": "crypto",
                "total_return_pct": 8.0,
                "sharpe_ratio": 0.7,
                "max_drawdown_pct": 18.0,
                "profit_factor": 1.1,
                "win_rate": 42.0,
                "total_trades": 3,
                "profit_capture_ratio": 0.08,
                "robustness_score": 80,
            },
            "SPY": {"status": "FAILED", "asset_class": "equity", "error": "unavailable", "robustness_score": 0},
        }

        payload = cross_asset_validation_payload(
            assets,
            requested_assets=("BTCUSDT", "ETHUSDT", "SPY"),
            common_start="2020-01-01T00:00:00+00:00",
            common_end="2024-01-01T00:00:00+00:00",
        )

        self.assertEqual(payload["rankings"]["by_return"], ["BTCUSDT", "ETHUSDT"])
        self.assertEqual(payload["failed_assets"], ["SPY"])
        self.assertEqual(payload["recommended_production_assets"], ["BTCUSDT", "ETHUSDT"])
        with TemporaryDirectory() as temp_dir:
            path = write_cross_asset_validation(Path(temp_dir), payload)
            self.assertTrue(path.exists())

    def test_normalize_yahoo_chart_payload(self) -> None:
        payload = {
            "chart": {
                "result": [
                    {
                        "timestamp": [1704067200, 1704070800],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [100.0, 101.0],
                                    "high": [102.0, 103.0],
                                    "low": [99.0, 100.0],
                                    "close": [101.0, 102.0],
                                    "volume": [1000, 1200],
                                }
                            ]
                        },
                    }
                ],
                "error": None,
            }
        }

        frame = normalize_yahoo_chart(payload, "SPY", "1h")

        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.iloc[0]["timestamp"], pd.Timestamp("2024-01-01T00:00:00Z"))
        self.assertEqual(frame.iloc[1]["close"], 102.0)

    def test_equity_validation_report_payload(self) -> None:
        cross_report = {
            "assets": {
                "BTCUSDT": {
                    "status": "OK",
                    "total_return_pct": 10.0,
                    "sharpe_ratio": 1.0,
                    "max_drawdown_pct": 12.0,
                    "profit_factor": 1.2,
                    "win_rate": 45.0,
                    "robustness_score": 100,
                },
                "SPY": {
                    "status": "OK",
                    "total_return_pct": 5.0,
                    "cagr": 2.0,
                    "sharpe_ratio": 0.9,
                    "max_drawdown_pct": 10.0,
                    "profit_factor": 1.1,
                    "win_rate": 42.0,
                    "total_trades": 3,
                    "profit_capture_ratio": 0.05,
                    "robustness_score": 100,
                },
                "QQQ": {
                    "status": "OK",
                    "total_return_pct": 3.0,
                    "cagr": 1.0,
                    "sharpe_ratio": 0.5,
                    "max_drawdown_pct": 8.0,
                    "profit_factor": 1.1,
                    "win_rate": 45.0,
                    "total_trades": 2,
                    "profit_capture_ratio": 0.04,
                    "robustness_score": 80,
                },
            },
            "asset_class_analysis": {
                "crypto_average": {"average_sharpe": 1.0},
                "equity_average": {"average_sharpe": 0.7},
            },
            "rankings": {"by_return": ["BTCUSDT", "SPY", "QQQ"]},
            "recommended_production_assets": ["BTCUSDT", "SPY", "QQQ"],
        }
        data_validation = {
            "SPY": {"validation": "passed", "rows": 1000},
            "QQQ": {"validation": "passed", "rows": 1000},
        }
        criteria = equity_success_criteria(
            {"SPY": cross_report["assets"]["SPY"], "QQQ": cross_report["assets"]["QQQ"]},
            data_validation,
        )
        self.assertTrue(criteria["spy_data_loads"])
        self.assertTrue(criteria["positive_return_on_both_required_equities"])

        assessment = universal_trend_following_assessment(
            {**cross_report, "success_criteria": {"crypto_positive_return_all_assets": True}},
            criteria,
        )
        self.assertEqual(assessment["assessment"], "Crypto-only Trend Following")

        payload = equity_validation_report_payload(
            equity_results=[],
            equity_failures={"SPY": "not used", "QQQ": "not used"},
            cross_asset_report=cross_report,
            requested_equities=("SPY", "QQQ"),
        )

        self.assertEqual(payload["equity_rankings"]["by_return"], ["SPY", "QQQ"])
        self.assertEqual(payload["equity_metrics"]["SPY"]["return"], 5.0)
        with TemporaryDirectory() as temp_dir:
            path = write_equity_validation_report(Path(temp_dir), payload)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
