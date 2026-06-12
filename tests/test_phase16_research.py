from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import numpy as np
import pandas as pd

from backtesting.backtest_engine import BacktestConfig
from backtesting.benchmarks.research import (
    benchmark_comparison_payload,
    hybrid_runner_optimization_payload,
    hybrid_trend_rider_report_payload,
    market_structure_stop_payload,
    profit_capture_payload,
    run_hybrid_runner_optimization,
    run_benchmark_suite,
    run_filter_attribution,
    run_regime_analysis,
    strategy_research_report_payload,
    trend_rider_analysis_payload,
    trend_participation_payload,
    write_hybrid_runner_optimization_output,
    write_hybrid_trend_rider_output,
    write_market_structure_stop_output,
    write_profit_capture_output,
    write_trend_rider_output,
    write_trend_participation_output,
    write_phase16_outputs,
)
from backtesting.benchmarks.strategies import (
    BenchmarkResult,
    BuyAndHoldStrategy,
    EMA200Strategy,
    GoldenCrossStrategy,
    RSITrendStrategy,
)


def ohlcv_frame(start: str, periods: int, freq: str, start_price: float = 100.0) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    close = np.linspace(start_price, start_price + periods, periods)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(periods, 1000.0),
        }
    )


@dataclass(frozen=True)
class FakeAgentResult:
    metrics: dict


class Phase16ResearchTests(unittest.TestCase):
    def test_benchmark_suite_runs_phase16_simple_strategies(self) -> None:
        config = BacktestConfig(
            symbol="BTCUSDT",
            start="2024-08-01",
            end="2024-09-01",
        )
        frames = {
            "1h": ohlcv_frame("2024-07-01", 2000, "1h"),
            "4h": ohlcv_frame("2024-01-01", 300, "4h"),
            "1d": ohlcv_frame("2024-01-01", 260, "1D"),
        }

        results = run_benchmark_suite(
            config,
            cached_data=frames,
            strategies=(BuyAndHoldStrategy(), EMA200Strategy(), GoldenCrossStrategy(), RSITrendStrategy()),
        )
        payload = benchmark_comparison_payload(results)

        self.assertEqual(set(results), {"buy_and_hold", "ema200", "golden_cross", "rsi_trend"})
        self.assertIn("total_trades", payload["strategies"]["ema200"])
        self.assertIn(payload["best_strategy_by_risk_adjusted_return"], payload["strategies"])

    def test_regime_analysis_slices_equity_curves(self) -> None:
        result = BenchmarkResult(
            name="agent_aggressive",
            symbol="BTCUSDT",
            start_date="2018-01-01",
            end_date="2018-01-10",
            initial_capital=10000.0,
            final_equity=11000.0,
            metrics={},
            trades=[],
            equity_curve=pd.DataFrame(
                {
                    "timestamp": pd.date_range("2018-01-01", periods=10, freq="1D", tz="UTC"),
                    "price": np.linspace(100.0, 110.0, 10),
                    "current_equity": np.linspace(10000.0, 11000.0, 10),
                }
            ),
        )

        payload = run_regime_analysis({"agent_aggressive": result})

        metrics = payload["regimes"]["2018 Bear"]["strategies"]["agent_aggressive"]
        self.assertEqual(metrics["return_pct"], 10.0)
        self.assertEqual(metrics["data_points"], 10)

    def test_filter_attribution_identifies_largest_return_drag(self) -> None:
        config = BacktestConfig(symbol="BTCUSDT", start="2024-01-01", end="2024-02-01")
        fake_results = [
            FakeAgentResult({"total_return_pct": 1.0, "max_drawdown_pct": 3.0, "sharpe_ratio": 0.2, "profit_factor": 1.0, "total_trades": 1}),
            FakeAgentResult({"total_return_pct": 1.0, "max_drawdown_pct": 3.0, "sharpe_ratio": 0.2, "profit_factor": 1.0, "total_trades": 1}),
            FakeAgentResult({"total_return_pct": 4.0, "max_drawdown_pct": 5.0, "sharpe_ratio": 0.5, "profit_factor": 1.4, "total_trades": 2}),
            FakeAgentResult({"total_return_pct": 2.0, "max_drawdown_pct": 4.0, "sharpe_ratio": 0.3, "profit_factor": 1.1, "total_trades": 2}),
            FakeAgentResult({"total_return_pct": 3.0, "max_drawdown_pct": 6.0, "sharpe_ratio": 0.4, "profit_factor": 1.2, "total_trades": 3}),
        ]

        with patch("backtesting.benchmarks.research._load_required_frames", return_value={}):
            with patch("backtesting.benchmarks.research._run_agent_with_profile", side_effect=fake_results):
                payload = run_filter_attribution(config, cached_data={})

        self.assertEqual(payload["most_expensive_filter"]["experiment"], "no_volume_filter")
        self.assertEqual(payload["experiments"]["no_volume_filter"]["return_delta_pct"], 3.0)

    def test_strategy_research_report_and_writers(self) -> None:
        benchmark_payload = {
            "best_strategy_by_risk_adjusted_return": "ema200",
            "strategies": {"ema200": {"sharpe_ratio": 1.0}},
        }
        regime_payload = {
            "regimes": {
                "2018 Bear": {
                    "strategies": {
                        "agent_aggressive": {"return_pct": -5.0, "max_drawdown_pct": 20.0, "sharpe_ratio": -0.5}
                    }
                },
                "2024 Bull": {
                    "strategies": {
                        "agent_aggressive": {"return_pct": 12.0, "max_drawdown_pct": 4.0, "sharpe_ratio": 1.2}
                    }
                },
            }
        }
        attribution_payload = {
            "most_expensive_filter": {
                "experiment": "no_rr_filter",
                "return_delta_pct": 4.0,
                "drawdown_delta_pct": 2.0,
            }
        }

        report = strategy_research_report_payload(benchmark_payload, regime_payload, attribution_payload)

        self.assertEqual(report["best_benchmark"], "ema200")
        self.assertEqual(report["best_regime"]["regime"], "2024 Bull")
        self.assertEqual(report["worst_regime"]["regime"], "2018 Bear")
        self.assertIn("RR threshold", report["recommended_next_optimization"])
        with TemporaryDirectory() as temp_dir:
            paths = write_phase16_outputs(
                Path(temp_dir),
                benchmark_payload,
                regime_payload,
                attribution_payload,
                report,
            )
            self.assertTrue(paths["benchmark_comparison"].exists())
            self.assertTrue(paths["regime_analysis"].exists())
            self.assertTrue(paths["filter_attribution"].exists())
            self.assertTrue(paths["strategy_research_report"].exists())

    def test_trend_participation_payload_compares_required_strategies(self) -> None:
        def result(name: str, total_return: float, drawdown: float, sharpe: float, trades: int) -> BenchmarkResult:
            return BenchmarkResult(
                name=name,
                symbol="BTCUSDT",
                start_date="2024-01-01",
                end_date="2024-01-31",
                initial_capital=10000.0,
                final_equity=10000.0,
                metrics={
                    "total_return_pct": total_return,
                    "cagr": total_return,
                    "max_drawdown_pct": drawdown,
                    "sharpe_ratio": sharpe,
                    "total_trades": trades,
                },
                trades=[],
                equity_curve=pd.DataFrame({"timestamp": [], "current_equity": []}),
            )

        payload = trend_participation_payload(
            {
                "agent_aggressive": result("agent_aggressive", 2.0, 10.0, 0.5, 4),
                "bull_mode_agent": result("bull_mode_agent", 6.0, 18.0, 1.0, 8),
                "rsi_trend": result("rsi_trend", 8.0, 25.0, 0.8, 2),
            }
        )

        self.assertEqual(payload["best_strategy_by_risk_adjusted_return"], "bull_mode_agent")
        self.assertEqual(payload["best_strategy_under_drawdown_limit"], "bull_mode_agent")
        self.assertTrue(payload["strategies"]["bull_mode_agent"]["drawdown_under_20_pct"])
        self.assertFalse(payload["strategies"]["rsi_trend"]["drawdown_under_20_pct"])
        with TemporaryDirectory() as temp_dir:
            path = write_trend_participation_output(Path(temp_dir), payload)
            self.assertTrue(path.exists())

    def test_profit_capture_payload_measures_missed_upside(self) -> None:
        result = BenchmarkResult(
            name="agent_aggressive",
            symbol="BTCUSDT",
            start_date="2024-01-01",
            end_date="2024-01-03",
            initial_capital=10000.0,
            final_equity=10000.0,
            metrics={},
            trades=[
                {
                    "entry_timestamp": "2024-01-01T00:00:00+00:00",
                    "exit_timestamp": "2024-01-03T00:00:00+00:00",
                    "entry_price": 100.0,
                    "exit_price": 110.0,
                    "pnl": 100.0,
                    "return_pct": 10.0,
                    "exit_reason": "TAKE_PROFIT",
                },
                {
                    "entry_timestamp": "2024-01-04T00:00:00+00:00",
                    "exit_timestamp": "2024-01-05T00:00:00+00:00",
                    "entry_price": 120.0,
                    "exit_price": 100.0,
                    "pnl": -100.0,
                    "return_pct": -16.67,
                    "exit_reason": "STOP_LOSS",
                },
            ],
            equity_curve=pd.DataFrame({"timestamp": [], "current_equity": []}),
        )
        price_history = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=40, freq="1D", tz="UTC"),
                "high": [100.0, 110.0, 112.0, 118.0, 100.0, 125.0, *([154.0] * 34)],
                "close": [100.0] * 40,
            }
        )

        payload = profit_capture_payload({"agent_aggressive": result}, price_history, windows_days=(7, 30))
        strategy = payload["strategies"]["agent_aggressive"]

        self.assertEqual(strategy["total_trades"], 2)
        self.assertEqual(strategy["average_holding_hours"], 36.0)
        self.assertEqual(strategy["median_holding_hours"], 36.0)
        self.assertEqual(strategy["longest_winning_trade"]["holding_hours"], 48.0)
        self.assertEqual(strategy["longest_losing_trade"]["holding_hours"], 24.0)
        self.assertEqual(strategy["top_10_winning_trades"][0]["return_pct"], 10.0)
        self.assertEqual(strategy["missed_opportunity_summary"]["30d"]["max_missed_gain_pct"], 54.0)
        self.assertGreater(strategy["profit_capture_ratio"], 0.0)
        with TemporaryDirectory() as temp_dir:
            path = write_profit_capture_output(Path(temp_dir), payload)
            self.assertTrue(path.exists())

    def test_trend_rider_analysis_payload_tracks_runner_and_capture_target(self) -> None:
        def result(name: str, exit_price: float, runner_return: float | None) -> BenchmarkResult:
            trade = {
                "entry_timestamp": "2024-01-01T00:00:00+00:00",
                "exit_timestamp": "2024-01-04T00:00:00+00:00",
                "entry_price": 100.0,
                "exit_price": exit_price,
                "pnl": 100.0,
                "return_pct": exit_price - 100.0,
                "exit_reason": "RUNNER_RSI_EXIT",
            }
            if runner_return is not None:
                trade["runner_return_pct"] = runner_return
                trade["runner_holding_hours"] = 72.0
                trade["partial_exits"] = [
                    {"price": 120.0, "position_fraction": 0.5},
                    {"price": 140.0, "position_fraction": 0.25},
                ]
            return BenchmarkResult(
                name=name,
                symbol="BTCUSDT",
                start_date="2024-01-01",
                end_date="2024-01-04",
                initial_capital=10000.0,
                final_equity=10100.0,
                metrics={
                    "total_return_pct": 1.0,
                    "cagr": 1.0,
                    "max_drawdown_pct": 2.0,
                    "sharpe_ratio": 0.5,
                    "total_trades": 1,
                },
                trades=[trade],
                equity_curve=pd.DataFrame({"timestamp": [], "current_equity": []}),
            )

        price_history = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="1D", tz="UTC"),
                "high": [100.0, 120.0, 130.0, *([150.0] * 97)],
                "close": [100.0] * 100,
            }
        )

        payload = trend_rider_analysis_payload(
            {
                "agent_aggressive": result("agent_aggressive", 105.0, None),
                "trend_rider_aggressive": result("trend_rider_aggressive", 94.0, -6.0),
            },
            price_history,
        )

        trend_rider = payload["strategies"]["trend_rider_aggressive"]
        self.assertEqual(payload["target_profit_capture_ratio"], 0.1)
        self.assertEqual(payload["trend_rider_rules"]["runner_pct"], 25)
        self.assertEqual(trend_rider["profit_capture_ratio"], 0.4)
        self.assertEqual(trend_rider["average_runner_return_pct"], -6.0)
        self.assertEqual(trend_rider["average_runner_holding_hours"], 72.0)
        self.assertTrue(trend_rider["profit_capture_target_met"])
        self.assertIsNone(payload["strategies"]["agent_aggressive"]["average_runner_return_pct"])
        with TemporaryDirectory() as temp_dir:
            path = write_trend_rider_output(Path(temp_dir), payload)
            self.assertTrue(path.exists())

    def test_hybrid_trend_rider_report_compares_targets_and_management_stats(self) -> None:
        def result(
            name: str,
            total_return: float,
            drawdown: float,
            capture_exit_price: float,
            metrics_extra: dict | None = None,
            trade_extra: dict | None = None,
        ) -> BenchmarkResult:
            trade = {
                "entry_timestamp": "2024-01-01T00:00:00+00:00",
                "exit_timestamp": "2024-02-01T00:00:00+00:00",
                "entry_price": 100.0,
                "exit_price": capture_exit_price,
                "pnl": 100.0,
                "return_pct": capture_exit_price - 100.0,
                "exit_reason": "RUNNER_TRAILING_STOP",
            }
            trade.update(trade_extra or {})
            metrics = {
                "total_return_pct": total_return,
                "cagr": total_return,
                "max_drawdown_pct": drawdown,
                "sharpe_ratio": 1.2,
                "profit_factor": 2.0,
                "total_trades": 1,
                "win_rate": 100.0,
            }
            metrics.update(metrics_extra or {})
            return BenchmarkResult(
                name=name,
                symbol="BTCUSDT",
                start_date="2024-01-01",
                end_date="2024-02-01",
                initial_capital=10000.0,
                final_equity=12000.0,
                metrics=metrics,
                trades=[trade],
                equity_curve=pd.DataFrame({"timestamp": [], "current_equity": []}),
            )

        price_history = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=120, freq="1D", tz="UTC"),
                "high": [100.0, 120.0, 130.0, *([150.0] * 117)],
                "close": [100.0] * 120,
            }
        )
        hybrid_trade = {
            "partial_exits": [
                {"price": 120.0, "position_fraction": 0.5, "reason": "TP1_2R"},
                {"price": 140.0, "position_fraction": 0.25, "reason": "TP2_4R"},
            ],
            "runner_return_pct": 30.0,
            "runner_holding_hours": 240.0,
            "runner_max_drawdown_pct": 12.0,
        }
        hybrid_metrics = {
            "tp1_hit_count": 1,
            "tp2_hit_count": 1,
            "runner_activation_count": 1,
            "runner_exit_reasons": {"RUNNER_TRAILING_STOP": 1},
            "average_runner_drawdown_pct": 12.0,
            "max_runner_drawdown_pct": 12.0,
        }

        payload = hybrid_trend_rider_report_payload(
            {
                "agent_aggressive": result("agent_aggressive", 80.0, 12.0, 105.0),
                "trend_rider_aggressive": result("trend_rider_aggressive", 200.0, 60.0, 130.0),
                "hybrid_trend_rider": result(
                    "hybrid_trend_rider",
                    130.0,
                    20.0,
                    130.0,
                    metrics_extra=hybrid_metrics,
                    trade_extra=hybrid_trade,
                ),
            },
            price_history,
        )

        hybrid = payload["strategies"]["hybrid_trend_rider"]
        self.assertTrue(payload["target_assessment"]["all_targets_met"])
        self.assertEqual(hybrid["tp1_hit_count"], 1)
        self.assertEqual(hybrid["tp2_hit_count"], 1)
        self.assertEqual(hybrid["runner_activation_count"], 1)
        self.assertEqual(hybrid["runner_exit_reasons"]["RUNNER_TRAILING_STOP"], 1)
        self.assertEqual(hybrid["average_runner_drawdown_pct"], 12.0)
        self.assertEqual(hybrid["max_runner_drawdown_pct"], 12.0)
        self.assertGreater(hybrid["profit_capture_ratio"], 0.15)
        with TemporaryDirectory() as temp_dir:
            path = write_hybrid_trend_rider_output(Path(temp_dir), payload)
            self.assertTrue(path.exists())

    def test_hybrid_runner_optimization_ranks_profiles_and_recommends_closest(self) -> None:
        def result(
            name: str,
            total_return: float,
            drawdown: float,
            sharpe: float,
            exit_price: float,
        ) -> BenchmarkResult:
            trade = {
                "entry_timestamp": "2024-01-01T00:00:00+00:00",
                "exit_timestamp": "2024-02-01T00:00:00+00:00",
                "entry_price": 100.0,
                "exit_price": exit_price,
                "pnl": 100.0,
                "return_pct": exit_price - 100.0,
                "exit_reason": "RUNNER_TRAILING_STOP",
                "partial_exits": [
                    {"price": 120.0, "position_fraction": 0.4, "reason": "TP1_2R"},
                    {"price": 140.0, "position_fraction": 0.2, "reason": "TP2_4R"},
                ],
                "runner_return_pct": exit_price - 100.0,
                "runner_holding_hours": 360.0,
                "runner_max_drawdown_pct": 10.0,
            }
            return BenchmarkResult(
                name=name,
                symbol="BTCUSDT",
                start_date="2024-01-01",
                end_date="2024-02-01",
                initial_capital=10000.0,
                final_equity=10000.0,
                metrics={
                    "total_return_pct": total_return,
                    "cagr": total_return,
                    "max_drawdown_pct": drawdown,
                    "sharpe_ratio": sharpe,
                    "profit_factor": 2.0,
                    "total_trades": 1,
                    "win_rate": 100.0,
                    "tp1_hit_count": 1,
                    "tp2_hit_count": 1,
                    "runner_activation_count": 1,
                    "runner_exit_reasons": {"RUNNER_TRAILING_STOP": 1},
                    "average_runner_drawdown_pct": 10.0,
                    "max_runner_drawdown_pct": 10.0,
                },
                trades=[trade],
                equity_curve=pd.DataFrame({"timestamp": [], "current_equity": []}),
            )

        price_history = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=120, freq="1D", tz="UTC"),
                "high": [100.0, 120.0, 130.0, *([150.0] * 117)],
                "close": [100.0] * 120,
            }
        )
        payload = hybrid_runner_optimization_payload(
            {
                "agent_aggressive": result("agent_aggressive", 87.0, 12.0, 0.83, 105.0),
                "trend_rider_aggressive": result("trend_rider_aggressive", 200.0, 60.0, 0.7, 130.0),
                "hybrid_conservative": result("hybrid_conservative", 90.0, 10.0, 0.8, 125.0),
                "hybrid_balanced": result("hybrid_balanced", 130.0, 20.0, 1.0, 130.0),
                "hybrid_aggressive": result("hybrid_aggressive", 160.0, 35.0, 0.9, 135.0),
            },
            price_history,
            focus_strategy="hybrid_balanced",
        )

        self.assertEqual(payload["focus_strategy"], "hybrid_balanced")
        self.assertEqual(payload["closest_profile"], "hybrid_balanced")
        self.assertEqual(payload["recommendation"]["profile"], "hybrid_balanced")
        self.assertIn("by_sharpe", payload["rankings"])
        self.assertIn("hybrid_balanced", payload["profiles_meeting_all_targets"])
        self.assertTrue(payload["strategies"]["hybrid_balanced"]["target_assessment"]["all_targets_met"])
        self.assertEqual(payload["strategies"]["hybrid_balanced"]["average_trend_duration_captured_hours"], 360.0)
        with TemporaryDirectory() as temp_dir:
            path = write_hybrid_runner_optimization_output(Path(temp_dir), payload)
            self.assertTrue(path.exists())

    def test_hybrid_runner_optimization_focus_runs_only_selected_strategy(self) -> None:
        captured_strategy_names: list[str] = []

        def fake_run_benchmark_suite(config, cached_data, progress_callback, strategies):
            del config, cached_data, progress_callback
            captured_strategy_names.extend(strategy.name for strategy in strategies)
            return {
                strategy.name: BenchmarkResult(
                    name=strategy.name,
                    symbol="BTCUSDT",
                    start_date="2024-01-01",
                    end_date="2024-01-02",
                    initial_capital=10000.0,
                    final_equity=10100.0,
                    metrics={
                        "total_return_pct": 1.0,
                        "cagr": 1.0,
                        "max_drawdown_pct": 1.0,
                        "sharpe_ratio": 0.5,
                        "profit_factor": 1.2,
                        "total_trades": 1,
                        "win_rate": 100.0,
                    },
                    trades=[
                        {
                            "entry_timestamp": "2024-01-01T00:00:00+00:00",
                            "exit_timestamp": "2024-01-02T00:00:00+00:00",
                            "entry_price": 100.0,
                            "exit_price": 101.0,
                            "pnl": 1.0,
                            "return_pct": 1.0,
                            "exit_reason": "TAKE_PROFIT",
                        }
                    ],
                    equity_curve=pd.DataFrame({"timestamp": [], "current_equity": []}),
                )
                for strategy in strategies
            }

        frames = {"1h": ohlcv_frame("2024-01-01", 30, "1h")}
        with TemporaryDirectory() as temp_dir:
            config = BacktestConfig(output_dir=Path(temp_dir))
            with patch("backtesting.benchmarks.research._load_required_frames", return_value=frames):
                with patch("backtesting.benchmarks.research.run_benchmark_suite", side_effect=fake_run_benchmark_suite):
                    payload = run_hybrid_runner_optimization(config, focus_strategy="hybrid_conservative")

        self.assertEqual(captured_strategy_names, ["hybrid_conservative"])
        self.assertEqual(set(payload["hybrid_runner_optimization"]["strategies"]), {"hybrid_conservative"})

    def test_market_structure_stop_payload_reports_survived_stopouts(self) -> None:
        def result(name: str, stop_outs: int, capture_exit_price: float, drawdown: float) -> BenchmarkResult:
            trade = {
                "entry_timestamp": "2024-01-01T00:00:00+00:00",
                "exit_timestamp": "2024-01-02T00:00:00+00:00",
                "entry_price": 100.0,
                "exit_price": capture_exit_price,
                "pnl": capture_exit_price - 100.0,
                "return_pct": capture_exit_price - 100.0,
                "exit_reason": "STOP_LOSS" if stop_outs else "TAKE_PROFIT",
                "entry_stop_distance_pct": 5.0,
                "entry_stop_distance_atr": 1.5,
                "entry_stop_candidates": {
                    "atr": {"stop_price": 90.0},
                    "swing_low": {"stop_price": 85.0},
                    "support_zone": {"stop_price": 80.0},
                },
            }
            return BenchmarkResult(
                name=name,
                symbol="BTCUSDT",
                start_date="2024-01-01",
                end_date="2024-01-02",
                initial_capital=10000.0,
                final_equity=10000.0,
                metrics={
                    "total_return_pct": 5.0,
                    "cagr": 5.0,
                    "max_drawdown_pct": drawdown,
                    "sharpe_ratio": 0.6,
                    "profit_factor": 1.2,
                    "total_trades": 1,
                    "stop_out_count": stop_outs,
                    "average_stop_distance_pct": 5.0,
                    "average_stop_distance_atr": 1.5,
                },
                trades=[trade],
                equity_curve=pd.DataFrame({"timestamp": [], "current_equity": []}),
            )

        price_history = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=120, freq="1D", tz="UTC"),
                "high": [100.0, *([140.0] * 119)],
                "close": [100.0] * 120,
            }
        )
        payload = market_structure_stop_payload(
            {
                "aggressive_current": result("aggressive_current", 1, 95.0, 12.0),
                "aggressive_atr_stop": result("aggressive_atr_stop", 0, 110.0, 15.0),
                "aggressive_swing_low_stop": result("aggressive_swing_low_stop", 0, 115.0, 18.0),
                "aggressive_support_zone_stop": result("aggressive_support_zone_stop", 0, 120.0, 19.0),
            },
            price_history,
            focus_stop_type="swing_low",
        )

        self.assertEqual(payload["focus_stop_type"], "swing_low")
        self.assertEqual(payload["baseline_stop_out_count"], 1)
        self.assertEqual(payload["strategies"]["aggressive_swing_low_stop"]["survived_stopouts_count"], 1)
        self.assertIn(payload["closest_strategy"], payload["strategies"])
        self.assertIn("by_profit_capture", payload["rankings"])
        with TemporaryDirectory() as temp_dir:
            path = write_market_structure_stop_output(Path(temp_dir), payload)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
