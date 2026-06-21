from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from trading_agent.config import AgentConfig
from trading_agent.data import BybitKlineProvider
from trading_agent.main import (
    _run_investor_command,
    build_audit_fees_parser,
    build_backtest_parser,
    build_coinbase_execution_audit_parser,
    build_collect_shadow_signals_parser,
    build_false_avoid_analysis_parser,
    build_investor_parser,
    build_market_data_provider,
    build_merge_coinbase_execution_audit_parser,
    build_shadow_coinbase_parser,
    build_validate_equities_parser,
    run,
)


class FakeProvider:
    def __init__(self) -> None:
        self.intervals: list[str] = []

    def fetch_ohlcv(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        self.intervals.append(interval)
        timestamps = pd.date_range("2024-01-01", periods=limit, freq="h", tz="UTC")
        close = np.linspace(100.0, 200.0, limit)
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": close - 1,
                "high": close + 2,
                "low": close - 2,
                "close": close,
                "volume": np.full(limit, 100.0),
            }
        )


class MainTests(unittest.TestCase):
    def test_hype_uses_bybit_provider_in_auto_mode(self) -> None:
        provider = build_market_data_provider(AgentConfig(symbol="HYPEUSDT"))

        self.assertIsInstance(provider, BybitKlineProvider)

    def test_run_writes_json_and_invokes_chart_writer(self) -> None:
        chart_calls: list[Path] = []

        def chart_writer(_frame: pd.DataFrame, output_dir: Path, symbol: str, setup_label: str) -> Path:
            chart_calls.append(output_dir)
            self.assertEqual(symbol, "BTCUSDT")
            self.assertIsInstance(setup_label, str)
            self.assertIn("Alignment:", setup_label)
            path = output_dir / "chart.png"
            path.write_bytes(b"fake-png")
            return path

        with TemporaryDirectory() as temp_dir:
            config = AgentConfig(output_dir=Path(temp_dir), history_limit=220)
            provider = FakeProvider()
            payload = run(config, provider=provider, chart_writer=chart_writer)

            self.assertEqual(payload["symbol"], "BTCUSDT")
            self.assertEqual(payload["market_data_source"], "BINANCE")
            self.assertEqual(payload["position_mode"], "NO_POSITION")
            self.assertIn("decision", payload)
            self.assertIn("decision_meaning", payload)
            self.assertIn("final_decision", payload)
            self.assertIn("multi_timeframe", payload)
            self.assertIn("signal_journal", payload)
            self.assertIn("rr_ratio", payload)
            self.assertIn("market_regime", payload)
            self.assertIn("setup", payload)
            self.assertTrue((Path(temp_dir) / "output.json").exists())
            self.assertTrue((Path(temp_dir) / "signal_journal.json").exists())
            self.assertEqual(chart_calls, [Path(temp_dir)])
            self.assertEqual(provider.intervals, ["1h", "4h", "1d"])

    def test_backtest_parser_accepts_cross_asset_validation_options(self) -> None:
        args = build_backtest_parser().parse_args(
            [
                "--strategy",
                "cross_asset_validation",
                "--assets",
                "BTCUSDT",
                "SPY",
                "--include-optional-assets",
            ]
        )

        self.assertEqual(args.strategy, "cross_asset_validation")
        self.assertEqual(args.assets, ["BTCUSDT", "SPY"])
        self.assertTrue(args.include_optional_assets)

    def test_backtest_parser_accepts_exit_optimization_strategy(self) -> None:
        args = build_backtest_parser().parse_args(["--strategy", "exit_optimization"])

        self.assertEqual(args.strategy, "exit_optimization")

    def test_backtest_parser_accepts_broker_cost_validation_strategy(self) -> None:
        args = build_backtest_parser().parse_args(["--strategy", "broker_cost_validation"])

        self.assertEqual(args.strategy, "broker_cost_validation")

    def test_audit_fees_parser_accepts_aggressive_strategy(self) -> None:
        args = build_audit_fees_parser().parse_args(["--symbol", "BTCUSDT", "--strategy", "aggressive"])

        self.assertEqual(args.symbol, "BTCUSDT")
        self.assertEqual(args.strategy, "aggressive")

    def test_validate_equities_parser_defaults_to_spy_and_qqq(self) -> None:
        args = build_validate_equities_parser().parse_args([])

        self.assertEqual(args.assets, ["SPY", "QQQ"])
        self.assertEqual(args.start, "2018-01-01")
        self.assertEqual(args.timeframes, ["1h", "4h", "1d"])

    def test_coinbase_execution_audit_parser_accepts_smoke_run_options(self) -> None:
        args = build_coinbase_execution_audit_parser().parse_args(
            [
                "--product",
                "BTC/USD",
                "--duration-hours",
                "0.1",
                "--interval-seconds",
                "60",
                "--intended-order-size",
                "2500",
            ]
        )

        self.assertEqual(args.product_id, "BTC/USD")
        self.assertEqual(args.duration_hours, 0.1)
        self.assertEqual(args.interval_seconds, 60.0)
        self.assertEqual(args.intended_order_size, 2500.0)

    def test_merge_coinbase_execution_audit_parser_accepts_input_csvs(self) -> None:
        args = build_merge_coinbase_execution_audit_parser().parse_args(
            [
                "--input-csv",
                "previous.csv",
                "continuation.csv",
                "--failed-samples",
                "0",
            ]
        )

        self.assertEqual(args.input_csv, [Path("previous.csv"), Path("continuation.csv")])
        self.assertEqual(args.failed_samples, 0)

    def test_shadow_coinbase_parser_defaults_to_30_day_btc_shadow_run(self) -> None:
        args = build_shadow_coinbase_parser().parse_args([])

        self.assertEqual(args.product_id, "BTC-USD")
        self.assertEqual(args.duration_days, 30.0)
        self.assertEqual(args.cycle_interval_seconds, 3600.0)
        self.assertEqual(args.initial_shadow_capital, 10000.0)
        self.assertEqual(args.intended_order_size, 2500.0)
        self.assertIsNone(args.target_signal_count)
        self.assertFalse(args.no_resume_signal_collection)

    def test_shadow_coinbase_parser_accepts_enriched_collection_target(self) -> None:
        args = build_shadow_coinbase_parser().parse_args(
            ["--target-signal-count", "50", "--no-resume-signal-collection"]
        )

        self.assertEqual(args.target_signal_count, 50)
        self.assertTrue(args.no_resume_signal_collection)

    def test_collect_shadow_signals_parser_accepts_quick_test_options(self) -> None:
        args = build_collect_shadow_signals_parser().parse_args(
            ["--product", "BTC-USD", "--target-signals", "3", "--interval-seconds", "60", "--reset"]
        )

        self.assertEqual(args.product_id, "BTC-USD")
        self.assertEqual(args.target_signals, 3)
        self.assertEqual(args.interval_seconds, 60.0)
        self.assertTrue(args.reset)

    def test_false_avoid_analysis_parser_accepts_output_dir(self) -> None:
        args = build_false_avoid_analysis_parser().parse_args(["--output-dir", "research_outputs"])

        self.assertEqual(args.output_dir, Path("research_outputs"))

    def test_investor_parser_accepts_offline_mode(self) -> None:
        args = build_investor_parser().parse_args(["--symbol", "BTC", "--offline"])

        self.assertEqual(args.symbol, "BTC")
        self.assertTrue(args.offline)

    def test_investor_parser_accepts_chainlink_alias_and_thesis_risk(self) -> None:
        args = build_investor_parser().parse_args(
            ["--symbol", "CHAINLNK", "--thesis-risk-level", "LOW", "--thesis-risk-flags", "oracle_competition"]
        )

        self.assertEqual(args.symbol, "CHAINLNK")
        self.assertEqual(args.thesis_risk_level, "LOW")
        self.assertEqual(args.thesis_risk_flags, ["oracle_competition"])

    def test_investor_parser_accepts_manual_crypto_snapshot(self) -> None:
        args = build_investor_parser().parse_args(
            [
                "--symbol",
                "SEI",
                "--asset-name",
                "Sei",
                "--sector",
                "Parallelized L1",
                "--market-data-source",
                "BINANCE",
                "--price",
                "0.25",
                "--ma200",
                "0.40",
                "--weekly-rsi",
                "36",
                "--monthly-ema20",
                "0.31",
                "--recent-cycle-high",
                "1.25",
                "--quote-volume-usd",
                "125000000",
                "--average-quote-volume-usd",
                "80000000",
                "--atr-pct",
                "4.5",
            ]
        )

        self.assertEqual(args.symbol, "SEI")
        self.assertEqual(args.asset_name, "Sei")
        self.assertEqual(args.sector, "Parallelized L1")
        self.assertEqual(args.market_data_source, "BINANCE")
        self.assertEqual(args.current_price, 0.25)
        self.assertEqual(args.quote_volume_usd, 125000000)
        self.assertEqual(args.average_quote_volume_usd, 80000000)

    def test_investor_parser_accepts_risk_engine_flags(self) -> None:
        args = build_investor_parser().parse_args(
            [
                "--symbol", "SPY",
                "--asset-class", "EQUITY",
                "--bucket", "core",
                "--default-position-usd", "3000",
                "--risk-config-path", "config/test_risk_config.json",
                "--portfolio-state-path", "data/test_portfolio_state.json",
                "--skip-risk-engine",
            ]
        )

        self.assertEqual(args.asset_class, "EQUITY")
        self.assertEqual(args.bucket, "core")
        self.assertEqual(args.default_position_usd, 3000.0)
        self.assertEqual(args.risk_config_path, Path("config/test_risk_config.json"))
        self.assertEqual(args.portfolio_state_path, Path("data/test_portfolio_state.json"))
        self.assertTrue(args.skip_risk_engine)

    def test_investor_command_auto_detects_core_etf_symbol(self) -> None:
        with TemporaryDirectory() as temp_dir:
            payload = _run_investor_command(
                ["--symbol", "SPY", "--output-dir", temp_dir, "--skip-risk-engine"]
            )

        self.assertEqual(payload["agent"], "EQUITY_INVESTOR")
        self.assertEqual(payload["bucket"], "core")
        self.assertNotIn("risk_decision", payload)

    def test_investor_command_attaches_risk_decision_for_equity(self) -> None:
        with TemporaryDirectory() as temp_dir:
            payload = _run_investor_command(
                [
                    "--symbol", "SPY",
                    "--output-dir", temp_dir,
                    "--risk-config-path", f"{temp_dir}/risk_config.json",
                    "--portfolio-state-path", f"{temp_dir}/portfolio_state.json",
                ]
            )

        self.assertIn("risk_decision", payload)
        self.assertEqual(payload["risk_decision"]["status"], "approved")
        self.assertEqual(payload["risk_decision"]["recommendation"]["asset_class"], "equity")

    def test_investor_command_attaches_risk_decision_for_btc_offline(self) -> None:
        with TemporaryDirectory() as temp_dir:
            payload = _run_investor_command(
                [
                    "--symbol", "BTC",
                    "--offline",
                    "--output-dir", temp_dir,
                    "--risk-config-path", f"{temp_dir}/risk_config.json",
                    "--portfolio-state-path", f"{temp_dir}/portfolio_state.json",
                ]
            )

        self.assertIn("risk_decision", payload)
        self.assertEqual(payload["risk_decision"]["recommendation"]["bucket"], "core")
        self.assertEqual(payload["risk_decision"]["recommendation"]["asset_class"], "crypto")


if __name__ == "__main__":
    unittest.main()
