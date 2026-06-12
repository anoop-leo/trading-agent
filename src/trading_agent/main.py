"""Command-line orchestration for the Phase 1 signal engine."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from decision.decision_engine import (
    DecisionInput,
    DecisionResult,
    apply_multi_timeframe_alignment,
    make_decision,
)
from scoring.market_regime_skill import MarketRegimeResult, calculate_market_regime
from scoring.multi_timeframe_skill import (
    TimeframeSignal,
    analyze_multi_timeframe,
    chart_annotation,
)
from scoring.risk_reward_skill import RiskRewardResult, calculate_risk_reward
from scoring.setup_detection_skill import SetupInput, SetupResult, detect_setup
from scoring.support_resistance_skill import SupportResistanceResult, calculate_support_resistance
from trading_agent.config import AgentConfig
from trading_agent.data import BinanceKlineProvider, BybitKlineProvider
from trading_agent.indicators import add_indicators
from trading_agent.journal import update_signal_journal
from trading_agent.models import MarketDataProvider, SignalScores
from trading_agent.output import build_output_payload, macd_direction, write_chart, write_json
from trading_agent.scoring import calculate_recent_swing_high, calculate_recent_swing_low, calculate_scores, calculate_volume_ratio


ChartWriter = Callable[[pd.DataFrame, Path, str, str], Path]


@dataclass(frozen=True)
class TimeframeAnalysis:
    timeframe: str
    indicators: pd.DataFrame
    scores: SignalScores
    decision: DecisionResult
    support_resistance: SupportResistanceResult
    risk_reward: RiskRewardResult
    market_regime: MarketRegimeResult
    setup: SetupResult


def calculate_trade_quality(
    indicator_frame: pd.DataFrame,
    support_resistance: SupportResistanceResult | None = None,
) -> tuple[SupportResistanceResult, RiskRewardResult, MarketRegimeResult]:
    """Calculate Phase 1.1 trade-quality skills from the latest indicator frame."""

    latest = indicator_frame.iloc[-1]
    current_price = float(latest["close"])
    if support_resistance is None:
        support_resistance = calculate_support_resistance(indicator_frame, current_price)
    risk_reward = calculate_risk_reward(
        current_price=current_price,
        support=support_resistance.support,
        resistance=support_resistance.resistance,
    )
    market_regime = calculate_market_regime(
        current_price=current_price,
        ema50=float(latest["ema_50"]),
        ema200=float(latest["ema_200"]),
    )
    return support_resistance, risk_reward, market_regime


def build_decision_input(
    symbol: str,
    position_mode: str,
    indicator_frame: pd.DataFrame,
    scores: SignalScores,
    support_resistance: SupportResistanceResult,
    risk_reward: RiskRewardResult,
    market_regime: MarketRegimeResult,
    setup: SetupResult,
) -> DecisionInput:
    """Build Decision Engine v2 input from the latest indicator frame."""

    latest = indicator_frame.iloc[-1]
    return DecisionInput(
        symbol=symbol,
        position_mode=position_mode,
        trend_score=scores.trend_score,
        momentum_score=scores.momentum_score,
        volume_score=scores.volume_score,
        bottom_score=scores.bottom_score,
        sr_score=scores.sr_score,
        rr_score=scores.rr_score,
        regime_score=scores.regime_score,
        current_price=float(latest["close"]),
        ema20=float(latest["ema_20"]),
        ema50=float(latest["ema_50"]),
        ema200=float(latest["ema_200"]),
        recent_swing_high=calculate_recent_swing_high(indicator_frame),
        recent_swing_low=calculate_recent_swing_low(indicator_frame),
        rr_ratio=risk_reward.rr_ratio,
        market_regime=market_regime.market_regime.value,
        support=support_resistance.support,
        setup=setup.setup.value,
        setup_score=setup.setup_score,
        setup_confidence=setup.setup_confidence,
        volume_ratio=calculate_volume_ratio(latest),
        rsi=float(latest["rsi_14"]),
        macd=macd_direction(latest),
    )


def analyze_timeframe(
    symbol: str,
    timeframe: str,
    position_mode: str,
    ohlcv: pd.DataFrame,
) -> TimeframeAnalysis:
    """Run the Phase 1 deterministic signal pipeline for one timeframe."""

    return analyze_indicator_frame(symbol, timeframe, position_mode, add_indicators(ohlcv))


def analyze_indicator_frame(
    symbol: str,
    timeframe: str,
    position_mode: str,
    indicators: pd.DataFrame,
    support_resistance: SupportResistanceResult | None = None,
) -> TimeframeAnalysis:
    """Run the Phase 1 deterministic signal pipeline from precomputed indicators."""

    support_resistance, risk_reward, market_regime = calculate_trade_quality(indicators, support_resistance)
    base_scores = calculate_scores(indicators)
    scores = replace(
        base_scores,
        sr_score=support_resistance.sr_score,
        rr_score=risk_reward.rr_score,
        regime_score=market_regime.regime_score,
    )
    latest = indicators.iloc[-1]
    setup = detect_setup(
        SetupInput(
            price=float(latest["close"]),
            ema20=float(latest["ema_20"]),
            ema50=float(latest["ema_50"]),
            ema200=float(latest["ema_200"]),
            support=support_resistance.support,
            resistance=support_resistance.resistance,
            volume_ratio=calculate_volume_ratio(latest),
            trend_score=scores.trend_score,
            momentum_score=scores.momentum_score,
            bottom_score=scores.bottom_score,
            market_regime=market_regime.market_regime.value,
            rsi=float(latest["rsi_14"]),
        )
    )
    decision_input = build_decision_input(
        symbol,
        position_mode,
        indicators,
        scores,
        support_resistance,
        risk_reward,
        market_regime,
        setup,
    )
    decision = make_decision(decision_input)
    return TimeframeAnalysis(
        timeframe=timeframe,
        indicators=indicators,
        scores=scores,
        decision=decision,
        support_resistance=support_resistance,
        risk_reward=risk_reward,
        market_regime=market_regime,
        setup=setup,
    )


def build_timeframe_signal(analysis: TimeframeAnalysis) -> TimeframeSignal:
    """Build a compact signal snapshot for multi-timeframe alignment."""

    latest = analysis.indicators.iloc[-1]
    return TimeframeSignal(
        timeframe=analysis.timeframe,
        trend_score=analysis.scores.trend_score,
        momentum_score=analysis.scores.momentum_score,
        volume_score=analysis.scores.volume_score,
        bottom_score=analysis.scores.bottom_score,
        sr_score=analysis.scores.sr_score,
        rr_score=analysis.scores.rr_score,
        regime_score=analysis.scores.regime_score,
        setup=analysis.setup.setup.value,
        setup_confidence=analysis.setup.setup_confidence,
        decision=analysis.decision.decision.value,
        price=float(latest["close"]),
        rsi=float(latest["rsi_14"]),
        macd=macd_direction(latest),
        ema20=float(latest["ema_20"]),
        ema50=float(latest["ema_50"]),
        ema200=float(latest["ema_200"]),
        market_regime=analysis.market_regime.market_regime.value,
    )


def _fetch_timeframes(config: AgentConfig) -> tuple[str, ...]:
    timeframes = [config.interval]
    for timeframe in config.timeframes:
        if timeframe not in timeframes:
            timeframes.append(timeframe)
    return tuple(timeframes)


def build_market_data_provider(config: AgentConfig) -> MarketDataProvider:
    """Build the configured public market-data provider."""

    if config.resolved_market_data_source == "BYBIT":
        return BybitKlineProvider(
            base_url=config.bybit_base_url,
            timeout_seconds=config.request_timeout_seconds,
        )
    return BinanceKlineProvider(
        base_url=config.binance_base_url,
        timeout_seconds=config.request_timeout_seconds,
    )


def run(
    config: AgentConfig,
    provider: MarketDataProvider | None = None,
    chart_writer: ChartWriter = write_chart,
) -> dict[str, Any]:
    """Run one local Phase 1 signal-generation cycle."""

    market_data_provider = provider or build_market_data_provider(config)

    analyses: dict[str, TimeframeAnalysis] = {}
    for timeframe in _fetch_timeframes(config):
        ohlcv = market_data_provider.fetch_ohlcv(
            symbol=config.symbol,
            interval=timeframe,
            limit=config.history_limit,
        )
        analyses[timeframe] = analyze_timeframe(config.symbol, timeframe, config.position_mode, ohlcv)

    primary_analysis = analyses[config.interval]
    multi_timeframe = analyze_multi_timeframe(
        {timeframe: build_timeframe_signal(analyses[timeframe]) for timeframe in config.timeframes}
    )
    final_decision = apply_multi_timeframe_alignment(
        primary_analysis.decision.decision,
        multi_timeframe.alignment.value,
        config.position_mode,
    )
    payload = build_output_payload(
        config,
        primary_analysis.indicators,
        primary_analysis.scores,
        primary_analysis.decision,
        primary_analysis.support_resistance,
        primary_analysis.risk_reward,
        primary_analysis.market_regime,
        primary_analysis.setup,
        multi_timeframe,
        final_decision,
    )
    journal_frame = analyses["1d"].indicators if "1d" in analyses else primary_analysis.indicators
    journal_path, journal_status = update_signal_journal(payload, journal_frame, config.output_dir)
    payload["signal_journal"] = {
        "path": str(journal_path),
        "inserted": journal_status["inserted"],
        "evaluated_count": journal_status["evaluated_count"],
    }

    write_json(payload, config.output_dir)
    chart_writer(
        primary_analysis.indicators,
        config.output_dir,
        config.symbol,
        chart_annotation(multi_timeframe),
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 1 technical signal engine.")
    _add_signal_arguments(parser)
    return parser


def _add_signal_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbol", default="BTCUSDT", help="Supported symbol, e.g. BTCUSDT.")
    parser.add_argument("--interval", default="1h", help="Kline interval.")
    parser.add_argument(
        "--market-data-source",
        default="AUTO",
        choices=["AUTO", "BINANCE", "BYBIT"],
        help="Public market data source. AUTO uses Bybit for HYPEUSDT and Binance otherwise.",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=None,
        help="Multi-timeframe intervals, e.g. --timeframes 1h 4h 1d.",
    )
    parser.add_argument("--history-limit", type=int, default=500, help="Number of candles to fetch.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Artifact output directory.")
    parser.add_argument(
        "--position-mode",
        default="NO_POSITION",
        choices=["NO_POSITION", "HOLDING"],
        help="Decision terminology mode.",
    )


def build_backtest_parser() -> argparse.ArgumentParser:
    from backtesting.profiles import PROFILE_NAMES

    parser = argparse.ArgumentParser(description="Run the Phase 1.5 historical backtesting engine.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Backtest symbol. BTCUSDT is the Phase 1.5 default.")
    parser.add_argument("--start", default="2017-01-01", help="Start date, e.g. 2017-01-01.")
    parser.add_argument("--end", default="latest", help="End date or latest.")
    parser.add_argument(
        "--profile",
        default="balanced",
        choices=[*PROFILE_NAMES, "all"],
        help="Strategy profile to backtest. Use all to write profile_comparison.json.",
    )
    parser.add_argument(
        "--strategy",
        choices=[
            "aggressive",
            "trend_holding",
            "regime_gated_trend_holding",
            "hybrid_trend_rider",
            "hybrid_optimization",
            "hybrid_conservative",
            "hybrid_balanced",
            "hybrid_aggressive",
        ],
        help="Run a named strategy research workflow.",
    )
    parser.add_argument(
        "--stop-type",
        default="fixed",
        choices=["fixed", "atr", "swing_low", "support_zone"],
        help="Stop placement model for aggressive backtests.",
    )
    parser.add_argument("--initial-capital", type=float, default=10000.0, help="Initial virtual capital.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"), help="Historical candle cache directory.")
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore existing candle cache files and rebuild the requested range.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Backtest artifact output directory.")
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["1h", "4h", "1d"],
        help="Backtest timeframes. Default: 1h 4h 1d.",
    )
    parser.add_argument("--primary-timeframe", default="1h", help="Primary replay timeframe.")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1000,
        help="Print replay progress every N primary candles. Use 0 to disable interval progress.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable backtest progress logging.")
    parser.add_argument(
        "--benchmarks",
        action="store_true",
        help="Run Phase 1.6 benchmark strategy comparison.",
    )
    parser.add_argument(
        "--research",
        action="store_true",
        help="Run Phase 1.6 benchmark, regime, and filter-attribution research.",
    )
    parser.add_argument(
        "--trend-participation",
        action="store_true",
        help="Run Phase 1.7 trend participation research.",
    )
    parser.add_argument(
        "--profit-capture",
        action="store_true",
        help="Run Phase 1.8 trade duration and profit-capture analysis.",
    )
    parser.add_argument(
        "--trend-rider",
        action="store_true",
        help="Run Phase 1.9 Trend Rider analysis.",
    )
    return parser


def _run_signal_command(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    config = AgentConfig(
        symbol=args.symbol,
        interval=args.interval,
        history_limit=args.history_limit,
        output_dir=args.output_dir,
        position_mode=args.position_mode,
        timeframes=args.timeframes,
        market_data_source=args.market_data_source,
    )
    return run(config)


def _run_backtest_command(argv: Sequence[str] | None = None) -> dict[str, Any]:
    from backtesting.backtest_engine import (
        BacktestConfig,
        load_or_download_timeframes,
        run_backtest,
    )
    from backtesting.benchmarks.research import (
        run_hybrid_runner_optimization,
        run_hybrid_trend_rider_analysis,
        run_profit_capture_analysis,
        run_benchmark_suite,
        run_phase16_research,
        run_regime_gated_trend_holding_analysis,
        run_trend_holding_analysis,
        run_trend_rider_analysis,
        run_trend_participation_research,
    )
    from backtesting.backtest_report import (
        write_backtest_report,
        write_benchmark_comparison,
        write_profile_comparison,
    )
    from backtesting.profiles import PROFILE_NAMES

    args = build_backtest_parser().parse_args(argv)
    config = BacktestConfig(
        symbol=args.symbol.upper(),
        start=args.start,
        end=args.end,
        primary_timeframe=args.primary_timeframe,
        timeframes=tuple(args.timeframes),
        initial_capital=args.initial_capital,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        progress_interval=args.progress_interval,
        profile="balanced" if args.profile == "all" else args.profile,
        refresh_cache=args.refresh_cache,
        stop_type=args.stop_type,
    )
    progress_callback = None if args.quiet else _print_backtest_progress
    if args.strategy == "aggressive":
        from backtesting.benchmarks.research import run_market_structure_stop_analysis

        payload = run_market_structure_stop_analysis(
            config,
            focus_stop_type=args.stop_type,
            progress_callback=progress_callback,
        )
        report = payload["market_structure_stop_report"]
        return {
            "symbol": config.symbol,
            "phase": "1.12",
            "focus_stop_type": args.stop_type,
            "targets": report["targets"],
            "baseline_stop_out_count": report["baseline_stop_out_count"],
            "stop_out_survival_analysis": report["stop_out_survival_analysis"],
            "closest_strategy": report["closest_strategy"],
            "best_stop_type": report["best_stop_type"],
            "best_profit_capture": report["best_profit_capture"],
            "best_risk_adjusted_return": report["best_risk_adjusted_return"],
            "recommended_production_configuration": report["recommended_production_configuration"],
            "recommendation": report["recommendation"],
            "rankings": report["rankings"],
            "strategies_meeting_targets": report["strategies_meeting_targets"],
            "selected_strategy": report["strategies"].get(f"aggressive_{args.stop_type}_stop")
            if args.stop_type != "fixed"
            else report["strategies"].get("aggressive_current"),
            "strategies": {
                strategy: {
                    "total_return_pct": metrics["total_return_pct"],
                    "cagr": metrics["cagr"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "max_drawdown_pct": metrics["max_drawdown_pct"],
                    "profit_factor": metrics["profit_factor"],
                    "win_rate": metrics["win_rate"],
                    "profit_capture_ratio": metrics["profit_capture_ratio"],
                    "total_trades": metrics["total_trades"],
                    "stop_out_count": metrics["stop_out_count"],
                    "stop_type_usage": metrics["stop_type_usage"],
                    "average_stop_distance_pct": metrics["average_stop_distance_pct"],
                    "average_stop_distance_atr": metrics["average_stop_distance_atr"],
                    "survived_stopouts_count": metrics["survived_stopouts_count"],
                    "target_assessment": metrics["target_assessment"],
                }
                for strategy, metrics in report["strategies"].items()
            },
            "artifacts": payload["artifacts"],
        }

    if args.strategy == "trend_holding":
        payload = run_trend_holding_analysis(config, progress_callback=progress_callback)
        report = payload["trend_holding_report"]
        return {
            "symbol": config.symbol,
            "phase": "1.13",
            "targets": report["targets"],
            "comparison": report["comparison"],
            "recommended_configuration": report["recommended_configuration"],
            "missed_opportunity_recheck": {
                "sample_size": report["missed_opportunity_recheck"]["sample_size"],
                "runner_survived_count": report["missed_opportunity_recheck"]["runner_survived_count"],
                "runner_captured_count": report["missed_opportunity_recheck"]["runner_captured_count"],
                "additional_profit_captured_pct": report["missed_opportunity_recheck"]["additional_profit_captured_pct"],
            },
            "strategies": {
                strategy: {
                    "total_return_pct": metrics["total_return_pct"],
                    "cagr": metrics["cagr"],
                    "max_drawdown_pct": metrics["max_drawdown_pct"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "profit_factor": metrics["profit_factor"],
                    "total_trades": metrics["total_trades"],
                    "win_rate": metrics["win_rate"],
                    "profit_capture_ratio": metrics["profit_capture_ratio"],
                    "average_runner_return_pct": metrics["average_runner_return_pct"],
                    "max_runner_return_pct": metrics["max_runner_return_pct"],
                    "average_runner_holding_hours": metrics["average_runner_holding_hours"],
                    "median_runner_holding_hours": metrics["median_runner_holding_hours"],
                    "tp1_hits": metrics["tp1_hits"],
                    "tp2_hits": metrics["tp2_hits"],
                    "runner_activations": metrics["runner_activations"],
                    "runner_exit_reasons": metrics["runner_exit_reasons"],
                    "target_assessment": metrics["target_assessment"],
                }
                for strategy, metrics in report["strategies"].items()
            },
            "artifacts": payload["artifacts"],
        }

    if args.strategy == "regime_gated_trend_holding":
        payload = run_regime_gated_trend_holding_analysis(config, progress_callback=progress_callback)
        report = payload["regime_gated_trend_holding_report"]
        return {
            "symbol": config.symbol,
            "phase": "1.14",
            "targets": report["targets"],
            "comparison": report["comparison"],
            "recommendation": report["recommendation"],
            "missed_opportunity_recheck": {
                "sample_size": report["missed_opportunity_recheck"]["sample_size"],
                "survived_count": report["missed_opportunity_recheck"]["survived_count"],
                "additional_profit_captured_pct": report["missed_opportunity_recheck"]["additional_profit_captured_pct"],
            },
            "strategies": {
                strategy: {
                    "total_return_pct": metrics["total_return_pct"],
                    "cagr": metrics["cagr"],
                    "max_drawdown_pct": metrics["max_drawdown_pct"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "profit_factor": metrics["profit_factor"],
                    "total_trades": metrics["total_trades"],
                    "win_rate": metrics["win_rate"],
                    "profit_capture_ratio": metrics["profit_capture_ratio"],
                    "runner_activation_count": metrics["runner_activation_count"],
                    "runner_disabled_count": metrics["runner_disabled_count"],
                    "strong_bull_periods": metrics["strong_bull_periods"],
                    "bull_periods": metrics["bull_periods"],
                    "range_periods": metrics["range_periods"],
                    "bear_periods": metrics["bear_periods"],
                    "target_assessment": metrics["target_assessment"],
                }
                for strategy, metrics in report["strategies"].items()
            },
            "artifacts": payload["artifacts"],
        }

    if args.strategy in {"hybrid_optimization", "hybrid_conservative", "hybrid_balanced", "hybrid_aggressive"}:
        focus_strategy = None if args.strategy == "hybrid_optimization" else args.strategy
        payload = run_hybrid_runner_optimization(
            config,
            focus_strategy=focus_strategy,
            progress_callback=progress_callback,
        )
        report = payload["hybrid_runner_optimization"]
        return {
            "symbol": config.symbol,
            "phase": "1.11",
            "focus_strategy": focus_strategy,
            "targets": report["targets"],
            "closest_profile": report["closest_profile"],
            "recommendation": report["recommendation"],
            "rankings": report["rankings"],
            "selected_strategy": report["strategies"].get(args.strategy) if focus_strategy else None,
            "strategies": {
                strategy: {
                    "total_return_pct": metrics["total_return_pct"],
                    "cagr": metrics["cagr"],
                    "max_drawdown_pct": metrics["max_drawdown_pct"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "profit_factor": metrics["profit_factor"],
                    "total_trades": metrics["total_trades"],
                    "profit_capture_ratio": metrics["profit_capture_ratio"],
                    "average_runner_holding_hours": metrics["average_runner_holding_hours"],
                    "median_runner_holding_hours": metrics["median_runner_holding_hours"],
                    "average_runner_return_pct": metrics["average_runner_return_pct"],
                    "max_runner_return_pct": metrics["max_runner_return_pct"],
                    "average_runner_drawdown_pct": metrics["average_runner_drawdown_pct"],
                    "max_runner_drawdown_pct": metrics["max_runner_drawdown_pct"],
                    "runner_exit_reasons": metrics["runner_exit_reasons"],
                    "target_assessment": metrics["target_assessment"],
                }
                for strategy, metrics in report["strategies"].items()
            },
            "artifacts": payload["artifacts"],
        }

    if args.strategy == "hybrid_trend_rider":
        payload = run_hybrid_trend_rider_analysis(config, progress_callback=progress_callback)
        report = payload["hybrid_trend_rider_report"]
        return {
            "symbol": config.symbol,
            "phase": "1.10",
            "targets": report["targets"],
            "target_assessment": report["target_assessment"],
            "strategies": {
                strategy: {
                    "total_return_pct": metrics["total_return_pct"],
                    "cagr": metrics["cagr"],
                    "max_drawdown_pct": metrics["max_drawdown_pct"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "profit_factor": metrics["profit_factor"],
                    "total_trades": metrics["total_trades"],
                    "win_rate": metrics["win_rate"],
                    "profit_capture_ratio": metrics["profit_capture_ratio"],
                    "average_runner_return_pct": metrics["average_runner_return_pct"],
                    "average_runner_holding_hours": metrics["average_runner_holding_hours"],
                    "tp1_hit_count": metrics["tp1_hit_count"],
                    "tp2_hit_count": metrics["tp2_hit_count"],
                    "runner_activation_count": metrics["runner_activation_count"],
                    "runner_exit_reasons": metrics["runner_exit_reasons"],
                    "average_runner_drawdown_pct": metrics["average_runner_drawdown_pct"],
                    "max_runner_drawdown_pct": metrics["max_runner_drawdown_pct"],
                }
                for strategy, metrics in report["strategies"].items()
            },
            "artifacts": payload["artifacts"],
        }

    if args.trend_rider:
        payload = run_trend_rider_analysis(config, progress_callback=progress_callback)
        report = payload["trend_rider_analysis"]
        return {
            "symbol": config.symbol,
            "phase": "1.9",
            "target_profit_capture_ratio": report["target_profit_capture_ratio"],
            "strategies": {
                strategy: {
                    "total_return_pct": metrics["total_return_pct"],
                    "max_drawdown_pct": metrics["max_drawdown_pct"],
                    "total_trades": metrics["total_trades"],
                    "profit_capture_ratio": metrics["profit_capture_ratio"],
                    "profit_capture_target_met": metrics["profit_capture_target_met"],
                    "average_runner_return_pct": metrics["average_runner_return_pct"],
                    "average_runner_holding_hours": metrics["average_runner_holding_hours"],
                }
                for strategy, metrics in report["strategies"].items()
            },
            "artifacts": payload["artifacts"],
        }

    if args.profit_capture:
        payload = run_profit_capture_analysis(config, progress_callback=progress_callback)
        report = payload["profit_capture_analysis"]
        return {
            "symbol": config.symbol,
            "phase": "1.8",
            "strategies": {
                strategy: {
                    "total_trades": metrics["total_trades"],
                    "average_holding_hours": metrics["average_holding_hours"],
                    "median_holding_hours": metrics["median_holding_hours"],
                    "profit_capture_ratio": metrics["profit_capture_ratio"],
                }
                for strategy, metrics in report["strategies"].items()
            },
            "artifacts": payload["artifacts"],
        }

    if args.trend_participation:
        trend_timeframes = _ordered_backtest_timeframes(
            config.primary_timeframe,
            tuple([*args.timeframes, "1d"]),
        )
        cached_data = load_or_download_timeframes(config, trend_timeframes)
        payload = run_trend_participation_research(config, cached_data=cached_data, progress_callback=progress_callback)
        report = payload["trend_participation"]
        return {
            "symbol": config.symbol,
            "phase": "1.7",
            "best_strategy_by_risk_adjusted_return": report["best_strategy_by_risk_adjusted_return"],
            "best_strategy_under_drawdown_limit": report["best_strategy_under_drawdown_limit"],
            "strategies": report["strategies"],
            "artifacts": payload["artifacts"],
        }

    if args.research:
        research_timeframes = _ordered_backtest_timeframes(
            config.primary_timeframe,
            tuple([*args.timeframes, "1d"]),
        )
        cached_data = load_or_download_timeframes(config, research_timeframes)
        payload = run_phase16_research(config, cached_data=cached_data, progress_callback=progress_callback)
        return {
            "symbol": config.symbol,
            "phase": "1.6",
            "best_benchmark": payload["strategy_research_report"]["best_benchmark"],
            "best_regime": payload["strategy_research_report"]["best_regime"],
            "worst_regime": payload["strategy_research_report"]["worst_regime"],
            "most_expensive_filter": payload["strategy_research_report"]["most_expensive_filter"],
            "artifacts": payload["artifacts"],
        }

    if args.benchmarks:
        benchmark_timeframes = _ordered_backtest_timeframes(
            config.primary_timeframe,
            tuple([*args.timeframes, "1d"]),
        )
        cached_data = load_or_download_timeframes(config, benchmark_timeframes)
        results = run_benchmark_suite(config, cached_data=cached_data, progress_callback=progress_callback)
        comparison_path = write_benchmark_comparison(results, args.output_dir)
        return {
            "symbol": config.symbol,
            "benchmark_comparison": {
                strategy: _benchmark_summary(result)
                for strategy, result in results.items()
            },
            "best_strategy_by_risk_adjusted_return": _best_strategy_from_results(results),
            "artifacts": {
                "benchmark_comparison": str(comparison_path),
            },
        }

    if args.profile == "all":
        comparison_timeframes = _ordered_backtest_timeframes(config.primary_timeframe, tuple(args.timeframes))
        cached_data = load_or_download_timeframes(config, comparison_timeframes)
        results = {}
        artifacts = {}
        for profile_name in PROFILE_NAMES:
            if progress_callback is not None:
                print(f"[backtest] profile {profile_name}", file=sys.stderr, flush=True)
            profile_config = replace(config, profile=profile_name)
            result = run_backtest(profile_config, cached_data=cached_data, progress_callback=progress_callback)
            results[profile_name] = result
            paths = write_backtest_report(
                result,
                args.output_dir / "profiles" / profile_name,
                write_chart=False,
            )
            artifacts[profile_name] = {name: str(path) for name, path in paths.items()}
        comparison_path = write_profile_comparison(results, args.output_dir)
        return {
            "symbol": config.symbol,
            "profile": "all",
            "best_profile_by_risk_adjusted_return": _best_profile_from_results(results),
            "profiles": {
                profile: _profile_summary(result)
                for profile, result in results.items()
            },
            "artifacts": {
                "profile_comparison": str(comparison_path),
                "profiles": artifacts,
            },
        }

    result = run_backtest(config, progress_callback=progress_callback)
    paths = write_backtest_report(result, args.output_dir)
    return {
        "symbol": result.symbol,
        "profile": result.profile,
        "strategy_profile": result.strategy_profile,
        "start_date": result.start_date,
        "end_date": result.end_date,
        "initial_capital": result.initial_capital,
        **result.metrics,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }


def _ordered_backtest_timeframes(primary_timeframe: str, timeframes: tuple[str, ...]) -> tuple[str, ...]:
    ordered = [primary_timeframe]
    for timeframe in timeframes:
        if timeframe not in ordered:
            ordered.append(timeframe)
    return tuple(ordered)


def _profile_summary(result: Any) -> dict[str, Any]:
    keys = (
        "total_return_pct",
        "cagr",
        "max_drawdown_pct",
        "profit_factor",
        "sharpe_ratio",
        "win_rate",
        "expectancy",
        "total_trades",
    )
    return {key: result.metrics.get(key) for key in keys}


def _benchmark_summary(result: Any) -> dict[str, Any]:
    keys = (
        "total_return_pct",
        "cagr",
        "max_drawdown_pct",
        "sharpe_ratio",
        "total_trades",
    )
    return {key: result.metrics.get(key) for key in keys}


def _best_profile_from_results(results: dict[str, Any]) -> str | None:
    if not results:
        return None
    return max(
        results,
        key=lambda profile: (
            float(results[profile].metrics.get("sharpe_ratio", float("-inf"))),
            -float(results[profile].metrics.get("max_drawdown_pct", float("inf"))),
            float(results[profile].metrics.get("cagr", float("-inf"))),
        ),
    )


def _best_strategy_from_results(results: dict[str, Any]) -> str | None:
    if not results:
        return None
    return max(
        results,
        key=lambda strategy: (
            float(results[strategy].metrics.get("sharpe_ratio", float("-inf"))),
            -float(results[strategy].metrics.get("max_drawdown_pct", float("inf"))),
            float(results[strategy].metrics.get("cagr", float("-inf"))),
        ),
    )


def _print_backtest_progress(event: dict[str, Any]) -> None:
    if event.get("phase") == "benchmark":
        print(
            f"[backtest] benchmark {event['strategy']}",
            file=sys.stderr,
            flush=True,
        )
        return

    if event.get("phase") == "filter_attribution":
        print(
            f"[backtest] filter attribution {event['experiment']}",
            file=sys.stderr,
            flush=True,
        )
        return

    if event.get("phase") == "prepared":
        print(
            "[backtest] prepared "
            f"{event['symbol']} with {event['primary_rows']} primary candles "
            f"across {', '.join(event['timeframes'])}",
            file=sys.stderr,
            flush=True,
        )
        return

    if event.get("phase") == "replay":
        timestamp = event.get("timestamp") or "warming up"
        latest_decision = event.get("latest_decision") or "none"
        print(
            "[backtest] replay "
            f"{event['processed_rows']}/{event['total_rows']} "
            f"({event['pct_complete']}%) "
            f"decisions={event['decisions']} trades={event['trades']} "
            f"latest={latest_decision} timestamp={timestamp}",
            file=sys.stderr,
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "backtest":
        payload = _run_backtest_command(args[1:])
    elif args and args[0] == "signal":
        payload = _run_signal_command(args[1:])
    else:
        payload = _run_signal_command(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
