"""Phase 1.6 benchmark, regime, and filter-attribution research."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from backtesting.benchmarks.strategies import (
    DEFAULT_BENCHMARK_STRATEGIES,
    AgentAggressiveStrategy,
    AgentStopStrategy,
    BenchmarkResult,
    BenchmarkStrategy,
    BullModeAgentStrategy,
    HybridTrendRiderStrategy,
    RegimeGatedPortfolioGovernorStrategy,
    RegimeGatedTrendHoldingStrategy,
    RSITrendStrategy,
    TrendHoldingStrategy,
    TrendRiderAggressiveStrategy,
)
from backtesting.performance_metrics import calculate_performance_metrics
from backtesting.profiles import StrategyProfile, get_strategy_profile


ProgressCallback = Callable[[dict[str, Any]], None]

BENCHMARK_METRICS = (
    "total_return_pct",
    "cagr",
    "max_drawdown_pct",
    "sharpe_ratio",
    "total_trades",
)
FILTER_METRICS = (
    "total_return_pct",
    "max_drawdown_pct",
    "sharpe_ratio",
    "profit_factor",
)
TREND_PARTICIPATION_METRICS = (
    "total_return_pct",
    "cagr",
    "max_drawdown_pct",
    "sharpe_ratio",
    "total_trades",
)
HYBRID_REPORT_METRICS = (
    "total_return_pct",
    "cagr",
    "max_drawdown_pct",
    "sharpe_ratio",
    "profit_factor",
    "total_trades",
    "win_rate",
)
MARKET_STRUCTURE_STOP_METRICS = (
    "total_return_pct",
    "cagr",
    "sharpe_ratio",
    "max_drawdown_pct",
    "profit_factor",
    "win_rate",
    "total_trades",
)
RUNNER_EXIT_REASONS = (
    "RUNNER_RSI_EXIT",
    "RUNNER_EMA_EXIT",
    "RUNNER_DAILY_EMA50_EXIT",
    "RUNNER_MAX_DRAWDOWN_EXIT",
    "RUNNER_TRAILING_STOP",
    "END_OF_BACKTEST",
)
REGIMES = (
    ("2018 Bear", "2018-01-01", "2018-12-31"),
    ("2019 Recovery", "2019-01-01", "2019-12-31"),
    ("2020 Bull", "2020-01-01", "2020-12-31"),
    ("2021 Bull", "2021-01-01", "2021-12-31"),
    ("2022 Bear", "2022-01-01", "2022-12-31"),
    ("2023 Recovery", "2023-01-01", "2023-12-31"),
    ("2024 Bull", "2024-01-01", "2024-12-31"),
    ("2025-2026 Current", "2025-01-01", "latest"),
)


def run_benchmark_suite(
    config: Any,
    cached_data: dict[str, pd.DataFrame] | None = None,
    progress_callback: ProgressCallback | None = None,
    strategies: tuple[BenchmarkStrategy, ...] = DEFAULT_BENCHMARK_STRATEGIES,
) -> dict[str, BenchmarkResult]:
    """Run all Phase 1.6 benchmark strategies on a shared candle cache."""

    frames = _load_required_frames(config, cached_data)
    return {
        strategy.name: strategy.run(config=config, frames=frames, progress_callback=progress_callback)
        for strategy in strategies
    }


def benchmark_comparison_payload(results: dict[str, BenchmarkResult]) -> dict[str, Any]:
    rows = {
        name: {metric: result.metrics.get(metric) for metric in BENCHMARK_METRICS}
        for name, result in results.items()
    }
    ranking = _rank_by_risk_adjusted_return(rows)
    return {
        "selection_method": "highest sharpe_ratio, then lower max_drawdown_pct, then higher CAGR",
        "best_strategy_by_risk_adjusted_return": ranking[0] if ranking else None,
        "ranking": ranking,
        "strategies": rows,
    }


def run_regime_analysis(results: dict[str, BenchmarkResult]) -> dict[str, Any]:
    """Calculate return, drawdown, and Sharpe by market regime for each strategy."""

    regimes: dict[str, Any] = {}
    for regime_name, start, end in REGIMES:
        strategies: dict[str, Any] = {}
        for strategy_name, result in results.items():
            strategies[strategy_name] = _regime_metrics(result.equity_curve, start, end)
        regimes[regime_name] = {
            "start": start,
            "end": end,
            "strategies": strategies,
        }
    return {"regimes": regimes}


def run_filter_attribution(
    config: Any,
    cached_data: dict[str, pd.DataFrame] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run aggressive-profile experiments with one execution filter removed."""

    frames = _load_required_frames(config, cached_data)
    base_profile = get_strategy_profile("aggressive")
    experiments = {
        "base": base_profile,
        "no_macd_filter": replace(base_profile, name="no_macd_filter", require_4h_macd_bullish=False),
        "no_volume_filter": replace(
            base_profile,
            name="no_volume_filter",
            min_volume_ratio=0.0,
            require_volume_ratio=False,
        ),
        "no_rr_filter": replace(
            base_profile,
            name="no_rr_filter",
            min_rr_ratio=0.0,
            require_rr_ratio=False,
        ),
        "no_alignment_filter": replace(base_profile, name="no_alignment_filter", require_alignment=False),
    }
    results: dict[str, Any] = {}
    for experiment_name, profile in experiments.items():
        if progress_callback is not None:
            progress_callback({"phase": "filter_attribution", "experiment": experiment_name})
        result = _run_agent_with_profile(config, frames, profile, progress_callback)
        row = {metric: result.metrics.get(metric) for metric in FILTER_METRICS}
        row["total_trades"] = result.metrics.get("total_trades")
        row["profile"] = profile.to_dict()
        results[experiment_name] = row

    base_return = _float_or_none(results["base"].get("total_return_pct"))
    base_drawdown = _float_or_none(results["base"].get("max_drawdown_pct"))
    for experiment_name, row in results.items():
        row_return = _float_or_none(row.get("total_return_pct"))
        row_drawdown = _float_or_none(row.get("max_drawdown_pct"))
        row["return_delta_pct"] = _round_or_none(None if row_return is None or base_return is None else row_return - base_return)
        row["drawdown_delta_pct"] = _round_or_none(
            None if row_drawdown is None or base_drawdown is None else row_drawdown - base_drawdown
        )

    return {
        "baseline": "base",
        "experiments": results,
        "most_expensive_filter": _most_expensive_filter(results),
    }


def run_phase16_research(
    config: Any,
    cached_data: dict[str, pd.DataFrame] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run Phase 1.6 research and write all requested JSON outputs."""

    frames = _load_required_frames(config, cached_data)
    benchmark_results = run_benchmark_suite(config, cached_data=frames, progress_callback=progress_callback)
    benchmark_payload = benchmark_comparison_payload(benchmark_results)
    regime_payload = run_regime_analysis(benchmark_results)
    attribution_payload = run_filter_attribution(config, cached_data=frames, progress_callback=progress_callback)
    final_report = strategy_research_report_payload(
        benchmark_payload,
        regime_payload,
        attribution_payload,
    )
    paths = write_phase16_outputs(
        config.output_dir,
        benchmark_payload,
        regime_payload,
        attribution_payload,
        final_report,
    )
    return {
        "benchmark_comparison": benchmark_payload,
        "regime_analysis": regime_payload,
        "filter_attribution": attribution_payload,
        "strategy_research_report": final_report,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }


def run_trend_participation_research(
    config: Any,
    cached_data: dict[str, pd.DataFrame] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run Phase 1.7 trend participation comparison."""

    frames = _load_required_frames(config, cached_data)
    strategies: tuple[BenchmarkStrategy, ...] = (
        AgentAggressiveStrategy(),
        BullModeAgentStrategy(),
        RSITrendStrategy(),
    )
    results = run_benchmark_suite(config, cached_data=frames, progress_callback=progress_callback, strategies=strategies)
    payload = trend_participation_payload(results)
    path = write_trend_participation_output(config.output_dir, payload)
    return {
        "trend_participation": payload,
        "artifacts": {"trend_participation_report": str(path)},
    }


def run_profit_capture_analysis(
    config: Any,
    cached_data: dict[str, pd.DataFrame] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run Phase 1.8 trade duration and profit-capture analysis."""

    analysis_config = replace(config, end=_extend_end_for_profit_capture(config.end))
    frames = _load_required_frames(analysis_config, cached_data)
    strategies: tuple[BenchmarkStrategy, ...] = (
        AgentAggressiveStrategy(),
        BullModeAgentStrategy(),
        RSITrendStrategy(),
    )
    results = run_benchmark_suite(config, cached_data=frames, progress_callback=progress_callback, strategies=strategies)
    payload = profit_capture_payload(results, frames[config.primary_timeframe])
    path = write_profit_capture_output(config.output_dir, payload)
    return {
        "profit_capture_analysis": payload,
        "artifacts": {"profit_capture_analysis": str(path)},
    }


def run_trend_rider_analysis(
    config: Any,
    cached_data: dict[str, pd.DataFrame] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run Phase 1.9 Trend Rider comparison."""

    history_config = replace(config, end=_extend_end_for_profit_capture(config.end))
    execution_config = replace(config, close_open_position_on_end=True)
    frames = _load_required_frames(history_config, cached_data)
    strategies: tuple[BenchmarkStrategy, ...] = (
        AgentAggressiveStrategy(),
        TrendRiderAggressiveStrategy(),
    )
    results = run_benchmark_suite(
        execution_config,
        cached_data=frames,
        progress_callback=progress_callback,
        strategies=strategies,
    )
    payload = trend_rider_analysis_payload(results, frames[execution_config.primary_timeframe])
    path = write_trend_rider_output(execution_config.output_dir, payload)
    return {
        "trend_rider_analysis": payload,
        "artifacts": {"trend_rider_analysis": str(path)},
    }


def run_trend_holding_analysis(
    config: Any,
    cached_data: dict[str, pd.DataFrame] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run Phase 1.13 Trend Holding Engine comparison."""

    history_config = replace(config, end=_extend_end_for_profit_capture(config.end))
    execution_config = replace(config, close_open_position_on_end=True)
    frames = _load_required_frames(history_config, cached_data)
    strategies: tuple[BenchmarkStrategy, ...] = (
        AgentAggressiveStrategy(),
        TrendHoldingStrategy(),
    )
    results = run_benchmark_suite(
        execution_config,
        cached_data=frames,
        progress_callback=progress_callback,
        strategies=strategies,
    )
    payload = trend_holding_report_payload(results, frames[execution_config.primary_timeframe])
    path = write_trend_holding_output(execution_config.output_dir, payload)
    return {
        "trend_holding_report": payload,
        "artifacts": {"trend_holding_report": str(path)},
    }


def run_regime_gated_trend_holding_analysis(
    config: Any,
    cached_data: dict[str, pd.DataFrame] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run Phase 1.14 Regime-Gated Trend Holding comparison."""

    history_config = replace(config, end=_extend_end_for_profit_capture(config.end))
    execution_config = replace(config, close_open_position_on_end=True)
    frames = _load_required_frames(history_config, cached_data)
    strategies: tuple[BenchmarkStrategy, ...] = (
        AgentAggressiveStrategy(),
        TrendHoldingStrategy(),
        RegimeGatedTrendHoldingStrategy(),
    )
    results = run_benchmark_suite(
        execution_config,
        cached_data=frames,
        progress_callback=progress_callback,
        strategies=strategies,
    )
    payload = regime_gated_trend_holding_report_payload(results, frames[execution_config.primary_timeframe])
    path = write_regime_gated_trend_holding_output(execution_config.output_dir, payload)
    return {
        "regime_gated_trend_holding_report": payload,
        "artifacts": {"regime_gated_trend_holding_report": str(path)},
    }


def run_portfolio_risk_governor_analysis(
    config: Any,
    cached_data: dict[str, pd.DataFrame] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run Phase 1.15 Portfolio Risk Governor comparison."""

    history_config = replace(config, end=_extend_end_for_profit_capture(config.end))
    execution_config = replace(config, close_open_position_on_end=True)
    frames = _load_required_frames(history_config, cached_data)
    strategies: tuple[BenchmarkStrategy, ...] = (
        AgentAggressiveStrategy(),
        TrendHoldingStrategy(),
        RegimeGatedTrendHoldingStrategy(),
        RegimeGatedPortfolioGovernorStrategy(),
    )
    results = run_benchmark_suite(
        execution_config,
        cached_data=frames,
        progress_callback=progress_callback,
        strategies=strategies,
    )
    payload = portfolio_risk_governor_report_payload(results, frames[execution_config.primary_timeframe])
    path = write_portfolio_risk_governor_output(execution_config.output_dir, payload)
    return {
        "portfolio_risk_governor_report": payload,
        "artifacts": {"portfolio_risk_governor_report": str(path)},
    }


def run_hybrid_trend_rider_analysis(
    config: Any,
    cached_data: dict[str, pd.DataFrame] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run Phase 1.10 Hybrid Trend Rider comparison."""

    history_config = replace(config, end=_extend_end_for_profit_capture(config.end))
    execution_config = replace(config, close_open_position_on_end=True)
    frames = _load_required_frames(history_config, cached_data)
    strategies: tuple[BenchmarkStrategy, ...] = (
        AgentAggressiveStrategy(),
        TrendRiderAggressiveStrategy(),
        HybridTrendRiderStrategy(),
    )
    results = run_benchmark_suite(
        execution_config,
        cached_data=frames,
        progress_callback=progress_callback,
        strategies=strategies,
    )
    payload = hybrid_trend_rider_report_payload(results, frames[execution_config.primary_timeframe])
    path = write_hybrid_trend_rider_output(execution_config.output_dir, payload)
    return {
        "hybrid_trend_rider_report": payload,
        "artifacts": {"hybrid_trend_rider_report": str(path)},
    }


def run_hybrid_runner_optimization(
    config: Any,
    focus_strategy: str | None = None,
    cached_data: dict[str, pd.DataFrame] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run Phase 1.11 profiled Hybrid runner optimization."""

    history_config = replace(config, end=_extend_end_for_profit_capture(config.end))
    execution_config = replace(config, close_open_position_on_end=True)
    frames = _load_required_frames(history_config, cached_data)
    strategies = _hybrid_runner_strategies(focus_strategy)
    results = run_benchmark_suite(
        execution_config,
        cached_data=frames,
        progress_callback=progress_callback,
        strategies=strategies,
    )
    payload = hybrid_runner_optimization_payload(results, frames[execution_config.primary_timeframe], focus_strategy)
    path = write_hybrid_runner_optimization_output(execution_config.output_dir, payload)
    return {
        "hybrid_runner_optimization": payload,
        "artifacts": {"hybrid_runner_optimization": str(path)},
    }


def _hybrid_runner_strategies(focus_strategy: str | None = None) -> tuple[BenchmarkStrategy, ...]:
    strategies: dict[str, BenchmarkStrategy] = {
        "agent_aggressive": AgentAggressiveStrategy(),
        "trend_rider_aggressive": TrendRiderAggressiveStrategy(),
        "hybrid_conservative": HybridTrendRiderStrategy("hybrid_conservative"),
        "hybrid_balanced": HybridTrendRiderStrategy("hybrid_balanced"),
        "hybrid_aggressive": HybridTrendRiderStrategy("hybrid_aggressive"),
    }
    if focus_strategy is None:
        return tuple(strategies.values())
    if focus_strategy not in strategies:
        supported = ", ".join(sorted(strategies))
        raise ValueError(f"Unsupported hybrid optimization strategy {focus_strategy!r}. Supported: {supported}.")
    return (strategies[focus_strategy],)


def run_market_structure_stop_analysis(
    config: Any,
    focus_stop_type: str | None = None,
    cached_data: dict[str, pd.DataFrame] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run Phase 1.12 aggressive stop-placement comparison."""

    history_config = replace(config, end=_extend_end_for_profit_capture(config.end))
    execution_config = replace(config, stop_type="fixed")
    frames = _load_required_frames(history_config, cached_data)
    strategies: tuple[BenchmarkStrategy, ...] = (
        AgentStopStrategy("aggressive_current", "fixed"),
        AgentStopStrategy("aggressive_atr_stop", "atr"),
        AgentStopStrategy("aggressive_swing_low_stop", "swing_low"),
        AgentStopStrategy("aggressive_support_zone_stop", "support_zone"),
    )
    results = run_benchmark_suite(
        execution_config,
        cached_data=frames,
        progress_callback=progress_callback,
        strategies=strategies,
    )
    payload = market_structure_stop_payload(results, frames[execution_config.primary_timeframe], focus_stop_type)
    path = write_market_structure_stop_output(execution_config.output_dir, payload)
    return {
        "market_structure_stop_report": payload,
        "artifacts": {"market_structure_stop_report": str(path)},
    }


def market_structure_stop_payload(
    results: dict[str, BenchmarkResult],
    price_history: pd.DataFrame,
    focus_stop_type: str | None = None,
) -> dict[str, Any]:
    capture = profit_capture_payload(results, price_history)
    targets = {
        "return_pct": 120.0,
        "profit_capture_ratio": 0.10,
        "max_drawdown_pct": 20.0,
        "sharpe_ratio": 0.80,
    }
    baseline = results.get("aggressive_current")
    survived_counts = _survived_stopout_counts(baseline)
    survival_analysis = {
        "atr_stop_survivals": survived_counts.get("atr", 0),
        "swing_stop_survivals": survived_counts.get("swing_low", 0),
        "support_zone_survivals": survived_counts.get("support_zone", 0),
    }
    rows: dict[str, Any] = {}
    for name, result in results.items():
        capture_metrics = capture["strategies"][name]
        row = {metric: result.metrics.get(metric) for metric in MARKET_STRUCTURE_STOP_METRICS}
        row.update(
            {
                "profit_capture_ratio": capture_metrics.get("profit_capture_ratio"),
                "total_trades": result.metrics.get("total_trades"),
                "stop_out_count": result.metrics.get("stop_out_count", _stop_out_count(result)),
                "stop_type_usage": result.metrics.get("stop_type_usage", _stop_type_usage(result)),
                "average_stop_distance_pct": result.metrics.get("average_stop_distance_pct"),
                "average_stop_distance_atr": result.metrics.get("average_stop_distance_atr"),
                "average_atr_distance": result.metrics.get("average_stop_distance_atr"),
                "survived_stopouts_count": survived_counts.get(_stop_type_from_strategy_name(name), 0),
                "profit_capture": capture_metrics,
            }
        )
        row["target_assessment"] = {
            "return_target_met": _target_gt(row.get("total_return_pct"), targets["return_pct"]),
            "profit_capture_target_met": _target_gt(row.get("profit_capture_ratio"), targets["profit_capture_ratio"]),
            "drawdown_target_met": _target_lt(row.get("max_drawdown_pct"), targets["max_drawdown_pct"]),
            "sharpe_target_met": _target_gt(row.get("sharpe_ratio"), targets["sharpe_ratio"]),
        }
        row["target_assessment"]["all_targets_met"] = all(row["target_assessment"].values())
        rows[name] = row

    rankings = {
        "by_profit_capture": _rank_metric(rows, "profit_capture_ratio", reverse=True),
        "by_return": _rank_metric(rows, "total_return_pct", reverse=True),
        "by_sharpe": _rank_metric(rows, "sharpe_ratio", reverse=True),
        "by_drawdown": _rank_metric(rows, "max_drawdown_pct", reverse=False),
    }
    passing = [name for name, row in rows.items() if name != "aggressive_current" and row["target_assessment"]["all_targets_met"]]
    closest = passing[0] if passing else _closest_stop_strategy(rows, targets)
    best_profit_capture = _best_strategy_by_metric(rows, "profit_capture_ratio", include_baseline=True, reverse=True)
    best_risk_adjusted = _best_strategy_by_metric(rows, "sharpe_ratio", include_baseline=True, reverse=True)
    return {
        "goal": "Reduce premature stop-outs during major BTC trends by widening stops with market structure.",
        "focus_stop_type": focus_stop_type,
        "targets": targets,
        "stop_definitions": {
            "fixed": "Existing decision stop, kept as the benchmark.",
            "atr": "entry_price - 1.5 ATR.",
            "swing_low": "lowest recent swing low over 20 candles minus 0.5 ATR, with a 1.5 ATR minimum distance.",
            "support_zone": "support zone low minus 0.5 ATR, with a 1.5 ATR minimum distance.",
        },
        "baseline_stop_out_count": _stop_out_count(baseline) if baseline is not None else 0,
        "stop_out_survival_analysis": survival_analysis,
        "rankings": rankings,
        "strategies_meeting_targets": passing,
        "closest_strategy": closest,
        "best_stop_type": _strategy_summary(rows, closest),
        "best_profit_capture": _strategy_summary(rows, best_profit_capture),
        "best_risk_adjusted_return": _strategy_summary(rows, best_risk_adjusted),
        "recommended_production_configuration": _recommended_stop_configuration(rows, baseline, closest),
        "recommendation": _market_structure_stop_recommendation(rows, closest),
        "strategies": rows,
    }


def write_market_structure_stop_output(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "market_structure_stop_report.json"
    _write_json(path, payload)
    return path


def hybrid_runner_optimization_payload(
    results: dict[str, BenchmarkResult],
    price_history: pd.DataFrame,
    focus_strategy: str | None = None,
) -> dict[str, Any]:
    base = hybrid_trend_rider_report_payload(results, price_history)
    targets = {
        "return_pct": 120.0,
        "max_drawdown_pct": 25.0,
        "profit_capture_ratio": 0.15,
        "sharpe_ratio": 0.90,
    }
    rows = base["strategies"]
    for row in rows.values():
        _attach_target_flags(row, targets)
        row["average_trend_duration_captured_hours"] = row["average_runner_holding_hours"]

    rankings = {
        "by_sharpe": _rank_metric(rows, "sharpe_ratio", reverse=True),
        "by_return": _rank_metric(rows, "total_return_pct", reverse=True),
        "by_profit_capture": _rank_metric(rows, "profit_capture_ratio", reverse=True),
        "by_drawdown": _rank_metric(rows, "max_drawdown_pct", reverse=False),
    }
    hybrid_profiles = ("hybrid_conservative", "hybrid_balanced", "hybrid_aggressive")
    passing_profiles = [
        name
        for name in hybrid_profiles
        if name in rows and rows[name].get("target_assessment", {}).get("all_targets_met")
    ]
    closest_profile = passing_profiles[0] if passing_profiles else _closest_hybrid_profile(rows, targets, hybrid_profiles)
    return {
        "goal": "Find the optimal balance between trend capture, drawdown control, and risk-adjusted returns.",
        "focus_strategy": focus_strategy,
        "targets": targets,
        "profile_definitions": {
            "hybrid_conservative": {
                "runner_size": 0.25,
                "tp1": "+2R",
                "tp2": "+4R",
                "runner_exits": ["daily_rsi < 50", "daily_ema20 < daily_ema50", "15% trailing stop"],
            },
            "hybrid_balanced": {
                "runner_size": 0.40,
                "tp1": "+2R",
                "tp2": "+4R",
                "runner_exits": ["daily_rsi < 45", "daily_ema50 < daily_ema100", "20% trailing stop"],
            },
            "hybrid_aggressive": {
                "runner_size": 0.50,
                "tp1": "+2R",
                "tp2": "+4R",
                "runner_exits": ["weekly_rsi < 45", "weekly_ema20 < weekly_ema50", "25% trailing stop"],
            },
        },
        "rankings": rankings,
        "profiles_meeting_all_targets": passing_profiles,
        "closest_profile": closest_profile,
        "recommendation": _hybrid_optimization_recommendation(rows, closest_profile),
        "strategies": rows,
    }


def write_hybrid_runner_optimization_output(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "hybrid_runner_optimization.json"
    _write_json(path, payload)
    return path


def hybrid_trend_rider_report_payload(
    results: dict[str, BenchmarkResult],
    price_history: pd.DataFrame,
) -> dict[str, Any]:
    capture = profit_capture_payload(results, price_history)
    targets = {
        "return_pct": 120.0,
        "max_drawdown_pct": 25.0,
        "profit_capture_ratio": 0.15,
    }
    rows: dict[str, Any] = {}
    for name, result in results.items():
        capture_metrics = capture["strategies"][name]
        ratio = _float_or_none(capture_metrics.get("profit_capture_ratio"))
        runner_returns = _trade_numeric_values(result.trades, "runner_return_pct")
        runner_hours = _trade_numeric_values(result.trades, "runner_holding_hours")
        runner_drawdowns = _trade_numeric_values(result.trades, "runner_max_drawdown_pct")
        row = {metric: result.metrics.get(metric) for metric in HYBRID_REPORT_METRICS}
        row.update(
            {
                "profit_capture_ratio": ratio,
                "average_runner_holding_hours": _round_or_none(_mean(runner_hours)),
                "median_runner_holding_hours": _round_or_none(_median(runner_hours)),
                "average_runner_return_pct": _round_or_none(_mean(runner_returns)),
                "max_runner_return_pct": _round_or_none(max(runner_returns), 2) if runner_returns else None,
                "tp1_hit_count": _metric_or_partial_count(result, "tp1_hit_count", "TP1_2R"),
                "tp2_hit_count": _metric_or_partial_count(result, "tp2_hit_count", "TP2_4R"),
                "runner_activation_count": _metric_or_count(result, "runner_activation_count", _partial_count(result.trades, "TP2_4R")),
                "runner_exit_reasons": _runner_exit_reasons(result),
                "average_runner_drawdown_pct": _metric_or_mean(result, "average_runner_drawdown_pct", runner_drawdowns),
                "max_runner_drawdown_pct": _metric_or_max(result, "max_runner_drawdown_pct", runner_drawdowns),
                "profit_capture": capture_metrics,
            }
        )
        rows[name] = row

    hybrid = rows.get("hybrid_trend_rider", {})
    return_pct = _float_or_none(hybrid.get("total_return_pct"))
    drawdown = _float_or_none(hybrid.get("max_drawdown_pct"))
    capture_ratio = _float_or_none(hybrid.get("profit_capture_ratio"))
    target_assessment = {
        "return_target_met": return_pct is not None and return_pct >= targets["return_pct"],
        "drawdown_target_met": drawdown is not None and drawdown < targets["max_drawdown_pct"],
        "profit_capture_target_met": capture_ratio is not None and capture_ratio > targets["profit_capture_ratio"],
    }
    target_assessment["all_targets_met"] = all(target_assessment.values())
    return {
        "goal": "Improve returns from Trend Rider while reducing drawdown.",
        "targets": targets,
        "target_assessment": target_assessment,
        "hybrid_rules": {
            "entry_rules": "existing aggressive entry rules",
            "allocation_per_trade": 0.25,
            "tp1": {"+R": 2, "close_pct": 50, "stop_after_hit": "breakeven"},
            "tp2": {"+R": 4, "close_pct": 25},
            "runner_pct": 25,
            "runner_exits": [
                "daily_rsi < 50",
                "daily_ema20 < daily_ema50",
                "daily_price < daily_ema50",
                "15% trailing stop from highest close after runner activation",
                "runner drawdown > 25%",
            ],
        },
        "strategies": rows,
    }


def write_hybrid_trend_rider_output(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "hybrid_trend_rider_report.json"
    _write_json(path, payload)
    return path


def trend_rider_analysis_payload(
    results: dict[str, BenchmarkResult],
    price_history: pd.DataFrame,
) -> dict[str, Any]:
    capture = profit_capture_payload(results, price_history)
    target_ratio = 0.10
    rows: dict[str, Any] = {}
    for name, result in results.items():
        capture_metrics = capture["strategies"][name]
        ratio = _float_or_none(capture_metrics.get("profit_capture_ratio"))
        runner_returns = [
            float(trade["runner_return_pct"])
            for trade in result.trades
            if trade.get("runner_return_pct") is not None
        ]
        runner_hours = [
            float(trade["runner_holding_hours"])
            for trade in result.trades
            if trade.get("runner_holding_hours") is not None
        ]
        rows[name] = {
            "total_return_pct": result.metrics.get("total_return_pct"),
            "cagr": result.metrics.get("cagr"),
            "max_drawdown_pct": result.metrics.get("max_drawdown_pct"),
            "sharpe_ratio": result.metrics.get("sharpe_ratio"),
            "total_trades": result.metrics.get("total_trades"),
            "profit_capture_ratio": ratio,
            "profit_capture_target_met": ratio is not None and ratio > target_ratio,
            "average_runner_return_pct": _round_or_none(_mean(runner_returns)),
            "average_runner_holding_hours": _round_or_none(_mean(runner_hours)),
            "profit_capture": capture_metrics,
        }
    return {
        "goal": "Increase profit capture ratio.",
        "target_profit_capture_ratio": target_ratio,
        "trend_rider_rules": {
            "entry_position_pct": 100,
            "tp1": {"+R": 2, "close_pct": 50},
            "tp2": {"+R": 4, "close_pct": 25},
            "runner_pct": 25,
            "runner_exits": [
                "daily_rsi < 50",
                "daily_ema20 < daily_ema50",
                "10% trailing stop",
            ],
        },
        "strategies": rows,
    }


def trend_holding_report_payload(
    results: dict[str, BenchmarkResult],
    price_history: pd.DataFrame,
) -> dict[str, Any]:
    capture = profit_capture_payload(results, price_history)
    targets = {
        "return_pct": 120.0,
        "profit_capture_ratio": 0.10,
        "max_drawdown_pct": 20.0,
        "sharpe_ratio": 0.80,
    }
    rows: dict[str, Any] = {}
    for name, result in results.items():
        capture_metrics = capture["strategies"][name]
        row = {
            "total_return_pct": result.metrics.get("total_return_pct"),
            "cagr": result.metrics.get("cagr"),
            "max_drawdown_pct": result.metrics.get("max_drawdown_pct"),
            "sharpe_ratio": result.metrics.get("sharpe_ratio"),
            "profit_factor": result.metrics.get("profit_factor"),
            "total_trades": result.metrics.get("total_trades"),
            "win_rate": result.metrics.get("win_rate"),
            "profit_capture_ratio": capture_metrics.get("profit_capture_ratio"),
            "average_runner_return_pct": _runner_metric(result, "runner_return_pct", "mean"),
            "max_runner_return_pct": _runner_metric(result, "runner_return_pct", "max"),
            "average_runner_holding_hours": _runner_metric(result, "runner_holding_hours", "mean"),
            "median_runner_holding_hours": _runner_metric(result, "runner_holding_hours", "median"),
            "tp1_hits": result.metrics.get("tp1_hits", result.metrics.get("tp1_hit_count", 0)),
            "tp2_hits": result.metrics.get("tp2_hits", result.metrics.get("tp2_hit_count", 0)),
            "runner_activations": result.metrics.get("runner_activations", result.metrics.get("runner_activation_count", 0)),
            "runner_exit_reasons": result.metrics.get("runner_exit_reasons", {}),
            "profit_capture": capture_metrics,
        }
        row["target_assessment"] = {
            "return_target_met": _target_gt(row.get("total_return_pct"), targets["return_pct"]),
            "profit_capture_target_met": _target_gt(row.get("profit_capture_ratio"), targets["profit_capture_ratio"]),
            "drawdown_target_met": _target_lt(row.get("max_drawdown_pct"), targets["max_drawdown_pct"]),
            "sharpe_target_met": _target_gt(row.get("sharpe_ratio"), targets["sharpe_ratio"]),
        }
        row["target_assessment"]["all_targets_met"] = all(row["target_assessment"].values())
        rows[name] = row

    deltas = _strategy_delta(rows.get("agent_aggressive"), rows.get("trend_holding"))
    missed_recheck = _missed_opportunity_recheck(
        results.get("agent_aggressive"),
        results.get("trend_holding"),
        price_history,
        limit=50,
    )
    return {
        "goal": "Increase profit capture without materially increasing drawdown.",
        "targets": targets,
        "trend_holding_rules": {
            "tp1": {"+R": 2, "close_pct": 50, "stop_action": "move stop to breakeven"},
            "tp2": {"+R": 4, "close_pct": 25, "runner_active": True},
            "runner_pct": 25,
            "runner_exits": [
                "daily_close < daily_ema50",
                "daily_macd bearish",
                "20% trailing stop from highest close after runner activation",
            ],
        },
        "comparison": {
            "return_delta_pct": deltas.get("total_return_pct"),
            "drawdown_delta_pct": deltas.get("max_drawdown_pct"),
            "sharpe_delta": deltas.get("sharpe_ratio"),
            "profit_capture_delta": deltas.get("profit_capture_ratio"),
        },
        "missed_opportunity_recheck": missed_recheck,
        "recommended_configuration": _trend_holding_recommendation(rows, targets),
        "strategies": rows,
    }


def regime_gated_trend_holding_report_payload(
    results: dict[str, BenchmarkResult],
    price_history: pd.DataFrame,
) -> dict[str, Any]:
    capture = profit_capture_payload(results, price_history)
    targets = {
        "return_pct": 120.0,
        "profit_capture_ratio": 0.10,
        "max_drawdown_pct": 25.0,
        "sharpe_ratio": 0.80,
    }
    rows: dict[str, Any] = {}
    for name, result in results.items():
        capture_metrics = capture["strategies"][name]
        row = {
            "total_return_pct": result.metrics.get("total_return_pct"),
            "cagr": result.metrics.get("cagr"),
            "max_drawdown_pct": result.metrics.get("max_drawdown_pct"),
            "sharpe_ratio": result.metrics.get("sharpe_ratio"),
            "profit_factor": result.metrics.get("profit_factor"),
            "total_trades": result.metrics.get("total_trades"),
            "win_rate": result.metrics.get("win_rate"),
            "profit_capture_ratio": capture_metrics.get("profit_capture_ratio"),
            "runner_activation_count": result.metrics.get("runner_activation_count", result.metrics.get("runner_activations", 0)),
            "runner_disabled_count": result.metrics.get("runner_disabled_count", 0),
            "strong_bull_periods": result.metrics.get("strong_bull_periods", 0),
            "bull_periods": result.metrics.get("bull_periods", 0),
            "range_periods": result.metrics.get("range_periods", 0),
            "bear_periods": result.metrics.get("bear_periods", 0),
            "average_runner_return_pct": _runner_metric(result, "runner_return_pct", "mean"),
            "max_runner_return_pct": _runner_metric(result, "runner_return_pct", "max"),
            "average_runner_holding_hours": _runner_metric(result, "runner_holding_hours", "mean"),
            "median_runner_holding_hours": _runner_metric(result, "runner_holding_hours", "median"),
            "runner_exit_reasons": result.metrics.get("runner_exit_reasons", {}),
            "profit_capture": capture_metrics,
        }
        row["target_assessment"] = {
            "return_target_met": _target_gt(row.get("total_return_pct"), targets["return_pct"]),
            "profit_capture_target_met": _target_gt(row.get("profit_capture_ratio"), targets["profit_capture_ratio"]),
            "drawdown_target_met": _target_lt(row.get("max_drawdown_pct"), targets["max_drawdown_pct"]),
            "sharpe_target_met": _target_gt(row.get("sharpe_ratio"), targets["sharpe_ratio"]),
        }
        row["target_assessment"]["all_targets_met"] = all(row["target_assessment"].values())
        rows[name] = row

    missed_recheck = _missed_opportunity_recheck(
        results.get("agent_aggressive"),
        results.get("regime_gated_trend_holding"),
        price_history,
        limit=50,
    )
    return {
        "goal": "Capture trends only during strong bull regimes while controlling drawdown.",
        "targets": targets,
        "regime_rules": {
            "strong_bull": [
                "daily_ema20 > daily_ema50 > daily_ema200",
                "daily_rsi > 55",
                "daily_macd bullish",
                "weekly_close > weekly_ema20",
            ],
            "bull": ["daily_ema20 > daily_ema50", "daily_rsi > 50"],
            "range": ["daily_ema20 and daily_ema50 crossing frequently", "daily_rsi between 45 and 55"],
            "bear": ["daily_ema20 < daily_ema50 < daily_ema200", "weekly_close < weekly_ema20"],
        },
        "risk_controls": {
            "drawdown_above_15_pct": "runner allocation reduced by 50%",
            "drawdown_above_20_pct": "new runners disabled; standard Agent Aggressive exits used",
        },
        "comparison": {
            "trend_holding_vs_agent": _strategy_delta(rows.get("agent_aggressive"), rows.get("trend_holding")),
            "regime_gated_vs_agent": _strategy_delta(rows.get("agent_aggressive"), rows.get("regime_gated_trend_holding")),
            "regime_gated_vs_trend_holding": _strategy_delta(rows.get("trend_holding"), rows.get("regime_gated_trend_holding")),
        },
        "missed_opportunity_recheck": {
            **missed_recheck,
            "survived_count": missed_recheck.get("runner_survived_count", 0),
        },
        "recommendation": _regime_gated_recommendation(rows, targets),
        "strategies": rows,
    }


def portfolio_risk_governor_report_payload(
    results: dict[str, BenchmarkResult],
    price_history: pd.DataFrame,
) -> dict[str, Any]:
    capture = profit_capture_payload(results, price_history)
    targets = {
        "return_pct": 120.0,
        "profit_capture_ratio": 0.10,
        "max_drawdown_pct": 25.0,
        "sharpe_ratio": 0.80,
    }
    rows: dict[str, Any] = {}
    for name, result in results.items():
        capture_metrics = capture["strategies"][name]
        row = {
            "total_return_pct": result.metrics.get("total_return_pct"),
            "cagr": result.metrics.get("cagr"),
            "max_drawdown_pct": result.metrics.get("max_drawdown_pct"),
            "sharpe_ratio": result.metrics.get("sharpe_ratio"),
            "profit_factor": result.metrics.get("profit_factor"),
            "total_trades": result.metrics.get("total_trades"),
            "win_rate": result.metrics.get("win_rate"),
            "profit_capture_ratio": capture_metrics.get("profit_capture_ratio"),
            "risk_state_counts": result.metrics.get("risk_state_counts", {}),
            "average_position_size": result.metrics.get("average_position_size"),
            "average_runner_size": result.metrics.get("average_runner_size"),
            "portfolio_stop_count": result.metrics.get("portfolio_stop_count", 0),
            "defensive_mode_hours": result.metrics.get("defensive_mode_hours", 0.0),
            "runner_activation_count": result.metrics.get("runner_activation_count", result.metrics.get("runner_activations", 0)),
            "runner_disabled_count": result.metrics.get("runner_disabled_count", 0),
            "profit_capture": capture_metrics,
        }
        row["target_assessment"] = {
            "return_target_met": _target_gt(row.get("total_return_pct"), targets["return_pct"]),
            "profit_capture_target_met": _target_gt(row.get("profit_capture_ratio"), targets["profit_capture_ratio"]),
            "drawdown_target_met": _target_lt(row.get("max_drawdown_pct"), targets["max_drawdown_pct"]),
            "sharpe_target_met": _target_gt(row.get("sharpe_ratio"), targets["sharpe_ratio"]),
        }
        row["target_assessment"]["all_targets_met"] = all(row["target_assessment"].values())
        rows[name] = row

    rankings = {
        "by_sharpe": _rank_metric(rows, "sharpe_ratio", reverse=True),
        "by_drawdown": _rank_metric(rows, "max_drawdown_pct", reverse=False),
        "by_return": _rank_metric(rows, "total_return_pct", reverse=True),
    }
    return {
        "goal": "Preserve trend capture while reducing portfolio drawdown below 25%.",
        "targets": targets,
        "risk_state_rules": {
            "NORMAL": {"drawdown": "< 10%", "allocation": "100%", "runner": "enabled"},
            "CAUTION": {"drawdown": ">= 10%", "allocation": "75%", "runner": "enabled"},
            "DEFENSIVE": {"drawdown": ">= 15%", "allocation": "50%", "runner": "disabled"},
            "CAPITAL_PRESERVATION": {
                "drawdown": ">= 20%",
                "allocation": "25%",
                "runner": "disabled",
                "trend_holding": "disabled",
            },
        },
        "position_sizing": {
            "risk_per_trade": "1% of current equity",
            "formula": "position_size = risk_amount / (entry_price - stop_price)",
            "volatility_adjustment": "if ATR > ATR_MA, multiply size by ATR_MA / ATR clamped to 0.25-1.0",
        },
        "portfolio_stop": {
            "trigger": "drawdown > 25%",
            "actions": ["close active runners", "disable new runners", "switch to defensive sizing"],
            "recovery": "portfolio stop deactivates once drawdown < 15%",
        },
        "rankings": rankings,
        "comparison": {
            "trend_holding_vs_agent": _strategy_delta(rows.get("agent_aggressive"), rows.get("trend_holding")),
            "regime_gated_vs_agent": _strategy_delta(rows.get("agent_aggressive"), rows.get("regime_gated_trend_holding")),
            "governor_vs_agent": _strategy_delta(rows.get("agent_aggressive"), rows.get("regime_gated_portfolio_governor")),
            "governor_vs_regime_gated": _strategy_delta(
                rows.get("regime_gated_trend_holding"),
                rows.get("regime_gated_portfolio_governor"),
            ),
        },
        "recommendation": _portfolio_governor_recommendation(rows, targets),
        "strategies": rows,
    }


def write_trend_rider_output(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "trend_rider_analysis.json"
    _write_json(path, payload)
    return path


def write_trend_holding_output(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "trend_holding_report.json"
    _write_json(path, payload)
    return path


def write_regime_gated_trend_holding_output(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "regime_gated_trend_holding_report.json"
    _write_json(path, payload)
    return path


def write_portfolio_risk_governor_output(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "portfolio_risk_governor_report.json"
    _write_json(path, payload)
    return path


def profit_capture_payload(
    results: dict[str, BenchmarkResult],
    price_history: pd.DataFrame,
    windows_days: tuple[int, ...] = (7, 30, 90),
) -> dict[str, Any]:
    """Analyze holding duration, missed upside, and profit capture for closed trades."""

    history = _prepare_price_history(price_history)
    return {
        "goal": "Measure whether exits are too early.",
        "missed_opportunity_windows_days": list(windows_days),
        "profit_capture_definition": (
            "sum of positive captured trade gains divided by sum of maximum positive gains "
            "available from entry through 90 days after exit"
        ),
        "strategies": {
            name: _profit_capture_strategy_payload(result, history, windows_days)
            for name, result in results.items()
        },
    }


def write_profit_capture_output(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "profit_capture_analysis.json"
    _write_json(path, payload)
    return path


def trend_participation_payload(results: dict[str, BenchmarkResult]) -> dict[str, Any]:
    rows = {
        name: {metric: result.metrics.get(metric) for metric in TREND_PARTICIPATION_METRICS}
        for name, result in results.items()
    }
    drawdown_limit_pct = 20.0
    for row in rows.values():
        drawdown = _float_or_none(row.get("max_drawdown_pct"))
        row["drawdown_under_20_pct"] = drawdown is not None and drawdown < drawdown_limit_pct
    ranking = _rank_by_risk_adjusted_return(rows)
    passing_ranking = [name for name in ranking if rows[name]["drawdown_under_20_pct"]]
    return {
        "goal": "Increase trend participation while keeping drawdown under 20%.",
        "bull_mode_rules": {
            "active_when": ["daily_rsi > 55", "daily_close > daily_ema200"],
            "min_rr_ratio": 1.2,
            "min_volume_ratio": 0.5,
            "allowed_alignments": ["BULLISH_ALIGNMENT", "PULLBACK_IN_UPTREND"],
        },
        "drawdown_limit_pct": drawdown_limit_pct,
        "best_strategy_by_risk_adjusted_return": ranking[0] if ranking else None,
        "best_strategy_under_drawdown_limit": passing_ranking[0] if passing_ranking else None,
        "ranking": ranking,
        "strategies": rows,
    }


def write_trend_participation_output(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "trend_participation_report.json"
    _write_json(path, payload)
    return path


def _profit_capture_strategy_payload(
    result: BenchmarkResult,
    price_history: pd.DataFrame,
    windows_days: tuple[int, ...],
) -> dict[str, Any]:
    trades = [_trade_analysis(trade, price_history, windows_days) for trade in result.trades]
    holding_hours = [trade["holding_hours"] for trade in trades if trade["holding_hours"] is not None]
    winning_trades = [trade for trade in trades if trade["return_pct"] is not None and trade["return_pct"] > 0]
    losing_trades = [trade for trade in trades if trade["return_pct"] is not None and trade["return_pct"] < 0]
    missed_summaries = _missed_gain_summary(trades, windows_days)
    maximum_available = sum(float(trade["maximum_trend_gain_pct"]) for trade in trades)
    captured = sum(float(trade["captured_gain_pct"]) for trade in trades)
    capture_ratio = captured / maximum_available if maximum_available > 0 else None

    return {
        "total_trades": len(trades),
        "average_holding_hours": _round_or_none(_mean(holding_hours)),
        "median_holding_hours": _round_or_none(_median(holding_hours)),
        "longest_winning_trade": _max_by(winning_trades, "holding_hours"),
        "longest_losing_trade": _max_by(losing_trades, "holding_hours"),
        "top_10_winning_trades": _top_n(winning_trades, "return_pct", 10),
        "top_10_missed_opportunities": _top_n(trades, "max_missed_gain_pct", 10),
        "missed_opportunity_summary": missed_summaries,
        "profit_captured_pct_sum": round(captured, 2),
        "maximum_trend_profit_available_pct_sum": round(maximum_available, 2),
        "profit_capture_ratio": _round_or_none(capture_ratio, digits=4),
    }


def _trade_analysis(
    trade: dict[str, Any],
    price_history: pd.DataFrame,
    windows_days: tuple[int, ...],
) -> dict[str, Any]:
    entry_timestamp = pd.Timestamp(trade["entry_timestamp"])
    exit_timestamp = pd.Timestamp(trade["exit_timestamp"])
    entry_price = float(trade["entry_price"])
    exit_price = float(trade["exit_price"])
    holding_hours = (exit_timestamp - entry_timestamp).total_seconds() / 3600
    return_pct = float(trade.get("return_pct", ((exit_price / entry_price) - 1) * 100))
    captured_gain_pct = _captured_gain_pct(trade, entry_price, exit_price)
    missed_by_window = {
        f"{window_days}d": _missed_gain_after_exit(price_history, exit_timestamp, exit_price, window_days)
        for window_days in windows_days
    }
    max_missed_gain_pct = max(
        (value["missed_gain_pct"] for value in missed_by_window.values() if value["missed_gain_pct"] is not None),
        default=0.0,
    )
    maximum_trend_high = _max_high_between(
        price_history,
        entry_timestamp,
        exit_timestamp + pd.Timedelta(days=max(windows_days)),
    )
    maximum_trend_gain_pct = (
        max(0.0, ((maximum_trend_high / entry_price) - 1) * 100)
        if maximum_trend_high is not None
        else captured_gain_pct
    )
    return {
        "entry_timestamp": entry_timestamp.isoformat(),
        "exit_timestamp": exit_timestamp.isoformat(),
        "entry_price": round(entry_price, 2),
        "exit_price": round(exit_price, 2),
        "return_pct": round(return_pct, 2),
        "pnl": round(float(trade.get("pnl", 0.0)), 2),
        "exit_reason": trade.get("exit_reason"),
        "holding_hours": round(holding_hours, 2),
        "holding_days": round(holding_hours / 24, 2),
        "captured_gain_pct": round(captured_gain_pct, 2),
        "maximum_trend_gain_pct": round(maximum_trend_gain_pct, 2),
        "max_missed_gain_pct": round(float(max_missed_gain_pct), 2),
        "missed_opportunity": missed_by_window,
    }


def _captured_gain_pct(trade: dict[str, Any], entry_price: float, exit_price: float) -> float:
    partial_exits = trade.get("partial_exits") or []
    if not partial_exits:
        return max(0.0, ((exit_price / entry_price) - 1) * 100)

    captured = 0.0
    closed_fraction = 0.0
    for partial_exit in partial_exits:
        fraction = max(0.0, float(partial_exit.get("position_fraction", 0.0)))
        partial_price = float(partial_exit.get("price", exit_price))
        captured += fraction * max(0.0, ((partial_price / entry_price) - 1) * 100)
        closed_fraction += fraction

    runner_fraction = max(0.0, 1.0 - closed_fraction)
    runner_gain = max(0.0, ((exit_price / entry_price) - 1) * 100)
    return captured + (runner_fraction * runner_gain)


def _runner_metric(result: BenchmarkResult, field: str, aggregation: str) -> float | None:
    values = [
        float(trade[field])
        for trade in result.trades
        if trade.get(field) is not None
    ]
    if aggregation == "max":
        return _round_or_none(max(values) if values else None)
    if aggregation == "median":
        return _round_or_none(_median(values))
    return _round_or_none(_mean(values))


def _strategy_delta(
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> dict[str, float | None]:
    if baseline is None or candidate is None:
        return {}
    deltas: dict[str, float | None] = {}
    for metric in ("total_return_pct", "max_drawdown_pct", "sharpe_ratio", "profit_capture_ratio"):
        base_value = _float_or_none(baseline.get(metric))
        candidate_value = _float_or_none(candidate.get(metric))
        deltas[metric] = _round_or_none(candidate_value - base_value, digits=4) if base_value is not None and candidate_value is not None else None
    return deltas


def _missed_opportunity_recheck(
    baseline: BenchmarkResult | None,
    trend_holding: BenchmarkResult | None,
    price_history: pd.DataFrame,
    limit: int = 50,
) -> dict[str, Any]:
    if baseline is None or trend_holding is None:
        return {
            "sample_size": 0,
            "runner_survived_count": 0,
            "runner_captured_count": 0,
            "additional_profit_captured_pct": 0.0,
            "trades": [],
        }
    history = _prepare_price_history(price_history)
    baseline_trades = [_trade_analysis(trade, history, (7, 30, 90)) for trade in baseline.trades]
    top_missed = _top_n(baseline_trades, "max_missed_gain_pct", limit)
    trend_by_entry = {
        pd.Timestamp(trade["entry_timestamp"]).isoformat(): trade
        for trade in trend_holding.trades
    }
    rows: list[dict[str, Any]] = []
    additional_profit = 0.0
    runner_survived_count = 0
    runner_captured_count = 0
    for trade in top_missed:
        matched = trend_by_entry.get(pd.Timestamp(trade["entry_timestamp"]).isoformat())
        trend_analysis = _trade_analysis(matched, history, (7, 30, 90)) if matched is not None else None
        baseline_capture = float(trade.get("captured_gain_pct") or 0.0)
        trend_capture = float(trend_analysis.get("captured_gain_pct") or 0.0) if trend_analysis is not None else 0.0
        additional = max(0.0, trend_capture - baseline_capture)
        runner_survived = bool(matched and matched.get("runner_exit_price") is not None)
        runner_captured = runner_survived and additional > 0
        runner_survived_count += int(runner_survived)
        runner_captured_count += int(runner_captured)
        additional_profit += additional
        rows.append(
            {
                "entry_timestamp": trade["entry_timestamp"],
                "baseline_exit_timestamp": trade["exit_timestamp"],
                "baseline_exit_reason": trade["exit_reason"],
                "baseline_captured_gain_pct": trade["captured_gain_pct"],
                "max_missed_gain_pct": trade["max_missed_gain_pct"],
                "trend_holding_exit_timestamp": trend_analysis["exit_timestamp"] if trend_analysis is not None else None,
                "trend_holding_exit_reason": matched.get("exit_reason") if matched is not None else None,
                "runner_stayed_alive": runner_survived,
                "runner_captured_trend": runner_captured,
                "additional_profit_captured_pct": round(additional, 2),
            }
        )
    return {
        "sample_size": len(rows),
        "runner_survived_count": runner_survived_count,
        "runner_captured_count": runner_captured_count,
        "additional_profit_captured_pct": round(additional_profit, 2),
        "trades": rows,
    }


def _trend_holding_recommendation(rows: dict[str, dict[str, Any]], targets: dict[str, float]) -> dict[str, Any]:
    candidate = rows.get("trend_holding")
    baseline = rows.get("agent_aggressive")
    if candidate is None:
        return {
            "strategy": None,
            "reason": "Trend Holding Engine did not produce comparable metrics.",
        }
    if candidate.get("target_assessment", {}).get("all_targets_met"):
        return {
            "strategy": "trend_holding",
            "reason": "Trend Holding Engine met all Phase 1.13 targets.",
        }
    deltas = _strategy_delta(baseline, candidate)
    capture_delta = _float_or_none(deltas.get("profit_capture_ratio")) or 0.0
    drawdown = _float_or_none(candidate.get("max_drawdown_pct"))
    sharpe = _float_or_none(candidate.get("sharpe_ratio"))
    if capture_delta > 0 and drawdown is not None and drawdown < targets["max_drawdown_pct"] and sharpe is not None and sharpe >= targets["sharpe_ratio"]:
        return {
            "strategy": "trend_holding",
            "reason": "Trend Holding improved profit capture while staying inside drawdown and Sharpe limits.",
        }
    return {
        "strategy": "agent_aggressive",
        "reason": "Keep Agent Aggressive as production default until Trend Holding improves capture without weakening risk-adjusted performance.",
    }


def _regime_gated_recommendation(rows: dict[str, dict[str, Any]], targets: dict[str, float]) -> dict[str, Any]:
    candidate = rows.get("regime_gated_trend_holding")
    baseline = rows.get("agent_aggressive")
    trend_holding = rows.get("trend_holding")
    if candidate is None:
        return {
            "production_default": "agent_aggressive",
            "reason": "Regime-Gated Trend Holding did not produce comparable metrics.",
        }
    if candidate.get("target_assessment", {}).get("all_targets_met"):
        return {
            "production_default": "regime_gated_trend_holding",
            "reason": "Regime-Gated Trend Holding met all Phase 1.14 targets.",
        }
    baseline_capture = _float_or_none(baseline.get("profit_capture_ratio")) if baseline else None
    candidate_capture = _float_or_none(candidate.get("profit_capture_ratio"))
    candidate_drawdown = _float_or_none(candidate.get("max_drawdown_pct"))
    candidate_sharpe = _float_or_none(candidate.get("sharpe_ratio"))
    trend_drawdown = _float_or_none(trend_holding.get("max_drawdown_pct")) if trend_holding else None
    if (
        baseline_capture is not None
        and candidate_capture is not None
        and candidate_capture > baseline_capture
        and candidate_drawdown is not None
        and candidate_drawdown < targets["max_drawdown_pct"]
        and candidate_sharpe is not None
        and candidate_sharpe >= targets["sharpe_ratio"]
    ):
        return {
            "production_default": "regime_gated_trend_holding",
            "reason": "Regime gate improved profit capture versus Agent Aggressive while meeting drawdown and Sharpe constraints.",
        }
    if trend_drawdown is not None and candidate_drawdown is not None and candidate_drawdown < trend_drawdown:
        return {
            "production_default": "agent_aggressive",
            "candidate": "regime_gated_trend_holding",
            "reason": "Regime gate reduced drawdown versus raw Trend Holding, but it did not meet the full production target set.",
        }
    return {
        "production_default": "agent_aggressive",
        "reason": "Keep Agent Aggressive as production default until the gated runner meets return, capture, drawdown, and Sharpe targets.",
    }


def _portfolio_governor_recommendation(rows: dict[str, dict[str, Any]], targets: dict[str, float]) -> dict[str, Any]:
    candidate = rows.get("regime_gated_portfolio_governor")
    baseline = rows.get("agent_aggressive")
    regime_gated = rows.get("regime_gated_trend_holding")
    if candidate is None:
        return {
            "production_default": "agent_aggressive",
            "reason": "Portfolio Governor did not produce comparable metrics.",
        }
    if candidate.get("target_assessment", {}).get("all_targets_met"):
        return {
            "production_default": "regime_gated_portfolio_governor",
            "reason": "Portfolio Governor met the Phase 1.15 return, Sharpe, profit-capture, and drawdown targets.",
        }

    candidate_drawdown = _float_or_none(candidate.get("max_drawdown_pct"))
    candidate_sharpe = _float_or_none(candidate.get("sharpe_ratio"))
    baseline_sharpe = _float_or_none(baseline.get("sharpe_ratio")) if baseline else None
    regime_drawdown = _float_or_none(regime_gated.get("max_drawdown_pct")) if regime_gated else None
    if (
        candidate_drawdown is not None
        and regime_drawdown is not None
        and candidate_drawdown < regime_drawdown
        and candidate_drawdown < targets["max_drawdown_pct"]
        and candidate_sharpe is not None
        and baseline_sharpe is not None
        and candidate_sharpe >= baseline_sharpe
    ):
        return {
            "production_default": "regime_gated_portfolio_governor",
            "reason": "Portfolio Governor reduced drawdown below target and preserved risk-adjusted return versus Agent Aggressive.",
        }
    return {
        "production_default": "agent_aggressive",
        "candidate": "regime_gated_portfolio_governor",
        "reason": "Keep Agent Aggressive as production default until the governor meets the full return, Sharpe, capture, and drawdown target set.",
    }


def _missed_gain_after_exit(
    price_history: pd.DataFrame,
    exit_timestamp: pd.Timestamp,
    exit_price: float,
    window_days: int,
) -> dict[str, Any]:
    window_end = exit_timestamp + pd.Timedelta(days=window_days)
    future = price_history[
        (price_history["timestamp"] > exit_timestamp)
        & (price_history["timestamp"] <= window_end)
    ]
    if future.empty:
        return {
            "max_price": None,
            "max_timestamp": None,
            "missed_gain_pct": None,
            "data_points": 0,
        }
    high_index = future["high"].astype(float).idxmax()
    max_price = float(future.loc[high_index, "high"])
    missed_gain_pct = max(0.0, ((max_price / exit_price) - 1) * 100)
    return {
        "max_price": round(max_price, 2),
        "max_timestamp": pd.Timestamp(future.loc[high_index, "timestamp"]).isoformat(),
        "missed_gain_pct": round(missed_gain_pct, 2),
        "data_points": int(len(future)),
    }


def _missed_gain_summary(trades: list[dict[str, Any]], windows_days: tuple[int, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for window_days in windows_days:
        key = f"{window_days}d"
        values = [
            float(trade["missed_opportunity"][key]["missed_gain_pct"])
            for trade in trades
            if trade["missed_opportunity"][key]["missed_gain_pct"] is not None
        ]
        summary[key] = {
            "average_missed_gain_pct": _round_or_none(_mean(values)),
            "max_missed_gain_pct": _round_or_none(max(values) if values else None),
            "sample_size": len(values),
        }
    return summary


def _max_high_between(
    price_history: pd.DataFrame,
    start_timestamp: pd.Timestamp,
    end_timestamp: pd.Timestamp,
) -> float | None:
    frame = price_history[
        (price_history["timestamp"] >= start_timestamp)
        & (price_history["timestamp"] <= end_timestamp)
    ]
    if frame.empty:
        return None
    return float(frame["high"].max())


def _prepare_price_history(price_history: pd.DataFrame) -> pd.DataFrame:
    frame = price_history.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["high"] = frame["high"].astype(float)
    return frame.sort_values("timestamp").reset_index(drop=True)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _trade_numeric_values(trades: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for trade in trades:
        value = _float_or_none(trade.get(key))
        if value is not None:
            values.append(value)
    return values


def _metric_or_count(result: BenchmarkResult, metric_name: str, fallback: int) -> int:
    value = result.metrics.get(metric_name)
    if value is None:
        return fallback
    return int(value)


def _metric_or_partial_count(result: BenchmarkResult, metric_name: str, reason: str) -> int:
    return _metric_or_count(result, metric_name, _partial_count(result.trades, reason))


def _partial_count(trades: list[dict[str, Any]], reason: str) -> int:
    count = 0
    for trade in trades:
        for partial_exit in trade.get("partial_exits") or []:
            if partial_exit.get("reason") == reason:
                count += 1
    return count


def _runner_exit_reasons(result: BenchmarkResult) -> dict[str, int]:
    metric_reasons = result.metrics.get("runner_exit_reasons")
    if isinstance(metric_reasons, dict):
        return {str(reason): int(count) for reason, count in metric_reasons.items()}
    reasons = {reason: 0 for reason in RUNNER_EXIT_REASONS}
    for trade in result.trades:
        reason = str(trade.get("exit_reason", ""))
        if reason in reasons and trade.get("runner_return_pct") is not None:
            reasons[reason] += 1
    return reasons


def _metric_or_mean(result: BenchmarkResult, metric_name: str, values: list[float]) -> float:
    value = _float_or_none(result.metrics.get(metric_name))
    if value is not None:
        return round(value, 2)
    mean = _mean(values)
    return round(mean, 2) if mean is not None else 0.0


def _metric_or_max(result: BenchmarkResult, metric_name: str, values: list[float]) -> float:
    value = _float_or_none(result.metrics.get(metric_name))
    if value is not None:
        return round(value, 2)
    return round(max(values), 2) if values else 0.0


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2


def _max_by(trades: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    if not trades:
        return None
    return max(trades, key=lambda trade: float(trade.get(key) or 0.0))


def _top_n(trades: list[dict[str, Any]], key: str, limit: int) -> list[dict[str, Any]]:
    return sorted(
        trades,
        key=lambda trade: float(trade.get(key) or 0.0),
        reverse=True,
    )[:limit]


def strategy_research_report_payload(
    benchmark_payload: dict[str, Any],
    regime_payload: dict[str, Any],
    attribution_payload: dict[str, Any],
) -> dict[str, Any]:
    best_benchmark = benchmark_payload.get("best_strategy_by_risk_adjusted_return")
    agent_regimes = _agent_regime_rows(regime_payload)
    best_regime = max(agent_regimes, key=lambda row: row["return_pct"], default=None)
    worst_regime = min(agent_regimes, key=lambda row: row["return_pct"], default=None)
    expensive_filter = attribution_payload.get("most_expensive_filter")
    return {
        "best_benchmark": best_benchmark,
        "best_regime": best_regime,
        "worst_regime": worst_regime,
        "most_expensive_filter": expensive_filter,
        "recommended_next_optimization": _recommend_next_optimization(expensive_filter),
    }


def write_phase16_outputs(
    output_dir: Path,
    benchmark_payload: dict[str, Any],
    regime_payload: dict[str, Any],
    attribution_payload: dict[str, Any],
    final_report: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "benchmark_comparison": output_dir / "benchmark_comparison.json",
        "regime_analysis": output_dir / "regime_analysis.json",
        "filter_attribution": output_dir / "filter_attribution.json",
        "strategy_research_report": output_dir / "strategy_research_report.json",
    }
    _write_json(paths["benchmark_comparison"], benchmark_payload)
    _write_json(paths["regime_analysis"], regime_payload)
    _write_json(paths["filter_attribution"], attribution_payload)
    _write_json(paths["strategy_research_report"], final_report)
    return paths


def _run_agent_with_profile(
    config: Any,
    frames: dict[str, pd.DataFrame],
    profile: StrategyProfile,
    progress_callback: ProgressCallback | None,
) -> Any:
    from backtesting.backtest_engine import run_backtest

    return run_backtest(
        replace(
            config,
            profile="aggressive",
            strategy_profile_override=profile,
        ),
        cached_data=frames,
        progress_callback=progress_callback,
    )


def _load_required_frames(config: Any, cached_data: dict[str, pd.DataFrame] | None) -> dict[str, pd.DataFrame]:
    from backtesting.backtest_engine import load_or_download_timeframes

    frames = dict(cached_data or {})
    timeframes = _ordered_timeframes(config.primary_timeframe, (*config.timeframes, "1d"))
    missing = tuple(timeframe for timeframe in timeframes if timeframe not in frames)
    if missing:
        frames.update(load_or_download_timeframes(config, missing))
    return frames


def _extend_end_for_profit_capture(end: str) -> str:
    if end == "latest":
        return end
    timestamp = pd.Timestamp(end)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return (timestamp + pd.Timedelta(days=90)).isoformat()


def _ordered_timeframes(primary_timeframe: str, timeframes: tuple[str, ...]) -> tuple[str, ...]:
    ordered = [primary_timeframe]
    for timeframe in timeframes:
        if timeframe not in ordered:
            ordered.append(timeframe)
    return tuple(ordered)


def _regime_metrics(equity_curve: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    frame = equity_curve.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame[frame["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end != "latest":
        frame = frame[frame["timestamp"] <= pd.Timestamp(end, tz="UTC")]
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if len(frame) < 2:
        return {
            "return_pct": None,
            "max_drawdown_pct": None,
            "sharpe_ratio": None,
            "data_points": len(frame),
        }
    initial_capital = float(frame.iloc[0]["current_equity"])
    price_column = "price" if "price" in frame.columns else "current_equity"
    metrics = calculate_performance_metrics(
        equity_curve=frame,
        trades=[],
        initial_capital=initial_capital,
        start_price=float(frame.iloc[0][price_column]),
        end_price=float(frame.iloc[-1][price_column]),
    )
    return {
        "return_pct": metrics["total_return_pct"],
        "max_drawdown_pct": metrics["max_drawdown_pct"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "data_points": len(frame),
    }


def _rank_by_risk_adjusted_return(rows: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        rows,
        key=lambda name: (
            _finite_or_floor(rows[name].get("sharpe_ratio")),
            -_finite_or_floor(rows[name].get("max_drawdown_pct")),
            _finite_or_floor(rows[name].get("cagr")),
        ),
        reverse=True,
    )


def _rank_metric(rows: dict[str, dict[str, Any]], metric: str, reverse: bool) -> list[str]:
    return sorted(
        rows,
        key=lambda name: _finite_or_floor(rows[name].get(metric)),
        reverse=reverse,
    )


def _attach_target_flags(row: dict[str, Any], targets: dict[str, float]) -> None:
    return_pct = _float_or_none(row.get("total_return_pct"))
    drawdown = _float_or_none(row.get("max_drawdown_pct"))
    capture = _float_or_none(row.get("profit_capture_ratio"))
    sharpe = _float_or_none(row.get("sharpe_ratio"))
    target_assessment = {
        "return_target_met": return_pct is not None and return_pct > targets["return_pct"],
        "drawdown_target_met": drawdown is not None and drawdown < targets["max_drawdown_pct"],
        "profit_capture_target_met": capture is not None and capture > targets["profit_capture_ratio"],
        "sharpe_target_met": sharpe is not None and sharpe > targets["sharpe_ratio"],
    }
    target_assessment["all_targets_met"] = all(target_assessment.values())
    row["target_assessment"] = target_assessment


def _closest_hybrid_profile(
    rows: dict[str, dict[str, Any]],
    targets: dict[str, float],
    hybrid_profiles: tuple[str, ...],
) -> str | None:
    candidates = [name for name in hybrid_profiles if name in rows]
    if not candidates:
        return None
    return max(candidates, key=lambda name: _target_closeness_score(rows[name], targets))


def _target_closeness_score(row: dict[str, Any], targets: dict[str, float]) -> tuple[float, float, float]:
    return_pct = max(0.0, _float_or_none(row.get("total_return_pct")) or 0.0)
    drawdown = max(0.0, _float_or_none(row.get("max_drawdown_pct")) or 0.0)
    capture = max(0.0, _float_or_none(row.get("profit_capture_ratio")) or 0.0)
    sharpe = max(0.0, _float_or_none(row.get("sharpe_ratio")) or 0.0)
    score = (
        min(return_pct / targets["return_pct"], 1.0)
        + (1.0 if drawdown == 0 else min(targets["max_drawdown_pct"] / drawdown, 1.0))
        + min(capture / targets["profit_capture_ratio"], 1.0)
        + min(sharpe / targets["sharpe_ratio"], 1.0)
    )
    return score, sharpe, -drawdown


def _hybrid_optimization_recommendation(rows: dict[str, dict[str, Any]], closest_profile: str | None) -> dict[str, Any]:
    if closest_profile is None:
        return {
            "profile": None,
            "reason": "No Hybrid profile produced comparable metrics.",
        }
    row = rows[closest_profile]
    assessment = row.get("target_assessment", {})
    if assessment.get("all_targets_met"):
        reason = "This profile met all Phase 1.11 targets."
    else:
        missed = [
            label
            for key, label in (
                ("return_target_met", "return"),
                ("drawdown_target_met", "drawdown"),
                ("profit_capture_target_met", "profit capture"),
                ("sharpe_target_met", "Sharpe"),
            )
            if not assessment.get(key)
        ]
        reason = f"This profile is closest to the target set but still misses: {', '.join(missed)}."
    return {
        "profile": closest_profile,
        "reason": reason,
        "metrics": {
            "total_return_pct": row.get("total_return_pct"),
            "max_drawdown_pct": row.get("max_drawdown_pct"),
            "profit_capture_ratio": row.get("profit_capture_ratio"),
            "sharpe_ratio": row.get("sharpe_ratio"),
        },
    }


def _stop_type_from_strategy_name(strategy_name: str) -> str:
    mapping = {
        "aggressive_current": "fixed",
        "aggressive_atr_stop": "atr",
        "aggressive_swing_low_stop": "swing_low",
        "aggressive_support_zone_stop": "support_zone",
    }
    return mapping.get(strategy_name, strategy_name)


def _survived_stopout_counts(baseline: BenchmarkResult | None) -> dict[str, int]:
    counts = {"fixed": 0, "atr": 0, "swing_low": 0, "support_zone": 0}
    if baseline is None:
        return counts
    for trade in baseline.trades:
        if trade.get("exit_reason") != "STOP_LOSS":
            continue
        exit_price = _float_or_none(trade.get("exit_price"))
        candidates = trade.get("entry_stop_candidates") or {}
        if exit_price is None or not isinstance(candidates, dict):
            continue
        for stop_type in ("atr", "swing_low", "support_zone"):
            candidate = candidates.get(stop_type) or {}
            stop_price = _float_or_none(candidate.get("stop_price")) if isinstance(candidate, dict) else None
            if stop_price is not None and exit_price > stop_price:
                counts[stop_type] += 1
    return counts


def _stop_out_count(result: BenchmarkResult | None) -> int:
    if result is None:
        return 0
    metric_count = result.metrics.get("stop_out_count")
    if metric_count is not None:
        return int(metric_count)
    return sum(1 for trade in result.trades if trade.get("exit_reason") == "STOP_LOSS")


def _stop_type_usage(result: BenchmarkResult) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trade in result.trades:
        stop_type = str(trade.get("entry_stop_type") or "UNKNOWN")
        counts[stop_type] = counts.get(stop_type, 0) + 1
    return counts


def _target_gt(value: Any, target: float) -> bool:
    numeric = _float_or_none(value)
    return numeric is not None and numeric > target


def _target_lt(value: Any, target: float) -> bool:
    numeric = _float_or_none(value)
    return numeric is not None and numeric < target


def _closest_stop_strategy(rows: dict[str, dict[str, Any]], targets: dict[str, float]) -> str | None:
    candidates = [name for name in rows if name != "aggressive_current"]
    if not candidates:
        return None
    return max(candidates, key=lambda name: _stop_target_score(rows[name], targets))


def _stop_target_score(row: dict[str, Any], targets: dict[str, float]) -> tuple[float, float, float]:
    capture = max(0.0, _float_or_none(row.get("profit_capture_ratio")) or 0.0)
    drawdown = max(0.0, _float_or_none(row.get("max_drawdown_pct")) or 0.0)
    sharpe = max(0.0, _float_or_none(row.get("sharpe_ratio")) or 0.0)
    total_return = max(0.0, _float_or_none(row.get("total_return_pct")) or 0.0)
    score = (
        min(total_return / targets["return_pct"], 1.0)
        + min(capture / targets["profit_capture_ratio"], 1.0)
        + (1.0 if drawdown == 0 else min(targets["max_drawdown_pct"] / drawdown, 1.0))
        + min(sharpe / targets["sharpe_ratio"], 1.0)
    )
    return score, sharpe, total_return


def _best_strategy_by_metric(
    rows: dict[str, dict[str, Any]],
    metric: str,
    *,
    include_baseline: bool,
    reverse: bool,
) -> str | None:
    candidates = [name for name in rows if include_baseline or name != "aggressive_current"]
    if not candidates:
        return None
    return sorted(candidates, key=lambda name: _finite_or_floor(rows[name].get(metric)), reverse=reverse)[0]


def _strategy_summary(rows: dict[str, dict[str, Any]], strategy: str | None) -> dict[str, Any]:
    if strategy is None or strategy not in rows:
        return {"strategy": None}
    row = rows[strategy]
    return {
        "strategy": strategy,
        "stop_type": _stop_type_from_strategy_name(strategy),
        "total_return_pct": row.get("total_return_pct"),
        "profit_capture_ratio": row.get("profit_capture_ratio"),
        "max_drawdown_pct": row.get("max_drawdown_pct"),
        "sharpe_ratio": row.get("sharpe_ratio"),
        "stop_out_count": row.get("stop_out_count"),
        "stop_type_usage": row.get("stop_type_usage"),
    }


def _recommended_stop_configuration(
    rows: dict[str, dict[str, Any]],
    baseline: BenchmarkResult | None,
    closest_strategy: str | None,
) -> dict[str, Any]:
    baseline_row = rows.get("aggressive_current") if baseline is not None else None
    candidate_row = rows.get(closest_strategy) if closest_strategy is not None else None
    if candidate_row is None:
        return {
            "stop_type": "fixed",
            "reason": "No market-structure stop experiment produced comparable results.",
        }
    if candidate_row.get("target_assessment", {}).get("all_targets_met"):
        return {
            "stop_type": _stop_type_from_strategy_name(closest_strategy),
            "strategy": closest_strategy,
            "reason": "This stop type met all Phase 1.12 targets.",
        }
    if baseline_row is not None:
        candidate_score = (
            _float_or_none(candidate_row.get("profit_capture_ratio")) or 0.0,
            _float_or_none(candidate_row.get("sharpe_ratio")) or 0.0,
            _float_or_none(candidate_row.get("total_return_pct")) or 0.0,
        )
        baseline_score = (
            _float_or_none(baseline_row.get("profit_capture_ratio")) or 0.0,
            _float_or_none(baseline_row.get("sharpe_ratio")) or 0.0,
            _float_or_none(baseline_row.get("total_return_pct")) or 0.0,
        )
        candidate_drawdown = _float_or_none(candidate_row.get("max_drawdown_pct"))
        if candidate_score > baseline_score and candidate_drawdown is not None and candidate_drawdown < 20.0:
            return {
                "stop_type": _stop_type_from_strategy_name(closest_strategy),
                "strategy": closest_strategy,
                "reason": "This stop type improved profit capture/risk-adjusted metrics versus the baseline while staying under the drawdown limit.",
            }
    return {
        "stop_type": "fixed",
        "strategy": "aggressive_current",
        "reason": "Keep the current aggressive stop configuration until a market-structure stop beats the baseline target mix.",
    }


def _market_structure_stop_recommendation(rows: dict[str, dict[str, Any]], closest_strategy: str | None) -> dict[str, Any]:
    if closest_strategy is None:
        return {
            "strategy": None,
            "reason": "No market-structure stop strategy produced comparable metrics.",
        }
    row = rows[closest_strategy]
    assessment = row.get("target_assessment", {})
    if assessment.get("all_targets_met"):
        reason = "This stop strategy met all Phase 1.12 targets."
    else:
        missed = [
            label
            for key, label in (
                ("return_target_met", "return"),
                ("profit_capture_target_met", "profit capture"),
                ("drawdown_target_met", "drawdown"),
                ("sharpe_target_met", "Sharpe"),
            )
            if not assessment.get(key)
        ]
        reason = f"This stop strategy is closest to the target set but still misses: {', '.join(missed)}."
    return {
        "strategy": closest_strategy,
        "stop_type": _stop_type_from_strategy_name(closest_strategy),
        "reason": reason,
        "metrics": {
            "total_return_pct": row.get("total_return_pct"),
            "max_drawdown_pct": row.get("max_drawdown_pct"),
            "profit_capture_ratio": row.get("profit_capture_ratio"),
            "sharpe_ratio": row.get("sharpe_ratio"),
            "stop_out_count": row.get("stop_out_count"),
            "survived_stopouts_count": row.get("survived_stopouts_count"),
        },
    }


def _most_expensive_filter(results: dict[str, Any]) -> dict[str, Any] | None:
    candidates = []
    for experiment, row in results.items():
        if experiment == "base":
            continue
        delta = _float_or_none(row.get("return_delta_pct"))
        if delta is None:
            continue
        candidates.append((experiment, delta, _float_or_none(row.get("drawdown_delta_pct"))))
    if not candidates:
        return None
    experiment, delta, drawdown_delta = max(candidates, key=lambda item: item[1])
    if delta <= 0:
        return None
    return {
        "experiment": experiment,
        "return_delta_pct": round(delta, 2),
        "drawdown_delta_pct": None if drawdown_delta is None else round(drawdown_delta, 2),
        "interpretation": "Removing this filter improved return the most versus Base.",
    }


def _agent_regime_rows(regime_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for regime_name, regime in regime_payload.get("regimes", {}).items():
        metrics = regime.get("strategies", {}).get("agent_aggressive", {})
        return_pct = _float_or_none(metrics.get("return_pct"))
        if return_pct is None:
            continue
        rows.append(
            {
                "regime": regime_name,
                "return_pct": return_pct,
                "max_drawdown_pct": metrics.get("max_drawdown_pct"),
                "sharpe_ratio": metrics.get("sharpe_ratio"),
            }
        )
    return rows


def _recommend_next_optimization(expensive_filter: dict[str, Any] | None) -> str:
    if expensive_filter is None:
        return "No removed filter improved return; focus next on setup quality and exit timing instead of loosening gates."
    experiment = expensive_filter["experiment"]
    if experiment == "no_alignment_filter":
        return "Review the alignment gate and add a controlled pullback-continuation exception before loosening it globally."
    if experiment == "no_rr_filter":
        return "Tune the RR threshold by setup type; a single hard RR floor may be rejecting too many continuation trades."
    if experiment == "no_volume_filter":
        return "Tune the volume threshold by regime; volume confirmation may be too strict in quieter bullish trends."
    if experiment == "no_macd_filter":
        return "MACD confirmation appears costly; prefer slower 4h confirmation or make it regime-specific."
    return "Use the highest-return attribution experiment as the next controlled optimization target."


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _finite_or_floor(value: Any) -> float:
    numeric = _float_or_none(value)
    if numeric is None:
        return float("-inf")
    return numeric if math.isfinite(numeric) else float("-inf")


def _float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _round_or_none(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(value, digits)
