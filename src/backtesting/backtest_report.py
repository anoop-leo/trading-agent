"""Backtest artifact writer for Phase 1.5."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from backtesting.trade_simulator import TRADE_COLUMNS


def write_backtest_report(result: Any, output_dir: Path, write_chart: bool = True) -> dict[str, Path]:
    """Write JSON, trade CSV, equity CSV, and optional chart artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "backtest_report.json"
    trades_path = output_dir / "backtest_trades.csv"
    equity_path = output_dir / "equity_curve.csv"
    chart_path = output_dir / "equity_curve.png"

    report = {
        "symbol": result.symbol,
        "profile": getattr(result, "profile", None),
        "strategy_profile": getattr(result, "strategy_profile", None),
        "start_date": result.start_date,
        "end_date": result.end_date,
        "initial_capital": result.initial_capital,
        **result.metrics,
    }
    report_path.write_text(json.dumps(_json_safe(report), indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(result.trades, columns=TRADE_COLUMNS).to_csv(trades_path, index=False)
    result.equity_curve.to_csv(equity_path, index=False)

    paths = {
        "report": report_path,
        "trades": trades_path,
        "equity_curve": equity_path,
    }
    if write_chart:
        _write_equity_chart(result.equity_curve, chart_path)
        paths["chart"] = chart_path
    return paths


COMPARISON_METRICS = (
    "total_return_pct",
    "cagr",
    "max_drawdown_pct",
    "profit_factor",
    "sharpe_ratio",
    "win_rate",
    "expectancy",
    "total_trades",
)

BENCHMARK_COMPARISON_METRICS = (
    "total_return_pct",
    "cagr",
    "max_drawdown_pct",
    "sharpe_ratio",
    "total_trades",
)


def write_profile_comparison(results: dict[str, Any], output_dir: Path) -> Path:
    """Write a profile comparison report ranked by risk-adjusted performance."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "profile_comparison.json"
    profile_rows = {
        profile: {
            metric: result.metrics.get(metric)
            for metric in COMPARISON_METRICS
        }
        for profile, result in results.items()
    }
    ranked_profiles = sorted(
        profile_rows,
        key=lambda profile: (
            _finite_or_floor(profile_rows[profile].get("sharpe_ratio")),
            -_finite_or_floor(profile_rows[profile].get("max_drawdown_pct")),
            _finite_or_floor(profile_rows[profile].get("cagr")),
        ),
        reverse=True,
    )
    payload = {
        "selection_method": "highest sharpe_ratio, then lower max_drawdown_pct, then higher CAGR",
        "best_profile_by_risk_adjusted_return": ranked_profiles[0] if ranked_profiles else None,
        "ranking": ranked_profiles,
        "profiles": profile_rows,
    }
    path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n", encoding="utf-8")
    return path


def write_benchmark_comparison(results: dict[str, Any], output_dir: Path) -> Path:
    """Write a benchmark comparison report ranked by risk-adjusted performance."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "benchmark_comparison.json"
    strategy_rows = {
        strategy: {
            metric: result.metrics.get(metric)
            for metric in BENCHMARK_COMPARISON_METRICS
        }
        for strategy, result in results.items()
    }
    ranked_strategies = sorted(
        strategy_rows,
        key=lambda strategy: (
            _finite_or_floor(strategy_rows[strategy].get("sharpe_ratio")),
            -_finite_or_floor(strategy_rows[strategy].get("max_drawdown_pct")),
            _finite_or_floor(strategy_rows[strategy].get("cagr")),
        ),
        reverse=True,
    )
    payload = {
        "selection_method": "highest sharpe_ratio, then lower max_drawdown_pct, then higher CAGR",
        "best_strategy_by_risk_adjusted_return": ranked_strategies[0] if ranked_strategies else None,
        "ranking": ranked_strategies,
        "strategies": strategy_rows,
    }
    path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n", encoding="utf-8")
    return path


def _write_equity_chart(equity_curve: pd.DataFrame, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    frame = equity_curve.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(frame["timestamp"], frame["current_equity"], label="Strategy", color="#2f80ed")
    if "buy_and_hold_equity" in frame.columns:
        ax.plot(frame["timestamp"], frame["buy_and_hold_equity"], label="Buy and Hold", color="#f2994a")
    ax.set_title("Backtest Equity Curve")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _finite_or_floor(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return numeric if math.isfinite(numeric) else float("-inf")
