"""Broker execution-cost validation for Phase 1.18 research."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from backtesting.backtest_engine import BacktestConfig, BacktestResult, load_or_download_timeframes, run_backtest


ProgressCallback = Callable[[dict[str, Any]], None]
BROKER_COST_REPORT = "broker_cost_validation_report.json"
BROKER_COST_RANKINGS = "broker_cost_rankings.json"


@dataclass(frozen=True)
class BrokerCostProfile:
    name: str
    fee_rate: float
    slippage_rate: float

    @property
    def all_in_cost_per_side(self) -> float:
        return self.fee_rate + self.slippage_rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "fee_rate": self.fee_rate,
            "slippage_rate": self.slippage_rate,
            "all_in_cost_per_side": self.all_in_cost_per_side,
            "all_in_cost_per_side_pct": round(self.all_in_cost_per_side * 100, 4),
        }


BROKER_COST_PROFILES: tuple[BrokerCostProfile, ...] = (
    BrokerCostProfile("current_baseline", fee_rate=0.001, slippage_rate=0.0005),
    BrokerCostProfile("coinbase_conservative", fee_rate=0.004, slippage_rate=0.0005),
    BrokerCostProfile("coinbase_maker_like", fee_rate=0.0015, slippage_rate=0.0003),
    BrokerCostProfile("coinbase_high_volume", fee_rate=0.001, slippage_rate=0.0003),
    BrokerCostProfile("robinhood_moderate_spread", fee_rate=0.0, slippage_rate=0.005),
    BrokerCostProfile("robinhood_harsh_spread", fee_rate=0.0, slippage_rate=0.0095),
    BrokerCostProfile("zero_cost_reference", fee_rate=0.0, slippage_rate=0.0),
)

LIVE_READINESS_CRITERIA = {
    "min_sharpe_ratio": 0.80,
    "min_profit_factor": 1.30,
    "max_drawdown_pct": 15.0,
    "requires_positive_total_return": True,
    "max_cost_drag_pct": 40.0,
}


def run_broker_cost_validation(
    config: BacktestConfig,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run Agent Aggressive under the Phase 1.18 broker cost profiles."""

    execution_config = replace(
        config,
        profile="aggressive",
        stop_type="fixed",
        close_open_position_on_end=True,
    )
    timeframes = _ordered_timeframes(execution_config.primary_timeframe, execution_config.timeframes)
    cached_data = load_or_download_timeframes(execution_config, timeframes)
    results: dict[str, BacktestResult] = {}

    for profile in BROKER_COST_PROFILES:
        _emit(progress_callback, {"phase": "broker_cost_profile", "profile": profile.name})
        profile_config = replace(
            execution_config,
            fee_rate=profile.fee_rate,
            slippage_rate=profile.slippage_rate,
        )
        results[profile.name] = run_backtest(
            profile_config,
            cached_data=cached_data,
            progress_callback=progress_callback,
        )

    report = broker_cost_validation_payload(results, BROKER_COST_PROFILES)
    rankings = broker_cost_rankings_payload(report["profiles"])
    report["rankings"] = rankings
    paths = write_broker_cost_validation_outputs(execution_config.output_dir, report, rankings)
    return {
        "broker_cost_validation_report": report,
        "broker_cost_rankings": rankings,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }


def broker_cost_validation_payload(
    results: dict[str, BacktestResult],
    profiles: tuple[BrokerCostProfile, ...] = BROKER_COST_PROFILES,
) -> dict[str, Any]:
    rows = {
        profile.name: broker_cost_profile_row(results[profile.name], profile)
        for profile in profiles
        if profile.name in results
    }
    rankings = broker_cost_rankings_payload(rows)
    return {
        "phase": "1.18",
        "objective": "Validate whether Agent Aggressive remains viable under realistic broker execution costs.",
        "symbol": _first_result_value(results, "symbol"),
        "strategy": "aggressive",
        "stop_type": "fixed",
        "initial_capital": _first_result_value(results, "initial_capital"),
        "start_date": _first_result_value(results, "start_date"),
        "end_date": _first_result_value(results, "end_date"),
        "success_criteria": LIVE_READINESS_CRITERIA,
        "profiles": rows,
        "rankings": rankings,
        "recommendation": broker_cost_recommendation(rows, rankings),
    }


def broker_cost_profile_row(result: BacktestResult, profile: BrokerCostProfile) -> dict[str, Any]:
    trades = result.trades
    total_gross_before = _sum_trade_field(trades, "gross_pnl_before_fees_and_slippage")
    total_fees = _sum_trade_field(trades, "total_fee")
    total_slippage = _sum_trade_field(trades, "total_slippage_cost")
    total_net_pnl = _sum_trade_field(trades, "net_pnl")
    total_cost = total_gross_before - total_net_pnl
    total_trades = int(result.metrics.get("total_trades", len(trades)))
    cost_drag_pct = (total_cost / total_gross_before * 100) if total_gross_before else 0.0
    row = {
        **profile.to_dict(),
        "final_equity": round(_exact_final_equity(result), 6),
        "total_return_pct": result.metrics.get("total_return_pct"),
        "total_gross_pnl_before_costs": round(total_gross_before, 6),
        "total_fees": round(total_fees, 6),
        "total_slippage_cost": round(total_slippage, 6),
        "total_net_pnl": round(total_net_pnl, 6),
        "cagr": result.metrics.get("cagr"),
        "sharpe_ratio": result.metrics.get("sharpe_ratio"),
        "max_drawdown_pct": result.metrics.get("max_drawdown_pct"),
        "profit_factor": result.metrics.get("profit_factor"),
        "win_rate": result.metrics.get("win_rate"),
        "total_trades": total_trades,
        "average_net_pnl_per_trade": round(total_net_pnl / total_trades, 6) if total_trades else 0.0,
        "cost_per_trade": round(total_cost / total_trades, 6) if total_trades else 0.0,
        "cost_drag_pct": round(cost_drag_pct, 6),
    }
    row["live_readiness"] = live_readiness_flags(row)
    return row


def broker_cost_rankings_payload(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": sorted(
            rows,
            key=lambda name: (
                _metric(rows[name], "sharpe_ratio"),
                _metric(rows[name], "total_return_pct"),
                -_metric(rows[name], "max_drawdown_pct"),
                _metric(rows[name], "profit_factor"),
                -_metric(rows[name], "cost_drag_pct"),
            ),
            reverse=True,
        ),
        "by_sharpe_ratio": _rank_metric(rows, "sharpe_ratio", reverse=True),
        "by_net_return": _rank_metric(rows, "total_return_pct", reverse=True),
        "by_max_drawdown": _rank_metric(rows, "max_drawdown_pct", reverse=False),
        "by_profit_factor": _rank_metric(rows, "profit_factor", reverse=True),
        "by_cost_drag": _rank_metric(rows, "cost_drag_pct", reverse=False),
    }


def live_readiness_flags(row: dict[str, Any]) -> dict[str, Any]:
    flags = {
        "sharpe_ratio_ok": _metric(row, "sharpe_ratio") >= LIVE_READINESS_CRITERIA["min_sharpe_ratio"],
        "profit_factor_ok": _metric(row, "profit_factor") >= LIVE_READINESS_CRITERIA["min_profit_factor"],
        "max_drawdown_ok": _metric(row, "max_drawdown_pct") <= LIVE_READINESS_CRITERIA["max_drawdown_pct"],
        "positive_return_ok": _metric(row, "total_return_pct") > 0,
        "cost_drag_ok": _metric(row, "cost_drag_pct") <= LIVE_READINESS_CRITERIA["max_cost_drag_pct"],
    }
    flags["live_ready"] = all(flags.values())
    return flags


def broker_cost_recommendation(
    rows: dict[str, dict[str, Any]],
    rankings: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    rankings = rankings or broker_cost_rankings_payload(rows)
    live_tradable = [
        name for name, row in rows.items()
        if row.get("live_readiness", {}).get("live_ready")
    ]
    destructive = _destructive_profile(rows)
    highest_live_cost = max(
        (rows[name]["all_in_cost_per_side"] for name in live_tradable),
        default=None,
    )
    coinbase_live = [
        name for name in live_tradable
        if name.startswith("coinbase") or name == "current_baseline"
    ]
    robinhood_live = [name for name in live_tradable if name.startswith("robinhood")]
    if robinhood_live:
        broker_choice = "Robinhood profile is live-tradable in the tested set, but prefer the best risk-adjusted profile."
    elif coinbase_live:
        broker_choice = "Use Coinbase Advanced only if actual all-in costs stay near the live-ready tested profiles."
    else:
        broker_choice = "No live broker yet; tested broker costs do not meet live-readiness criteria."
    return {
        "live_tradable_profiles": live_tradable,
        "best_overall_profile": rankings["overall"][0] if rankings.get("overall") else None,
        "profile_that_destroys_strategy": destructive,
        "maximum_tested_all_in_cost_per_side": highest_live_cost,
        "maximum_tested_all_in_cost_per_side_pct": round(highest_live_cost * 100, 4)
        if highest_live_cost is not None
        else None,
        "broker_choice": broker_choice,
        "coinbase_advanced_assessment": _broker_family_assessment(rows, "coinbase"),
        "robinhood_assessment": _broker_family_assessment(rows, "robinhood"),
        "do_not_proceed_to_live_trading": True,
    }


def write_broker_cost_validation_outputs(
    output_dir: Path,
    report: dict[str, Any],
    rankings: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "broker_cost_validation_report": output_dir / BROKER_COST_REPORT,
        "broker_cost_rankings": output_dir / BROKER_COST_RANKINGS,
    }
    paths["broker_cost_validation_report"].write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["broker_cost_rankings"].write_text(
        json.dumps(_json_safe(rankings), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def _broker_family_assessment(rows: dict[str, dict[str, Any]], prefix: str) -> dict[str, Any]:
    profile_names = [name for name in rows if name.startswith(prefix)]
    live_profiles = [
        name for name in profile_names
        if rows[name].get("live_readiness", {}).get("live_ready")
    ]
    return {
        "tested_profiles": profile_names,
        "live_tradable_profiles": live_profiles,
        "best_profile": max(
            profile_names,
            key=lambda name: (
                _metric(rows[name], "sharpe_ratio"),
                _metric(rows[name], "total_return_pct"),
                -_metric(rows[name], "cost_drag_pct"),
            ),
        )
        if profile_names
        else None,
    }


def _destructive_profile(rows: dict[str, dict[str, Any]]) -> str | None:
    if not rows:
        return None
    negative_profiles = [name for name, row in rows.items() if _metric(row, "total_return_pct") <= 0]
    candidates = negative_profiles or list(rows)
    return min(
        candidates,
        key=lambda name: (
            _metric(rows[name], "total_return_pct"),
            _metric(rows[name], "sharpe_ratio"),
            -_metric(rows[name], "cost_drag_pct"),
        ),
    )


def _rank_metric(rows: dict[str, dict[str, Any]], metric: str, reverse: bool) -> list[str]:
    return sorted(rows, key=lambda name: _metric(rows[name], metric), reverse=reverse)


def _metric(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None:
        return float("-inf")
    if isinstance(value, float) and math.isnan(value):
        return float("-inf")
    return float(value)


def _sum_trade_field(trades: list[dict[str, Any]], field: str) -> float:
    return sum(float(trade.get(field, 0.0) or 0.0) for trade in trades)


def _exact_final_equity(result: BacktestResult) -> float:
    if result.equity_curve.empty:
        return float(result.final_equity)
    return float(result.equity_curve.iloc[-1]["current_equity"])


def _first_result_value(results: dict[str, BacktestResult], field: str) -> Any:
    if not results:
        return None
    return getattr(next(iter(results.values())), field)


def _ordered_timeframes(primary_timeframe: str, timeframes: tuple[str, ...]) -> tuple[str, ...]:
    ordered = [primary_timeframe]
    for timeframe in timeframes:
        if timeframe not in ordered:
            ordered.append(timeframe)
    return tuple(ordered)


def _emit(progress_callback: ProgressCallback | None, event: dict[str, Any]) -> None:
    if progress_callback is not None:
        progress_callback(event)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
