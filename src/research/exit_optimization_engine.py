"""Phase 1.17 exit optimization research framework.

The engine keeps the production entry, stop, sizing, and filter logic fixed by
using Agent Aggressive baseline entries as the entry schedule. Each research
model then replays only the exit behavior on the same primary candles.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from backtesting.benchmarks.research import profit_capture_payload
from backtesting.benchmarks.strategies import BenchmarkResult
from backtesting.performance_metrics import calculate_performance_metrics
from backtesting.profiles import get_strategy_profile
from trading_agent.indicators import add_indicators


ProgressCallback = Callable[[dict[str, Any]], None]

EXIT_TARGETS = {
    "profit_capture_ratio": 0.30,
    "sharpe_ratio": 0.80,
    "profit_factor": 1.40,
    "max_drawdown_pct": 10.0,
}
EXIT_METRICS = (
    "total_return_pct",
    "cagr",
    "sharpe_ratio",
    "max_drawdown_pct",
    "profit_factor",
    "win_rate",
    "total_trades",
    "profit_capture_ratio",
    "average_holding_days",
)


@dataclass(frozen=True)
class ExitModel:
    name: str
    model_type: str
    description: str
    parameters: dict[str, Any]
    category: str = "single"

    @property
    def is_hybrid(self) -> bool:
        return self.category == "hybrid"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "category": self.category,
            "description": self.description,
            "parameters": self.parameters,
        }


def exit_models() -> tuple[ExitModel, ...]:
    """Return all Phase 1.17 exit models."""

    models = [
        ExitModel(
            name="baseline",
            model_type="baseline",
            description="Current production Agent Aggressive exit logic.",
            parameters={},
        ),
        ExitModel(
            name="ema20_trend_rider",
            model_type="ema20_trend_rider",
            description="Remain in trade until close breaks EMA20 or existing 4h momentum exit fires.",
            parameters={"ema": 20, "include_existing_momentum_exit": True},
        ),
        ExitModel(
            name="ema20_ema50_cross",
            model_type="ema_cross",
            description="Remain in trade until EMA20 crosses below EMA50.",
            parameters={"fast_ema": 20, "slow_ema": 50},
        ),
    ]
    models.extend(
        ExitModel(
            name=f"atr_trailing_{str(multiplier).replace('.', '_')}x",
            model_type="atr_trailing",
            description=f"Trail from highest close by ATR * {multiplier}.",
            parameters={"atr_multiplier": multiplier},
        )
        for multiplier in (2.0, 2.5, 3.0, 4.0)
    )
    models.extend(
        [
            ExitModel(
                name="chandelier_exit",
                model_type="chandelier",
                description="Exit when close falls below highest high(22) minus 3 ATR.",
                parameters={"lookback": 22, "atr_multiplier": 3.0},
            ),
            ExitModel(
                name="partial_profit_trend_ride",
                model_type="partial_profit_trend_ride",
                description="Take 50% at current TP, move stop to breakeven, ride remainder until EMA20 break.",
                parameters={"tp_fraction": 0.50, "runner_exit": "ema20_break"},
            ),
            ExitModel(
                name="multi_target_25_25_50",
                model_type="multi_target",
                description="Take 25% at 1R, 25% at 2R, and ride 50% until EMA20 break.",
                parameters={"targets": [{"r": 1.0, "fraction": 0.25}, {"r": 2.0, "fraction": 0.25}]},
            ),
            ExitModel(
                name="multi_target_33_33_34",
                model_type="multi_target",
                description="Take 33% at 1R, 33% at 2R, and ride 34% until EMA20 break.",
                parameters={"targets": [{"r": 1.0, "fraction": 0.33}, {"r": 2.0, "fraction": 0.33}]},
            ),
            ExitModel(
                name="multi_target_50_25_25",
                model_type="multi_target",
                description="Take 50% at 1R, 25% at 2R, and ride 25% until EMA20 break.",
                parameters={"targets": [{"r": 1.0, "fraction": 0.50}, {"r": 2.0, "fraction": 0.25}]},
            ),
            ExitModel(
                name="trend_strength_adx25",
                model_type="trend_strength",
                description="Stay while ADX > 25 and EMA20 remains above EMA50.",
                parameters={"adx_threshold": 25.0},
            ),
            ExitModel(
                name="volatility_adaptive_exit",
                model_type="volatility_adaptive",
                description="Use ATR percentile to widen trailing exits in high volatility and tighten in low volatility.",
                parameters={"low_percentile_multiplier": 2.0, "mid_percentile_multiplier": 3.0, "high_percentile_multiplier": 4.0},
            ),
        ]
    )
    models.extend(
        ExitModel(
            name=f"hybrid_partial_ema20_atr_{str(multiplier).replace('.', '_')}x",
            model_type="hybrid",
            category="hybrid",
            description=f"Take 50% at current TP, move stop to breakeven, then exit runner on EMA20 break or {multiplier} ATR trail.",
            parameters={"tp_fraction": 0.50, "ema": 20, "atr_multiplier": multiplier},
        )
        for multiplier in (2.5, 3.0, 4.0)
    )
    return tuple(models)


def run_exit_optimization(
    config: Any,
    cached_data: dict[str, pd.DataFrame] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run every Phase 1.17 exit model and write the requested reports."""

    from backtesting.backtest_engine import load_or_download_timeframes, run_backtest

    timeframes = _ordered_timeframes(config.primary_timeframe, tuple([*config.timeframes, "4h", "1d"]))
    frames = dict(cached_data or {})
    missing = tuple(timeframe for timeframe in timeframes if timeframe not in frames)
    if missing:
        frames.update(load_or_download_timeframes(config, missing))

    baseline_config = replace(
        config,
        profile="aggressive",
        strategy_profile_override=None,
        stop_type="fixed",
        collect_stop_candidates=False,
    )
    _emit(progress_callback, {"phase": "exit_model", "model": "baseline"})
    baseline_result = run_backtest(
        baseline_config,
        cached_data=frames,
        progress_callback=progress_callback,
    )
    primary_frame = prepare_exit_frame(frames[config.primary_timeframe], config.primary_timeframe, config.start, config.end)
    four_hour_frame = prepare_exit_frame(frames["4h"], "4h", config.start, config.end) if "4h" in frames else None

    results: dict[str, BenchmarkResult] = {
        "baseline": BenchmarkResult(
            name="baseline",
            symbol=baseline_result.symbol,
            start_date=baseline_result.start_date,
            end_date=baseline_result.end_date,
            initial_capital=baseline_result.initial_capital,
            final_equity=baseline_result.final_equity,
            metrics=baseline_result.metrics,
            trades=baseline_result.trades,
            equity_curve=baseline_result.equity_curve,
        )
    }

    for model in exit_models():
        if model.name == "baseline":
            continue
        _emit(progress_callback, {"phase": "exit_model", "model": model.name})
        results[model.name] = simulate_exit_model(
            model,
            baseline_result.trades,
            primary_frame,
            symbol=config.symbol,
            initial_capital=config.initial_capital,
            fee_rate=config.fee_rate,
            slippage_rate=config.slippage_rate,
            allocation_per_trade=get_strategy_profile("aggressive").allocation_per_trade,
            four_hour_frame=four_hour_frame,
            minimum_hold_hours=48.0,
        )

    report = exit_optimization_report_payload(results, primary_frame, exit_models())
    rankings = exit_model_rankings_payload(report["strategies"])
    paths = write_exit_optimization_outputs(config.output_dir, report, rankings)
    return {
        "exit_optimization_report": report,
        "exit_model_rankings": rankings,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }


def prepare_exit_frame(frame: pd.DataFrame, timeframe: str = "1h", start: str = "2020-01-01", end: str = "latest") -> pd.DataFrame:
    """Return a close-timestamped indicator frame for exit research."""

    enriched = add_indicators(frame).copy()
    enriched["timestamp"] = pd.to_datetime(enriched["timestamp"], utc=True) + _timeframe_delta(timeframe)
    enriched = _filter_date_range(enriched, start, end)
    enriched["atr_14"] = calculate_atr_series(enriched)
    enriched["adx_14"] = calculate_adx_series(enriched)
    enriched["atr_percentile_100"] = rolling_percentile_rank(enriched["atr_14"], window=100)
    return enriched.dropna(subset=["ema_20", "ema_50", "atr_14"]).reset_index(drop=True)


def calculate_atr_series(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window=period, min_periods=period).mean()


def calculate_adx_series(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate a research-local ADX series for exit-only experiments."""

    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = calculate_atr_series(frame, period)
    plus_di = 100 * plus_dm.rolling(period, min_periods=period).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(period, min_periods=period).mean() / atr.replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.rolling(period, min_periods=period).mean()


def rolling_percentile_rank(series: pd.Series, window: int = 100) -> pd.Series:
    return series.rolling(window=window, min_periods=max(10, window // 5)).apply(
        lambda values: float(pd.Series(values).rank(pct=True).iloc[-1]),
        raw=False,
    )


def simulate_exit_model(
    model: ExitModel,
    baseline_trades: list[dict[str, Any]],
    price_frame: pd.DataFrame,
    *,
    symbol: str = "BTCUSDT",
    initial_capital: float = 10000.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    allocation_per_trade: float = 0.30,
    four_hour_frame: pd.DataFrame | None = None,
    minimum_hold_hours: float = 48.0,
) -> BenchmarkResult:
    """Replay a single alternate exit model against baseline entry timestamps."""

    if price_frame.empty:
        raise ValueError("price_frame must not be empty.")
    entries = sorted(baseline_trades, key=lambda trade: pd.Timestamp(trade["entry_timestamp"]))
    cash = float(initial_capital)
    realized_pnl = 0.0
    position: dict[str, Any] | None = None
    entry_index = 0
    skipped_entries = 0
    trades: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    frame = price_frame.sort_values("timestamp").reset_index(drop=True)

    for row_number, row in enumerate(frame.itertuples(index=False), start=0):
        timestamp = pd.Timestamp(row.timestamp)
        close = float(row.close)

        if position is not None:
            _update_position_progress(position, row)
            holding_hours = (timestamp - position["entry_timestamp"]).total_seconds() / 3600
            if holding_hours >= minimum_hold_hours:
                cash += _process_partial_exits(model, position, row)
            exit_reason = _model_exit_reason(
                model,
                position,
                row,
                row_number,
                frame,
                four_hour_frame,
                timestamp,
                minimum_hold_hours,
            )
            if exit_reason is not None:
                trade, cash, realized = _close_position(
                    position,
                    timestamp,
                    close,
                    exit_reason,
                    cash,
                    fee_rate,
                    slippage_rate,
                )
                realized_pnl += realized
                trades.append(trade)
                position = None

        if position is not None:
            while entry_index < len(entries) and pd.Timestamp(entries[entry_index]["entry_timestamp"]) <= timestamp:
                skipped_entries += 1
                entry_index += 1

        if position is None and entry_index < len(entries):
            next_entry_time = pd.Timestamp(entries[entry_index]["entry_timestamp"])
            if next_entry_time <= timestamp:
                position, cash = _open_position(
                    entries[entry_index],
                    row,
                    cash,
                    fee_rate,
                    slippage_rate,
                    allocation_per_trade,
                )
                entry_index += 1

        snapshots.append(
            _snapshot(timestamp, close, cash, realized_pnl, position)
        )

    if position is not None:
        last = frame.iloc[-1]
        trade, cash, realized = _close_position(
            position,
            pd.Timestamp(last["timestamp"]),
            float(last["close"]),
            "END_OF_BACKTEST",
            cash,
            fee_rate,
            slippage_rate,
        )
        realized_pnl += realized
        trades.append(trade)
        snapshots[-1] = _snapshot(pd.Timestamp(last["timestamp"]), float(last["close"]), cash, realized_pnl, None)

    equity_curve = pd.DataFrame(snapshots)
    metrics = calculate_performance_metrics(
        equity_curve=equity_curve,
        trades=trades,
        initial_capital=initial_capital,
        start_price=float(frame.iloc[0]["close"]),
        end_price=float(frame.iloc[-1]["close"]),
    )
    metrics["skipped_entries_while_holding"] = skipped_entries
    metrics["exit_model"] = model.to_dict()
    metrics["average_holding_days"] = round(float(metrics.get("average_holding_hours", 0.0)) / 24, 2)
    metrics["exit_reason_distribution"] = _count_exit_reasons(trades)

    return BenchmarkResult(
        name=model.name,
        symbol=symbol,
        start_date=pd.Timestamp(equity_curve.iloc[0]["timestamp"]).date().isoformat(),
        end_date=pd.Timestamp(equity_curve.iloc[-1]["timestamp"]).date().isoformat(),
        initial_capital=initial_capital,
        final_equity=float(metrics["final_equity"]),
        metrics=metrics,
        trades=trades,
        equity_curve=equity_curve,
    )


def exit_optimization_report_payload(
    results: dict[str, BenchmarkResult],
    price_history: pd.DataFrame,
    models: tuple[ExitModel, ...],
) -> dict[str, Any]:
    capture = profit_capture_payload(results, price_history)
    rows: dict[str, dict[str, Any]] = {}
    model_lookup = {model.name: model for model in models}

    for name, result in results.items():
        capture_metrics = capture["strategies"][name]
        row = {metric: result.metrics.get(metric) for metric in EXIT_METRICS}
        if row.get("average_holding_days") is None:
            average_hours = _optional_float(result.metrics.get("average_holding_hours"))
            row["average_holding_days"] = round(average_hours / 24, 2) if average_hours is not None else None
        row["profit_capture_ratio"] = capture_metrics.get("profit_capture_ratio")
        row["profit_capture"] = capture_metrics
        row["exit_reason_distribution"] = result.metrics.get("exit_reason_distribution", {})
        row["skipped_entries_while_holding"] = result.metrics.get("skipped_entries_while_holding", 0)
        row["model"] = model_lookup.get(name, ExitModel(name, name, "", {})).to_dict()
        row["target_assessment"] = _target_assessment(row)
        rows[name] = row

    ranking = rank_exit_models(rows)
    best_capture = _best_by(rows, "profit_capture_ratio")
    best_sharpe = _best_by(rows, "sharpe_ratio")
    best_hybrid = _best_hybrid(rows)
    recommended = _recommended_exit(rows, ranking)
    baseline = rows.get("baseline", {})
    recommended_row = rows.get(recommended["model"], {})
    return {
        "phase": "1.17",
        "goal": "Increase profit capture ratio while preserving Sharpe, drawdown, and profit factor.",
        "constraints": {
            "entry_logic": "unchanged",
            "market_regime_detection": "unchanged",
            "position_sizing": "unchanged",
            "risk_management": "unchanged",
            "stop_loss_logic": "fixed stop unchanged",
            "support_resistance": "unchanged",
            "signal_generation": "unchanged",
            "modified_surface": "exit logic research only",
        },
        "targets": EXIT_TARGETS,
        "model_definitions": {model.name: model.to_dict() for model in models},
        "ranking_method": [
            "profit_capture_ratio descending",
            "sharpe_ratio descending",
            "total_return_pct descending",
            "max_drawdown_pct ascending",
            "profit_factor descending",
        ],
        "ranking_table": ranking,
        "best_profit_capture_model": _model_summary(rows, best_capture),
        "best_sharpe_model": _model_summary(rows, best_sharpe),
        "best_hybrid_model": _model_summary(rows, best_hybrid),
        "recommended_production_exit": recommended,
        "comparison_vs_current_production": _comparison_vs_baseline(baseline, recommended_row),
        "trend_capture_analysis": {
            name: _trend_capture_analysis(result, price_history, limit=20)
            for name, result in results.items()
        },
        "strategies": rows,
    }


def exit_model_rankings_payload(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "phase": "1.17",
        "ranking_method": [
            "profit_capture_ratio",
            "sharpe_ratio",
            "total_return_pct",
            "max_drawdown_pct",
            "profit_factor",
        ],
        "overall": rank_exit_models(rows),
        "by_profit_capture_ratio": _rank_metric(rows, "profit_capture_ratio", reverse=True),
        "by_sharpe_ratio": _rank_metric(rows, "sharpe_ratio", reverse=True),
        "by_total_return_pct": _rank_metric(rows, "total_return_pct", reverse=True),
        "by_max_drawdown_pct": _rank_metric(rows, "max_drawdown_pct", reverse=False),
        "by_profit_factor": _rank_metric(rows, "profit_factor", reverse=True),
    }


def rank_exit_models(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda name: (
            -_metric_value(rows[name].get("profit_capture_ratio")),
            -_metric_value(rows[name].get("sharpe_ratio")),
            -_metric_value(rows[name].get("total_return_pct")),
            _metric_value(rows[name].get("max_drawdown_pct"), missing=float("inf")),
            -_metric_value(rows[name].get("profit_factor")),
        ),
    )
    return [
        {
            "rank": index,
            "model": name,
            "total_return_pct": rows[name].get("total_return_pct"),
            "cagr": rows[name].get("cagr"),
            "sharpe_ratio": rows[name].get("sharpe_ratio"),
            "max_drawdown_pct": rows[name].get("max_drawdown_pct"),
            "profit_factor": rows[name].get("profit_factor"),
            "win_rate": rows[name].get("win_rate"),
            "total_trades": rows[name].get("total_trades"),
            "profit_capture_ratio": rows[name].get("profit_capture_ratio"),
            "average_holding_days": rows[name].get("average_holding_days"),
            "success_criteria_met": rows[name].get("target_assessment", {}).get("all_targets_met", False),
        }
        for index, name in enumerate(ordered, start=1)
    ]


def write_exit_optimization_outputs(
    output_dir: Path,
    report: dict[str, Any],
    rankings: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "exit_optimization_report": output_dir / "exit_optimization_report.json",
        "exit_model_rankings": output_dir / "exit_model_rankings.json",
    }
    _write_json(paths["exit_optimization_report"], report)
    _write_json(paths["exit_model_rankings"], rankings)
    return paths


def _open_position(
    template: dict[str, Any],
    row: Any,
    cash: float,
    fee_rate: float,
    slippage_rate: float,
    allocation_per_trade: float,
) -> tuple[dict[str, Any], float]:
    raw_entry_price = float(row.close)
    entry_price = raw_entry_price * (1 + slippage_rate)
    entry_budget = cash * allocation_per_trade
    entry_notional = entry_budget / (1 + fee_rate)
    entry_fee = entry_notional * fee_rate
    position_size = entry_notional / entry_price
    stop_loss = _optional_float(template.get("entry_stop_loss"))
    initial_risk = _optional_float(template.get("initial_risk"))
    if initial_risk is None or initial_risk <= 0:
        initial_risk = entry_price - stop_loss if stop_loss is not None and stop_loss < entry_price else 0.0
    target_1 = _optional_float(template.get("entry_target_1"))
    if target_1 is None and initial_risk > 0:
        target_1 = entry_price + (2 * initial_risk)
    position = {
        "entry_timestamp": pd.Timestamp(row.timestamp),
        "entry_price": entry_price,
        "entry_raw_price": raw_entry_price,
        "entry_notional": entry_notional,
        "entry_fee": entry_fee,
        "entry_total_cost": entry_notional + entry_fee,
        "fee_rate": fee_rate,
        "slippage_rate": slippage_rate,
        "initial_size": position_size,
        "remaining_size": position_size,
        "cash_before_entry": cash,
        "exit_fees": 0.0,
        "net_exit_proceeds": 0.0,
        "stop_loss": stop_loss,
        "active_stop": stop_loss,
        "initial_risk": float(initial_risk),
        "target_1": target_1,
        "highest_close": raw_entry_price,
        "highest_high": float(row.high),
        "partial_exits": [],
        "closed_fraction": 0.0,
        "levels_hit": set(),
        "template": template,
    }
    return position, cash - position["entry_total_cost"]


def _close_position(
    position: dict[str, Any],
    timestamp: pd.Timestamp,
    price: float,
    exit_reason: str,
    cash: float,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[dict[str, Any], float, float]:
    adjusted_exit_price = price * (1 - slippage_rate)
    gross_proceeds = position["remaining_size"] * adjusted_exit_price
    exit_fee = gross_proceeds * fee_rate
    net_proceeds = gross_proceeds - exit_fee
    position["exit_fees"] += exit_fee
    position["net_exit_proceeds"] += net_proceeds
    cash += net_proceeds
    pnl = position["net_exit_proceeds"] - position["entry_total_cost"]
    return_pct = (pnl / position["entry_total_cost"]) * 100 if position["entry_total_cost"] else 0.0
    weighted_exit_price = _weighted_exit_price(position, adjusted_exit_price)
    r_multiple = (
        (weighted_exit_price - position["entry_price"]) / position["initial_risk"]
        if position["initial_risk"] > 0
        else 0.0
    )
    holding_hours = (timestamp - position["entry_timestamp"]).total_seconds() / 3600
    template = position["template"]
    trade = {
        "entry_timestamp": position["entry_timestamp"].isoformat(),
        "exit_timestamp": timestamp.isoformat(),
        "entry_price": round(position["entry_price"], 8),
        "exit_price": round(adjusted_exit_price, 8),
        "position_size": position["initial_size"],
        "entry_fee": round(position["entry_fee"], 8),
        "exit_fee": round(position["exit_fees"], 8),
        "pnl": round(pnl, 8),
        "return_pct": round(return_pct, 8),
        "exit_reason": exit_reason,
        "open_reason": template.get("open_reason", "baseline_entry_replayed"),
        "close_reason": exit_reason,
        "entry_decision": template.get("entry_decision", ""),
        "exit_decision": exit_reason,
        "entry_alignment": template.get("entry_alignment", ""),
        "exit_alignment": "",
        "entry_rr_ratio": template.get("entry_rr_ratio", 0.0),
        "entry_volume_ratio": template.get("entry_volume_ratio", 0.0),
        "entry_market_regime": template.get("entry_market_regime", ""),
        "entry_daily_setup": template.get("entry_daily_setup", ""),
        "entry_4h_price": template.get("entry_4h_price", 0.0),
        "entry_4h_ema20": template.get("entry_4h_ema20", 0.0),
        "entry_stop_loss": position["stop_loss"],
        "entry_target_1": position["target_1"],
        "initial_risk": position["initial_risk"],
        "r_multiple": round(r_multiple, 4),
        "max_price": round(position["highest_close"], 8),
        "trailing_stop": position.get("trailing_stop"),
        "partial_exits": position["partial_exits"],
        "runner_return_pct": round(((adjusted_exit_price / position["entry_price"]) - 1) * 100, 4)
        if position["partial_exits"]
        else None,
        "runner_holding_hours": round(holding_hours, 2) if position["partial_exits"] else None,
        "rejected_entry_reasons": [],
    }
    return trade, cash, pnl


def _update_position_progress(position: dict[str, Any], row: Any) -> None:
    position["highest_close"] = max(float(position["highest_close"]), float(row.close))
    position["highest_high"] = max(float(position["highest_high"]), float(row.high))


def _model_exit_reason(
    model: ExitModel,
    position: dict[str, Any],
    row: Any,
    row_number: int,
    frame: pd.DataFrame,
    four_hour_frame: pd.DataFrame | None,
    timestamp: pd.Timestamp,
    minimum_hold_hours: float,
) -> str | None:
    close = float(row.close)
    active_stop = _optional_float(position.get("active_stop"))
    if active_stop is not None and close <= active_stop:
        return "STOP_LOSS"

    holding_hours = (timestamp - position["entry_timestamp"]).total_seconds() / 3600
    if holding_hours < minimum_hold_hours:
        return None

    model_type = model.model_type
    if model_type == "ema20_trend_rider":
        if close < float(row.ema_20):
            return "EMA20_BREAK"
        if _existing_momentum_exit(position, row, four_hour_frame, timestamp):
            return "MOMENTUM_EXIT"
    elif model_type == "ema_cross":
        if _crossed_below(frame, row_number, "ema_20", "ema_50"):
            return "EMA20_EMA50_CROSS"
    elif model_type == "atr_trailing":
        trail = _atr_trailing_stop(position, row, float(model.parameters["atr_multiplier"]))
        position["trailing_stop"] = trail
        if trail is not None and close <= trail:
            return "ATR_TRAILING_STOP"
    elif model_type == "chandelier":
        trail = _chandelier_stop(frame, row_number, int(model.parameters["lookback"]), float(model.parameters["atr_multiplier"]))
        position["trailing_stop"] = trail
        if trail is not None and close <= trail:
            return "CHANDELIER_EXIT"
    elif model_type in {"partial_profit_trend_ride", "multi_target"}:
        if close < float(row.ema_20):
            return "EMA20_BREAK"
    elif model_type == "trend_strength":
        adx = _optional_float(getattr(row, "adx_14", None))
        if adx is not None and (adx <= float(model.parameters["adx_threshold"]) or float(row.ema_20) <= float(row.ema_50)):
            return "TREND_STRENGTH_EXIT"
    elif model_type == "volatility_adaptive":
        trail = _volatility_adaptive_stop(position, row, model.parameters)
        position["trailing_stop"] = trail
        if trail is not None and close <= trail:
            return "VOLATILITY_ADAPTIVE_EXIT"
    elif model_type == "hybrid":
        trail = _atr_trailing_stop(position, row, float(model.parameters["atr_multiplier"]))
        position["trailing_stop"] = trail
        if close < float(row.ema_20):
            return "EMA20_BREAK"
        if trail is not None and close <= trail:
            return "ATR_TRAILING_STOP"
    return None


def _process_partial_exits(model: ExitModel, position: dict[str, Any], row: Any) -> float:
    close = float(row.close)
    cash_delta = 0.0
    if model.model_type in {"partial_profit_trend_ride", "hybrid"}:
        target = _optional_float(position.get("target_1"))
        if target is not None and close >= target and "target_1" not in position["levels_hit"]:
            cash_delta += _sell_position_fraction(position, float(model.parameters["tp_fraction"]), close, pd.Timestamp(row.timestamp), "TP_CURRENT")
            position["levels_hit"].add("target_1")
            position["active_stop"] = max(
                _optional_float(position.get("active_stop")) or position["entry_price"],
                position["entry_price"],
            )
    elif model.model_type == "multi_target":
        for target in model.parameters["targets"]:
            key = f"{target['r']}R"
            level_price = position["entry_price"] + (float(target["r"]) * position["initial_risk"])
            if position["initial_risk"] > 0 and close >= level_price and key not in position["levels_hit"]:
                cash_delta += _sell_position_fraction(position, float(target["fraction"]), close, pd.Timestamp(row.timestamp), f"TP_{key}")
                position["levels_hit"].add(key)
                position["active_stop"] = max(
                    _optional_float(position.get("active_stop")) or position["entry_price"],
                    position["entry_price"],
                )
    return cash_delta


def _sell_position_fraction(
    position: dict[str, Any],
    fraction: float,
    price: float,
    timestamp: pd.Timestamp,
    reason: str,
    fee_rate: float | None = None,
    slippage_rate: float | None = None,
) -> float:
    fee_rate = float(position.get("fee_rate", 0.001) if fee_rate is None else fee_rate)
    slippage_rate = float(position.get("slippage_rate", 0.0005) if slippage_rate is None else slippage_rate)
    remaining_fraction = position["remaining_size"] / position["initial_size"] if position["initial_size"] else 0.0
    fraction = min(max(0.0, fraction), remaining_fraction)
    if fraction <= 0:
        return 0.0
    shares = position["initial_size"] * fraction
    adjusted_price = price * (1 - slippage_rate)
    gross_proceeds = shares * adjusted_price
    exit_fee = gross_proceeds * fee_rate
    net_proceeds = gross_proceeds - exit_fee
    position["remaining_size"] -= shares
    position["exit_fees"] += exit_fee
    position["net_exit_proceeds"] += net_proceeds
    position["closed_fraction"] += fraction
    position["partial_exits"].append(
        {
            "timestamp": timestamp.isoformat(),
            "price": round(adjusted_price, 8),
            "position_fraction": round(fraction, 4),
            "reason": reason,
        }
    )
    return net_proceeds


def _snapshot(timestamp: pd.Timestamp, price: float, cash: float, realized_pnl: float, position: dict[str, Any] | None) -> dict[str, Any]:
    position_size = float(position["remaining_size"]) if position is not None else 0.0
    position_value = position_size * price
    entry_cost = float(position["entry_total_cost"]) if position is not None else 0.0
    realized_exits = float(position["net_exit_proceeds"]) if position is not None else 0.0
    unrealized_pnl = position_value + realized_exits - entry_cost if position is not None else 0.0
    return {
        "timestamp": timestamp.isoformat(),
        "price": float(price),
        "cash": float(cash),
        "position_size": position_size,
        "current_equity": float(cash + position_value),
        "realized_pnl": float(realized_pnl),
        "unrealized_pnl": float(unrealized_pnl),
    }


def _existing_momentum_exit(
    position: dict[str, Any],
    row: Any,
    four_hour_frame: pd.DataFrame | None,
    timestamp: pd.Timestamp,
) -> bool:
    if float(row.close) <= float(position["entry_price"]):
        return False
    if four_hour_frame is None or four_hour_frame.empty:
        return False
    latest = four_hour_frame[four_hour_frame["timestamp"] <= timestamp]
    if latest.empty:
        return False
    four_hour = latest.iloc[-1]
    return (
        float(four_hour["macd"]) < float(four_hour["macd_signal"])
        and float(four_hour["close"]) < float(four_hour["ema_20"])
    )


def _atr_trailing_stop(position: dict[str, Any], row: Any, multiplier: float) -> float | None:
    atr = _optional_float(getattr(row, "atr_14", None))
    if atr is None or atr <= 0:
        return None
    return float(position["highest_close"]) - (atr * multiplier)


def _volatility_adaptive_stop(position: dict[str, Any], row: Any, parameters: dict[str, Any]) -> float | None:
    percentile = _optional_float(getattr(row, "atr_percentile_100", None))
    if percentile is None:
        multiplier = float(parameters["mid_percentile_multiplier"])
    elif percentile >= 0.70:
        multiplier = float(parameters["high_percentile_multiplier"])
    elif percentile <= 0.30:
        multiplier = float(parameters["low_percentile_multiplier"])
    else:
        multiplier = float(parameters["mid_percentile_multiplier"])
    return _atr_trailing_stop(position, row, multiplier)


def _chandelier_stop(frame: pd.DataFrame, row_number: int, lookback: int, multiplier: float) -> float | None:
    row = frame.iloc[row_number]
    atr = _optional_float(row.get("atr_14"))
    if atr is None or atr <= 0:
        return None
    start = max(0, row_number - lookback + 1)
    highest_high = float(frame.iloc[start : row_number + 1]["high"].max())
    return highest_high - (atr * multiplier)


def _crossed_below(frame: pd.DataFrame, row_number: int, fast: str, slow: str) -> bool:
    if row_number <= 0:
        return False
    previous = frame.iloc[row_number - 1]
    current = frame.iloc[row_number]
    return float(previous[fast]) >= float(previous[slow]) and float(current[fast]) < float(current[slow])


def _weighted_exit_price(position: dict[str, Any], final_exit_price: float) -> float:
    weighted = 0.0
    closed = 0.0
    for partial in position["partial_exits"]:
        fraction = float(partial["position_fraction"])
        weighted += fraction * float(partial["price"])
        closed += fraction
    runner_fraction = max(0.0, 1.0 - closed)
    return weighted + (runner_fraction * final_exit_price)


def _target_assessment(row: dict[str, Any]) -> dict[str, bool]:
    capture = _optional_float(row.get("profit_capture_ratio"))
    sharpe = _optional_float(row.get("sharpe_ratio"))
    profit_factor = _optional_float(row.get("profit_factor"))
    drawdown = _optional_float(row.get("max_drawdown_pct"))
    assessment = {
        "profit_capture_target_met": capture is not None and capture > EXIT_TARGETS["profit_capture_ratio"],
        "sharpe_target_met": sharpe is not None and sharpe >= EXIT_TARGETS["sharpe_ratio"],
        "profit_factor_target_met": profit_factor is not None and profit_factor > EXIT_TARGETS["profit_factor"],
        "drawdown_target_met": drawdown is not None and drawdown < EXIT_TARGETS["max_drawdown_pct"],
    }
    assessment["all_targets_met"] = all(assessment.values())
    return assessment


def _recommended_exit(rows: dict[str, dict[str, Any]], ranking: list[dict[str, Any]]) -> dict[str, Any]:
    passing = [row["model"] for row in ranking if rows[row["model"]]["target_assessment"]["all_targets_met"]]
    if passing:
        model = passing[0]
        return {
            "model": model,
            "reason": "This model met all Phase 1.17 profit capture, Sharpe, profit factor, and drawdown targets.",
            "metrics": _model_summary(rows, model),
        }
    baseline = rows.get("baseline", {})
    baseline_capture = _optional_float(baseline.get("profit_capture_ratio")) or 0.0
    candidates = [
        row["model"]
        for row in ranking
        if row["model"] != "baseline"
        and (_optional_float(rows[row["model"]].get("profit_capture_ratio")) or 0.0) > baseline_capture
        and rows[row["model"]]["target_assessment"]["drawdown_target_met"]
        and rows[row["model"]]["target_assessment"]["sharpe_target_met"]
        and rows[row["model"]]["target_assessment"]["profit_factor_target_met"]
    ]
    if candidates:
        model = candidates[0]
        return {
            "model": model,
            "reason": "This model improved profit capture versus baseline while preserving the non-capture risk targets.",
            "metrics": _model_summary(rows, model),
        }
    return {
        "model": "baseline",
        "reason": "No researched exit model beat the current production exit while meeting the Phase 1.17 risk constraints.",
        "metrics": _model_summary(rows, "baseline"),
    }


def _comparison_vs_baseline(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for metric in ("profit_capture_ratio", "sharpe_ratio", "total_return_pct", "max_drawdown_pct", "profit_factor"):
        base = _optional_float(baseline.get(metric))
        current = _optional_float(candidate.get(metric))
        comparison[f"{metric}_baseline"] = base
        comparison[f"{metric}_candidate"] = current
        comparison[f"{metric}_delta"] = round(current - base, 4) if base is not None and current is not None else None
    return comparison


def _trend_capture_analysis(result: BenchmarkResult, price_history: pd.DataFrame, limit: int = 20) -> dict[str, Any]:
    history = price_history.copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
    rows = []
    for trade in result.trades:
        entry_timestamp = pd.Timestamp(trade["entry_timestamp"])
        exit_timestamp = pd.Timestamp(trade["exit_timestamp"])
        entry_price = float(trade["entry_price"])
        available_high = _max_high_between(history, entry_timestamp, exit_timestamp + pd.Timedelta(days=90))
        available_profit = max(0.0, ((available_high / entry_price) - 1) * 100) if available_high is not None else 0.0
        captured_profit = _captured_gain_pct(trade, entry_price, float(trade["exit_price"]))
        rows.append(
            {
                "entry_timestamp": entry_timestamp.isoformat(),
                "exit_timestamp": exit_timestamp.isoformat(),
                "exit_reason": trade.get("exit_reason"),
                "captured_profit_pct": round(captured_profit, 2),
                "available_profit_pct": round(available_profit, 2),
                "capture_ratio": round(captured_profit / available_profit, 4) if available_profit > 0 else None,
            }
        )
    top = sorted(rows, key=lambda row: row["available_profit_pct"], reverse=True)[:limit]
    ratios = [float(row["capture_ratio"]) for row in top if row["capture_ratio"] is not None]
    return {
        "sample_size": len(top),
        "average_capture_ratio": round(float(np.mean(ratios)), 4) if ratios else None,
        "top_20_winning_trends": top,
    }


def _captured_gain_pct(trade: dict[str, Any], entry_price: float, exit_price: float) -> float:
    partial_exits = trade.get("partial_exits") or []
    if not partial_exits:
        return max(0.0, ((exit_price / entry_price) - 1) * 100)
    captured = 0.0
    closed_fraction = 0.0
    for partial in partial_exits:
        fraction = max(0.0, float(partial.get("position_fraction", 0.0)))
        partial_price = float(partial.get("price", exit_price))
        captured += fraction * max(0.0, ((partial_price / entry_price) - 1) * 100)
        closed_fraction += fraction
    runner_fraction = max(0.0, 1.0 - closed_fraction)
    return captured + runner_fraction * max(0.0, ((exit_price / entry_price) - 1) * 100)


def _max_high_between(price_history: pd.DataFrame, start_timestamp: pd.Timestamp, end_timestamp: pd.Timestamp) -> float | None:
    window = price_history[
        (price_history["timestamp"] >= start_timestamp)
        & (price_history["timestamp"] <= end_timestamp)
    ]
    if window.empty:
        return None
    return float(window["high"].max())


def _model_summary(rows: dict[str, dict[str, Any]], model_name: str | None) -> dict[str, Any] | None:
    if model_name is None or model_name not in rows:
        return None
    row = rows[model_name]
    return {
        "model": model_name,
        "total_return_pct": row.get("total_return_pct"),
        "cagr": row.get("cagr"),
        "sharpe_ratio": row.get("sharpe_ratio"),
        "max_drawdown_pct": row.get("max_drawdown_pct"),
        "profit_factor": row.get("profit_factor"),
        "win_rate": row.get("win_rate"),
        "total_trades": row.get("total_trades"),
        "profit_capture_ratio": row.get("profit_capture_ratio"),
        "average_holding_days": row.get("average_holding_days"),
        "target_assessment": row.get("target_assessment"),
    }


def _best_by(rows: dict[str, dict[str, Any]], metric: str) -> str | None:
    if not rows:
        return None
    return max(rows, key=lambda name: _metric_value(rows[name].get(metric)))


def _best_hybrid(rows: dict[str, dict[str, Any]]) -> str | None:
    hybrids = [name for name, row in rows.items() if row.get("model", {}).get("category") == "hybrid"]
    if not hybrids:
        return None
    return max(
        hybrids,
        key=lambda name: (
            _metric_value(rows[name].get("profit_capture_ratio")),
            _metric_value(rows[name].get("sharpe_ratio")),
            _metric_value(rows[name].get("total_return_pct")),
        ),
    )


def _rank_metric(rows: dict[str, dict[str, Any]], metric: str, reverse: bool) -> list[str]:
    return sorted(rows, key=lambda name: _metric_value(rows[name].get(metric), missing=float("-inf")), reverse=reverse)


def _count_exit_reasons(trades: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trade in trades:
        reason = str(trade.get("exit_reason", "UNKNOWN"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _metric_value(value: object, missing: float = float("-inf")) -> float:
    number = _optional_float(value)
    if number is None:
        return missing
    if math.isinf(number):
        return 1e12 if number > 0 else -1e12
    return number


def _filter_date_range(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    filtered = frame.copy()
    filtered["timestamp"] = pd.to_datetime(filtered["timestamp"], utc=True)
    filtered = filtered[filtered["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end != "latest":
        filtered = filtered[filtered["timestamp"] <= pd.Timestamp(end, tz="UTC")]
    return filtered.sort_values("timestamp").reset_index(drop=True)


def _timeframe_delta(timeframe: str) -> pd.Timedelta:
    if timeframe.endswith("h"):
        return pd.Timedelta(hours=int(timeframe[:-1]))
    if timeframe.endswith("d"):
        return pd.Timedelta(days=int(timeframe[:-1]))
    if timeframe.endswith("w"):
        return pd.Timedelta(weeks=int(timeframe[:-1]))
    return pd.Timedelta(0)


def _ordered_timeframes(primary_timeframe: str, timeframes: tuple[str, ...]) -> tuple[str, ...]:
    ordered = [primary_timeframe]
    for timeframe in timeframes:
        if timeframe not in ordered:
            ordered.append(timeframe)
    return tuple(ordered)


def _emit(progress_callback: ProgressCallback | None, event: dict[str, Any]) -> None:
    if progress_callback is not None:
        progress_callback(event)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
