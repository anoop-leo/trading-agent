"""Performance metrics for Phase 1.5 backtests."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def calculate_performance_metrics(
    equity_curve: pd.DataFrame,
    trades: list[dict[str, Any]],
    initial_capital: float,
    start_price: float,
    end_price: float,
) -> dict[str, Any]:
    """Calculate deterministic backtest performance metrics."""

    if equity_curve.empty:
        raise ValueError("equity_curve must not be empty.")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be greater than zero.")
    if start_price <= 0:
        raise ValueError("start_price must be greater than zero.")

    curve = equity_curve.copy()
    curve["timestamp"] = pd.to_datetime(curve["timestamp"], utc=True)
    curve = curve.sort_values("timestamp").reset_index(drop=True)
    final_equity = float(curve.iloc[-1]["current_equity"])
    total_return_pct = _pct(final_equity / initial_capital - 1)
    cagr = _calculate_cagr(initial_capital, final_equity, curve.iloc[0]["timestamp"], curve.iloc[-1]["timestamp"])
    trade_metrics = _trade_metrics(trades)

    return {
        "final_equity": round(final_equity, 2),
        "total_return_pct": total_return_pct,
        "cagr": cagr,
        "win_rate": trade_metrics["win_rate"],
        "loss_rate": trade_metrics["loss_rate"],
        "total_trades": trade_metrics["total_trades"],
        "winning_trades": trade_metrics["winning_trades"],
        "losing_trades": trade_metrics["losing_trades"],
        "average_win_pct": trade_metrics["average_win_pct"],
        "average_loss_pct": trade_metrics["average_loss_pct"],
        "profit_factor": trade_metrics["profit_factor"],
        "max_drawdown_pct": _calculate_max_drawdown(curve["current_equity"]),
        "sharpe_ratio": _calculate_sharpe_ratio(curve),
        "expectancy": trade_metrics["expectancy"],
        "average_holding_hours": trade_metrics["average_holding_hours"],
        "median_holding_hours": trade_metrics["median_holding_hours"],
        "average_r_multiple": trade_metrics["average_r_multiple"],
        "average_r_multiple_by_exit_reason": trade_metrics["average_r_multiple_by_exit_reason"],
        "best_trade": trade_metrics["best_trade"],
        "worst_trade": trade_metrics["worst_trade"],
        "buy_and_hold_return_pct": _pct(end_price / start_price - 1),
    }


def _trade_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    total_trades = len(trades)
    returns = [float(trade["return_pct"]) for trade in trades]
    wins = [value for value in returns if value > 0]
    losses = [value for value in returns if value < 0]
    gross_profit = sum(float(trade["pnl"]) for trade in trades if float(trade["pnl"]) > 0)
    gross_loss = sum(float(trade["pnl"]) for trade in trades if float(trade["pnl"]) < 0)
    winning_trades = len(wins)
    losing_trades = len(losses)
    win_rate_fraction = winning_trades / total_trades if total_trades else 0.0
    loss_rate_fraction = losing_trades / total_trades if total_trades else 0.0
    average_win_pct = sum(wins) / len(wins) if wins else 0.0
    average_loss_pct = sum(losses) / len(losses) if losses else 0.0
    if math.isclose(gross_loss, 0.0):
        profit_factor = float("inf") if gross_profit > 0 else 0.0
    else:
        profit_factor = gross_profit / abs(gross_loss)
    expectancy = (win_rate_fraction * average_win_pct) - (loss_rate_fraction * abs(average_loss_pct))

    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": round(win_rate_fraction * 100, 2),
        "loss_rate": round(loss_rate_fraction * 100, 2),
        "average_win_pct": round(average_win_pct, 2),
        "average_loss_pct": round(average_loss_pct, 2),
        "profit_factor": round(profit_factor, 2) if math.isfinite(profit_factor) else profit_factor,
        "expectancy": round(expectancy, 2),
        **_holding_period_metrics(trades),
        **_r_multiple_metrics(trades),
    }


def _holding_period_metrics(trades: list[dict[str, Any]]) -> dict[str, float]:
    holding_hours: list[float] = []
    for trade in trades:
        entry_timestamp = trade.get("entry_timestamp")
        exit_timestamp = trade.get("exit_timestamp")
        if entry_timestamp is None or exit_timestamp is None:
            continue
        hours = (
            pd.Timestamp(exit_timestamp) - pd.Timestamp(entry_timestamp)
        ).total_seconds() / 3600
        holding_hours.append(hours)
    if not holding_hours:
        return {"average_holding_hours": 0.0, "median_holding_hours": 0.0}
    return {
        "average_holding_hours": round(float(np.mean(holding_hours)), 2),
        "median_holding_hours": round(float(np.median(holding_hours)), 2),
    }


def _r_multiple_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {
            "average_r_multiple": 0.0,
            "average_r_multiple_by_exit_reason": {},
            "best_trade": None,
            "worst_trade": None,
        }
    r_multiples = [float(trade.get("r_multiple", 0.0)) for trade in trades]
    best_trade = max(trades, key=lambda trade: float(trade.get("r_multiple", 0.0)))
    worst_trade = min(trades, key=lambda trade: float(trade.get("r_multiple", 0.0)))
    return {
        "average_r_multiple": round(float(np.mean(r_multiples)), 2),
        "average_r_multiple_by_exit_reason": _average_r_multiple_by_exit_reason(trades),
        "best_trade": _summarize_trade(best_trade),
        "worst_trade": _summarize_trade(worst_trade),
    }


def _average_r_multiple_by_exit_reason(trades: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for trade in trades:
        reason = str(trade.get("exit_reason", "UNKNOWN"))
        grouped.setdefault(reason, []).append(float(trade.get("r_multiple", 0.0)))
    return {
        reason: round(float(np.mean(values)), 2)
        for reason, values in sorted(grouped.items())
    }


def _summarize_trade(trade: dict[str, Any]) -> dict[str, Any]:
    return {
        "entry_timestamp": trade.get("entry_timestamp"),
        "exit_timestamp": trade.get("exit_timestamp"),
        "exit_reason": trade.get("exit_reason"),
        "pnl": round(float(trade.get("pnl", 0.0)), 2),
        "return_pct": round(float(trade.get("return_pct", 0.0)), 2),
        "r_multiple": round(float(trade.get("r_multiple", 0.0)), 2),
    }


def _calculate_cagr(
    initial_capital: float,
    final_equity: float,
    start_timestamp: pd.Timestamp,
    end_timestamp: pd.Timestamp,
) -> float:
    days = max((end_timestamp - start_timestamp).total_seconds() / 86400, 0.0)
    years = days / 365.25
    if years <= 0:
        return 0.0
    return _pct((final_equity / initial_capital) ** (1 / years) - 1)


def _calculate_max_drawdown(equity: pd.Series) -> float:
    running_peak = equity.cummax()
    drawdown = (equity / running_peak) - 1
    return round(abs(float(drawdown.min())) * 100, 2)


def _calculate_sharpe_ratio(curve: pd.DataFrame) -> float:
    daily_equity = curve.set_index("timestamp")["current_equity"].resample("1D").last().dropna()
    returns = daily_equity.pct_change().dropna()
    if len(returns) < 2 or math.isclose(float(returns.std()), 0.0):
        return 0.0
    sharpe = (float(returns.mean()) / float(returns.std())) * math.sqrt(365)
    if np.isnan(sharpe):
        return 0.0
    return round(sharpe, 2)


def _pct(value: float) -> float:
    return round(value * 100, 2)
