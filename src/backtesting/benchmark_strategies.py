"""Benchmark strategies for comparing Phase 1.5 backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from backtesting.performance_metrics import calculate_performance_metrics
from trading_agent.indicators import add_indicators


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    metrics: dict[str, Any]
    trades: list[dict[str, Any]]
    equity_curve: pd.DataFrame


def run_buy_and_hold_benchmark(
    symbol: str,
    ohlcv: pd.DataFrame,
    initial_capital: float,
) -> BenchmarkResult:
    """Invest fully at the first candle and hold through the final candle."""

    frame = _prepare_ohlcv(ohlcv)
    start_price = float(frame.iloc[0]["close"])
    equity_curve = pd.DataFrame(
        {
            "timestamp": frame["timestamp"],
            "price": frame["close"].astype(float),
            "current_equity": initial_capital * frame["close"].astype(float) / start_price,
        }
    )
    metrics = calculate_performance_metrics(
        equity_curve=equity_curve,
        trades=[],
        initial_capital=initial_capital,
        start_price=start_price,
        end_price=float(frame.iloc[-1]["close"]),
    )
    metrics["profit_factor"] = None
    return _benchmark_result("buy_and_hold", symbol, initial_capital, metrics, [], equity_curve)


def run_daily_ema200_benchmark(
    symbol: str,
    daily_ohlcv: pd.DataFrame,
    initial_capital: float,
    start: str | None = None,
    end: str | None = None,
) -> BenchmarkResult:
    """Long-only daily EMA200 filter: hold BTC when daily close is above EMA200."""

    frame = add_indicators(_prepare_ohlcv(daily_ohlcv))
    if start is not None:
        frame = frame[frame["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None and end != "latest":
        frame = frame[frame["timestamp"] <= pd.Timestamp(end, tz="UTC")]
    frame = frame.dropna(subset=["ema_200"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("daily_ohlcv must include enough rows to calculate EMA200.")

    cash = float(initial_capital)
    position_size = 0.0
    entry_price: float | None = None
    entry_timestamp: str | None = None
    equity_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    start_price = float(frame.iloc[0]["close"])

    for row in frame.itertuples(index=False):
        timestamp = pd.Timestamp(row.timestamp)
        close = float(row.close)
        ema200 = float(row.ema_200)

        if position_size <= 0 and close > ema200:
            position_size = cash / close
            cash = 0.0
            entry_price = close
            entry_timestamp = timestamp.isoformat()
        elif position_size > 0 and close <= ema200:
            exit_value = position_size * close
            pnl = exit_value - float(initial_capital if entry_price is None else position_size * entry_price)
            cash = exit_value
            trades.append(
                {
                    "entry_timestamp": entry_timestamp,
                    "exit_timestamp": timestamp.isoformat(),
                    "entry_price": entry_price,
                    "exit_price": close,
                    "pnl": pnl,
                    "return_pct": ((close / float(entry_price)) - 1) * 100 if entry_price else 0.0,
                    "r_multiple": 0.0,
                    "exit_reason": "EMA200_EXIT",
                }
            )
            position_size = 0.0
            entry_price = None
            entry_timestamp = None

        equity_rows.append(
            {
                "timestamp": timestamp.isoformat(),
                "price": close,
                "current_equity": cash + (position_size * close),
            }
        )

    equity_curve = pd.DataFrame(equity_rows)
    metrics = calculate_performance_metrics(
        equity_curve=equity_curve,
        trades=trades,
        initial_capital=initial_capital,
        start_price=start_price,
        end_price=float(frame.iloc[-1]["close"]),
    )
    if not trades:
        metrics["profit_factor"] = None
    return _benchmark_result("daily_ema200", symbol, initial_capital, metrics, trades, equity_curve)


def _prepare_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    if ohlcv.empty:
        raise ValueError("ohlcv must not be empty.")
    frame = ohlcv.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values("timestamp").reset_index(drop=True)


def _benchmark_result(
    name: str,
    symbol: str,
    initial_capital: float,
    metrics: dict[str, Any],
    trades: list[dict[str, Any]],
    equity_curve: pd.DataFrame,
) -> BenchmarkResult:
    return BenchmarkResult(
        name=name,
        symbol=symbol,
        start_date=pd.Timestamp(equity_curve.iloc[0]["timestamp"]).date().isoformat(),
        end_date=pd.Timestamp(equity_curve.iloc[-1]["timestamp"]).date().isoformat(),
        initial_capital=initial_capital,
        final_equity=float(metrics["final_equity"]),
        metrics=metrics,
        trades=trades,
        equity_curve=equity_curve,
    )
