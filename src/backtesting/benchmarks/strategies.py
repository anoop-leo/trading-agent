"""Deterministic benchmark strategies for Phase 1.6 research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

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


class BenchmarkStrategy(Protocol):
    name: str

    def run(
        self,
        *,
        config: Any,
        frames: dict[str, pd.DataFrame],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> BenchmarkResult:
        """Run a benchmark strategy over cached OHLCV frames."""


class BuyAndHoldStrategy:
    name = "buy_and_hold"

    def run(
        self,
        *,
        config: Any,
        frames: dict[str, pd.DataFrame],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> BenchmarkResult:
        _emit(progress_callback, self.name)
        frame = _filter_date_range(_prepare_ohlcv(frames[config.primary_timeframe]), config.start, config.end)
        if frame.empty:
            raise ValueError("primary timeframe data is empty for BuyAndHoldStrategy.")
        start_price = float(frame.iloc[0]["close"])
        equity_curve = pd.DataFrame(
            {
                "timestamp": frame["timestamp"],
                "price": frame["close"].astype(float),
                "current_equity": config.initial_capital * frame["close"].astype(float) / start_price,
            }
        )
        metrics = calculate_performance_metrics(
            equity_curve=equity_curve,
            trades=[],
            initial_capital=config.initial_capital,
            start_price=start_price,
            end_price=float(frame.iloc[-1]["close"]),
        )
        metrics["profit_factor"] = None
        return _result(self.name, config.symbol, config.initial_capital, metrics, [], equity_curve)


class EMA200Strategy:
    name = "ema200"

    def run(
        self,
        *,
        config: Any,
        frames: dict[str, pd.DataFrame],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> BenchmarkResult:
        _emit(progress_callback, self.name)
        frame = _daily_indicator_frame(frames["1d"], config.start, config.end)
        return _run_long_flat_strategy(
            name=self.name,
            symbol=config.symbol,
            frame=frame,
            initial_capital=config.initial_capital,
            should_hold=lambda row: float(row.close) > float(row.ema_200),
        )


class GoldenCrossStrategy:
    name = "golden_cross"

    def run(
        self,
        *,
        config: Any,
        frames: dict[str, pd.DataFrame],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> BenchmarkResult:
        _emit(progress_callback, self.name)
        frame = _daily_indicator_frame(frames["1d"], config.start, config.end)
        return _run_long_flat_strategy(
            name=self.name,
            symbol=config.symbol,
            frame=frame,
            initial_capital=config.initial_capital,
            should_hold=lambda row: float(row.ema_50) > float(row.ema_200),
        )


class RSITrendStrategy:
    name = "rsi_trend"

    def run(
        self,
        *,
        config: Any,
        frames: dict[str, pd.DataFrame],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> BenchmarkResult:
        _emit(progress_callback, self.name)
        frame = _daily_indicator_frame(frames["1d"], config.start, config.end)
        in_position = False

        def should_hold(row: Any) -> bool:
            nonlocal in_position
            close = float(row.close)
            ema200 = float(row.ema_200)
            rsi = float(row.rsi_14)
            if in_position:
                in_position = close > ema200 and rsi >= 45
            else:
                in_position = close > ema200 and rsi >= 50
            return in_position

        return _run_long_flat_strategy(
            name=self.name,
            symbol=config.symbol,
            frame=frame,
            initial_capital=config.initial_capital,
            should_hold=should_hold,
        )


class AgentAggressiveStrategy:
    name = "agent_aggressive"

    def run(
        self,
        *,
        config: Any,
        frames: dict[str, pd.DataFrame],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> BenchmarkResult:
        _emit(progress_callback, self.name)
        from dataclasses import replace

        from backtesting.backtest_engine import run_backtest

        result = run_backtest(
            replace(config, profile="aggressive", strategy_profile_override=None),
            cached_data=frames,
            progress_callback=progress_callback,
        )
        return BenchmarkResult(
            name=self.name,
            symbol=result.symbol,
            start_date=result.start_date,
            end_date=result.end_date,
            initial_capital=result.initial_capital,
            final_equity=result.final_equity,
            metrics=result.metrics,
            trades=result.trades,
            equity_curve=result.equity_curve,
        )


class AgentStopStrategy:
    def __init__(self, name: str, stop_type: str) -> None:
        self.name = name
        self.stop_type = stop_type

    def run(
        self,
        *,
        config: Any,
        frames: dict[str, pd.DataFrame],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> BenchmarkResult:
        _emit(progress_callback, self.name)
        from dataclasses import replace

        from backtesting.backtest_engine import run_backtest

        result = run_backtest(
            replace(
                config,
                profile="aggressive",
                strategy_profile_override=None,
                stop_type=self.stop_type,
                collect_stop_candidates=self.stop_type == "fixed",
            ),
            cached_data=frames,
            progress_callback=progress_callback,
        )
        return BenchmarkResult(
            name=self.name,
            symbol=result.symbol,
            start_date=result.start_date,
            end_date=result.end_date,
            initial_capital=result.initial_capital,
            final_equity=result.final_equity,
            metrics=result.metrics,
            trades=result.trades,
            equity_curve=result.equity_curve,
        )


class BullModeAgentStrategy:
    name = "bull_mode_agent"

    def run(
        self,
        *,
        config: Any,
        frames: dict[str, pd.DataFrame],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> BenchmarkResult:
        _emit(progress_callback, self.name)
        from dataclasses import replace

        from backtesting.backtest_engine import run_backtest
        from backtesting.profiles import get_strategy_profile

        profile = replace(
            get_strategy_profile("aggressive"),
            name=self.name,
            enable_bull_market_mode=True,
            bull_min_rr_ratio=1.2,
            bull_min_volume_ratio=0.5,
            bull_allow_pullback_alignment=True,
        )
        result = run_backtest(
            replace(config, profile="aggressive", strategy_profile_override=profile),
            cached_data=frames,
            progress_callback=progress_callback,
        )
        return BenchmarkResult(
            name=self.name,
            symbol=result.symbol,
            start_date=result.start_date,
            end_date=result.end_date,
            initial_capital=result.initial_capital,
            final_equity=result.final_equity,
            metrics=result.metrics,
            trades=result.trades,
            equity_curve=result.equity_curve,
        )


class TrendRiderAggressiveStrategy:
    name = "trend_rider_aggressive"

    def run(
        self,
        *,
        config: Any,
        frames: dict[str, pd.DataFrame],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> BenchmarkResult:
        _emit(progress_callback, self.name)
        from dataclasses import replace

        from backtesting.backtest_engine import run_backtest

        result = run_backtest(
            replace(config, profile="aggressive", strategy_profile_override=None, use_trend_rider=True),
            cached_data=frames,
            progress_callback=progress_callback,
        )
        return BenchmarkResult(
            name=self.name,
            symbol=result.symbol,
            start_date=result.start_date,
            end_date=result.end_date,
            initial_capital=result.initial_capital,
            final_equity=result.final_equity,
            metrics=result.metrics,
            trades=result.trades,
            equity_curve=result.equity_curve,
        )


class TrendHoldingStrategy:
    name = "trend_holding"

    def run(
        self,
        *,
        config: Any,
        frames: dict[str, pd.DataFrame],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> BenchmarkResult:
        _emit(progress_callback, self.name)
        from dataclasses import replace

        from backtesting.backtest_engine import run_backtest

        result = run_backtest(
            replace(config, profile="aggressive", strategy_profile_override=None, use_trend_holding=True),
            cached_data=frames,
            progress_callback=progress_callback,
        )
        return BenchmarkResult(
            name=self.name,
            symbol=result.symbol,
            start_date=result.start_date,
            end_date=result.end_date,
            initial_capital=result.initial_capital,
            final_equity=result.final_equity,
            metrics=result.metrics,
            trades=result.trades,
            equity_curve=result.equity_curve,
        )


class RegimeGatedTrendHoldingStrategy:
    name = "regime_gated_trend_holding"

    def run(
        self,
        *,
        config: Any,
        frames: dict[str, pd.DataFrame],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> BenchmarkResult:
        _emit(progress_callback, self.name)
        from dataclasses import replace

        from backtesting.backtest_engine import run_backtest

        result = run_backtest(
            replace(
                config,
                profile="aggressive",
                strategy_profile_override=None,
                use_regime_gated_trend_holding=True,
                auxiliary_timeframes=("1w",),
            ),
            cached_data=frames,
            progress_callback=progress_callback,
        )
        return BenchmarkResult(
            name=self.name,
            symbol=result.symbol,
            start_date=result.start_date,
            end_date=result.end_date,
            initial_capital=result.initial_capital,
            final_equity=result.final_equity,
            metrics=result.metrics,
            trades=result.trades,
            equity_curve=result.equity_curve,
        )


class RegimeGatedPortfolioGovernorStrategy:
    name = "regime_gated_portfolio_governor"

    def run(
        self,
        *,
        config: Any,
        frames: dict[str, pd.DataFrame],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> BenchmarkResult:
        _emit(progress_callback, self.name)
        from dataclasses import replace

        from backtesting.backtest_engine import run_backtest

        result = run_backtest(
            replace(
                config,
                profile="aggressive",
                strategy_profile_override=None,
                use_portfolio_governor=True,
                auxiliary_timeframes=("1w",),
            ),
            cached_data=frames,
            progress_callback=progress_callback,
        )
        return BenchmarkResult(
            name=self.name,
            symbol=result.symbol,
            start_date=result.start_date,
            end_date=result.end_date,
            initial_capital=result.initial_capital,
            final_equity=result.final_equity,
            metrics=result.metrics,
            trades=result.trades,
            equity_curve=result.equity_curve,
        )


class HybridTrendRiderStrategy:
    def __init__(self, name: str = "hybrid_trend_rider") -> None:
        self.name = name

    def run(
        self,
        *,
        config: Any,
        frames: dict[str, pd.DataFrame],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> BenchmarkResult:
        _emit(progress_callback, self.name)
        from dataclasses import replace

        from backtesting.backtest_engine import run_backtest
        from backtesting.profiles import get_strategy_profile

        profile = replace(get_strategy_profile("aggressive"), name=self.name, allocation_per_trade=0.25)
        auxiliary_timeframes = ("1w",) if self.name == "hybrid_aggressive" else ()
        result = run_backtest(
            replace(
                config,
                profile="aggressive",
                strategy_profile_override=profile,
                use_hybrid_trend_rider=True,
                hybrid_runner_profile=self.name,
                auxiliary_timeframes=auxiliary_timeframes,
            ),
            cached_data=frames,
            progress_callback=progress_callback,
        )
        return BenchmarkResult(
            name=self.name,
            symbol=result.symbol,
            start_date=result.start_date,
            end_date=result.end_date,
            initial_capital=result.initial_capital,
            final_equity=result.final_equity,
            metrics=result.metrics,
            trades=result.trades,
            equity_curve=result.equity_curve,
        )


DEFAULT_BENCHMARK_STRATEGIES: tuple[BenchmarkStrategy, ...] = (
    BuyAndHoldStrategy(),
    EMA200Strategy(),
    GoldenCrossStrategy(),
    RSITrendStrategy(),
    AgentAggressiveStrategy(),
)


def _run_long_flat_strategy(
    *,
    name: str,
    symbol: str,
    frame: pd.DataFrame,
    initial_capital: float,
    should_hold: Callable[[Any], bool],
) -> BenchmarkResult:
    if frame.empty:
        raise ValueError(f"{name} data is empty.")

    cash = float(initial_capital)
    position_size = 0.0
    entry_price: float | None = None
    entry_timestamp: str | None = None
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    start_price = float(frame.iloc[0]["close"])

    for row in frame.itertuples(index=False):
        timestamp = pd.Timestamp(row.timestamp)
        close = float(row.close)
        hold_signal = should_hold(row)

        if position_size <= 0 and hold_signal:
            position_size = cash / close
            cash = 0.0
            entry_price = close
            entry_timestamp = timestamp.isoformat()
        elif position_size > 0 and not hold_signal:
            exit_value = position_size * close
            pnl = exit_value - (position_size * float(entry_price))
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
                    "exit_reason": f"{name.upper()}_EXIT",
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
    return _result(name, symbol, initial_capital, metrics, trades, equity_curve)


def _daily_indicator_frame(daily_ohlcv: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    frame = add_indicators(_prepare_ohlcv(daily_ohlcv))
    frame = _filter_date_range(frame, start, end)
    frame = frame.dropna(subset=["ema_200", "ema_50", "rsi_14"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("daily data must include enough warmup rows for benchmark indicators.")
    return frame


def _prepare_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    if ohlcv.empty:
        raise ValueError("ohlcv must not be empty.")
    frame = ohlcv.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values("timestamp").reset_index(drop=True)


def _filter_date_range(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    filtered = frame.copy()
    filtered["timestamp"] = pd.to_datetime(filtered["timestamp"], utc=True)
    filtered = filtered[filtered["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end != "latest":
        filtered = filtered[filtered["timestamp"] <= pd.Timestamp(end, tz="UTC")]
    return filtered.sort_values("timestamp").reset_index(drop=True)


def _result(
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


def _emit(progress_callback: Callable[[dict[str, Any]], None] | None, strategy: str) -> None:
    if progress_callback is not None:
        progress_callback({"phase": "benchmark", "strategy": strategy})
