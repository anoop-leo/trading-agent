"""Deterministic historical backtesting engine for Phase 1.5."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from backtesting.performance_metrics import calculate_performance_metrics
from backtesting.profiles import StrategyProfile, get_strategy_profile
from backtesting.trade_simulator import TradeSimulator
from data.equity_data_adapter import (
    download_equity_timeframe,
    normalize_yahoo_chart as normalize_yahoo_chart_payload,
    resample_ohlcv as resample_equity_ohlcv,
)
from decision.decision_engine import apply_multi_timeframe_alignment
from risk.structure_stop_engine import StructureStopEngine
from scoring.multi_timeframe_skill import analyze_multi_timeframe
from scoring.support_resistance_skill import SupportResistanceResult
from trading_agent.config import AgentConfig
from trading_agent.data import BINANCE_KLINES_PATH, DataLoadError, _load_json_response, normalize_klines
from trading_agent.indicators import add_indicators
from trading_agent.main import analyze_indicator_frame, build_timeframe_signal
from trading_agent.models import OHLCV_COLUMNS
from trading_agent.output import macd_direction
from trading_agent.scoring import calculate_volume_ratio
from risk.portfolio_risk_governor import calculate_atr_moving_average


INTERVAL_TO_MILLISECONDS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "1w": 604_800_000,
}
DEFAULT_BACKTEST_TIMEFRAMES = ("1h", "4h", "1d")
MIN_HISTORY_ROWS = 200
ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class BacktestConfig:
    symbol: str = "BTCUSDT"
    start: str = "2017-01-01"
    end: str = "latest"
    primary_timeframe: str = "1h"
    timeframes: tuple[str, ...] = DEFAULT_BACKTEST_TIMEFRAMES
    initial_capital: float = 10000.0
    fee_rate: float = 0.001
    slippage_rate: float = 0.0005
    cache_dir: Path = Path("data/cache")
    output_dir: Path = Path("outputs")
    binance_base_url: str = "https://api.binance.com"
    yahoo_base_url: str = "https://query1.finance.yahoo.com"
    request_timeout_seconds: float = 10.0
    progress_interval: int = 1000
    profile: str = "balanced"
    refresh_cache: bool = False
    strategy_profile_override: StrategyProfile | None = None
    use_trend_rider: bool = False
    use_hybrid_trend_rider: bool = False
    use_trend_holding: bool = False
    use_regime_gated_trend_holding: bool = False
    use_portfolio_governor: bool = False
    hybrid_runner_profile: str | None = None
    auxiliary_timeframes: tuple[str, ...] = ()
    close_open_position_on_end: bool = False
    stop_type: str = "fixed"
    collect_stop_candidates: bool = False


@dataclass(frozen=True)
class BacktestResult:
    symbol: str
    profile: str
    strategy_profile: dict[str, Any]
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    metrics: dict[str, Any]
    decisions: list[dict[str, Any]]
    trades: list[dict[str, Any]]
    equity_curve: pd.DataFrame


class BacktestError(RuntimeError):
    """Raised when a backtest cannot be completed."""


@dataclass(frozen=True)
class PreparedTimeframeData:
    timeframe: str
    frame: pd.DataFrame
    support_resistance: list[SupportResistanceResult]


def run_backtest(
    config: BacktestConfig,
    cached_data: dict[str, pd.DataFrame] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> BacktestResult:
    """Run a deterministic multi-timeframe backtest.

    Lookahead bias is avoided by slicing each timeframe to candles available at
    the replay timestamp. Higher-timeframe candles are included only after their
    interval has fully elapsed.
    """

    timeframes = _ordered_timeframes(config.primary_timeframe, config.timeframes)
    frames = cached_data or load_or_download_timeframes(config, timeframes)
    frames = _ensure_auxiliary_timeframes(frames, config)
    frames = {timeframe: _filter_history_until_end(frame, config.end) for timeframe, frame in frames.items()}
    prepared_frames = {timeframe: _prepare_timeframe_data(timeframe, frame) for timeframe, frame in frames.items()}
    primary_history = prepared_frames[config.primary_timeframe].frame
    primary_frame = _filter_date_range(primary_history, config.start, config.end)
    if len(primary_history) < MIN_HISTORY_ROWS:
        raise BacktestError("Not enough primary timeframe candles to run backtest.")
    if primary_frame.empty:
        raise BacktestError("No primary timeframe candles found for the requested backtest range.")
    _emit_progress(
        progress_callback,
        {
            "phase": "prepared",
            "symbol": config.symbol,
            "timeframes": list(timeframes),
            "primary_rows": len(primary_frame),
        },
    )

    strategy_profile = config.strategy_profile_override or get_strategy_profile(config.profile)
    simulator_class = _simulator_class(
        config.use_trend_rider,
        config.use_hybrid_trend_rider,
        config.use_trend_holding,
        config.use_regime_gated_trend_holding,
        config.use_portfolio_governor,
    )
    simulator = simulator_class(
        initial_capital=config.initial_capital,
        fee_rate=config.fee_rate,
        slippage_rate=config.slippage_rate,
        strategy_profile=strategy_profile,
        **({"hybrid_profile_name": config.hybrid_runner_profile} if config.use_hybrid_trend_rider else {}),
    )
    decisions: list[dict[str, Any]] = []
    first_signal_price: float | None = None
    last_decision_record: dict[str, Any] | None = None
    stop_engine = StructureStopEngine(config.stop_type, collect_candidates=config.collect_stop_candidates)

    for row_number, row in enumerate(primary_frame.itertuples(index=False), start=1):
        candle_open_timestamp = pd.Timestamp(row.timestamp)
        current_timestamp = _candle_completed_at(candle_open_timestamp, config.primary_timeframe)
        sliced_frames: dict[str, pd.DataFrame] = {}
        support_resistance: dict[str, SupportResistanceResult] = {}
        for timeframe in timeframes:
            prepared = prepared_frames[timeframe]
            sliced = _slice_available_history(
                prepared.frame,
                timeframe,
                current_timestamp,
                include_current=timeframe == config.primary_timeframe,
            )
            if len(sliced) < MIN_HISTORY_ROWS:
                break
            sliced_frames[timeframe] = sliced
            support_resistance[timeframe] = prepared.support_resistance[int(sliced.index[-1])]
        if len(sliced_frames) != len(timeframes):
            _maybe_emit_replay_progress(config, progress_callback, row_number, len(primary_frame), decisions, simulator)
            continue

        analyses = {
            timeframe: analyze_indicator_frame(
                config.symbol,
                timeframe,
                simulator.position_mode,
                sliced_frames[timeframe],
                support_resistance[timeframe],
            )
            for timeframe in timeframes
        }
        primary_analysis = analyses[config.primary_timeframe]
        multi_timeframe = analyze_multi_timeframe(
            {timeframe: build_timeframe_signal(analyses[timeframe]) for timeframe in config.timeframes}
        )
        final_decision = apply_multi_timeframe_alignment(
            primary_analysis.decision.decision,
            multi_timeframe.alignment.value,
            simulator.position_mode,
        )
        latest = primary_analysis.indicators.iloc[-1]
        daily_analysis = analyses.get("1d")
        four_hour_analysis = analyses.get("4h")
        daily_latest = daily_analysis.indicators.iloc[-1] if daily_analysis is not None else None
        four_hour_latest = four_hour_analysis.indicators.iloc[-1] if four_hour_analysis is not None else None
        weekly_latest = _latest_auxiliary_row(prepared_frames, "1w", current_timestamp)
        decision_record = {
            "timestamp": current_timestamp.isoformat(),
            "symbol": config.symbol,
            "position_mode": simulator.position_mode,
            "price": float(latest["close"]),
            "ema20": float(latest["ema_20"]),
            "setup": primary_analysis.setup.setup.value,
            "decision": primary_analysis.decision.decision.value,
            "final_decision": final_decision.decision.value,
            "final_decision_reason": final_decision.reason,
            "stop_loss": primary_analysis.decision.stop_loss,
            "target_1": primary_analysis.decision.target_1,
            "support": primary_analysis.support_resistance.support,
            "resistance": primary_analysis.support_resistance.resistance,
            "market_regime": primary_analysis.market_regime.market_regime.value,
            "trend_score": primary_analysis.scores.trend_score,
            "rr_ratio": primary_analysis.risk_reward.rr_ratio,
            "volume_ratio": calculate_volume_ratio(latest),
            "macd": macd_direction(latest),
            "daily_setup": daily_analysis.setup.setup.value if daily_analysis is not None else None,
            "daily_price": float(daily_latest["close"]) if daily_latest is not None else None,
            "daily_ema20": float(daily_latest["ema_20"]) if daily_latest is not None else None,
            "daily_ema50": float(daily_latest["ema_50"]) if daily_latest is not None else None,
            "daily_ema100": float(daily_latest["ema_100"]) if daily_latest is not None else None,
            "daily_ema200": float(daily_latest["ema_200"]) if daily_latest is not None else None,
            "daily_rsi": float(daily_latest["rsi_14"]) if daily_latest is not None else None,
            "daily_macd": macd_direction(daily_latest) if daily_latest is not None else None,
            "weekly_price": float(weekly_latest["close"]) if weekly_latest is not None else None,
            "weekly_ema20": float(weekly_latest["ema_20"]) if weekly_latest is not None else None,
            "weekly_ema50": float(weekly_latest["ema_50"]) if weekly_latest is not None else None,
            "weekly_rsi": float(weekly_latest["rsi_14"]) if weekly_latest is not None else None,
            "four_hour_price": float(four_hour_latest["close"]) if four_hour_latest is not None else None,
            "four_hour_ema20": float(four_hour_latest["ema_20"]) if four_hour_latest is not None else None,
            "four_hour_macd": macd_direction(four_hour_latest) if four_hour_latest is not None else None,
            "alignment": multi_timeframe.alignment.value,
            "alignment_score": multi_timeframe.alignment_score,
        }
        structure_stop = stop_engine.evaluate(
            sliced_frames[config.primary_timeframe],
            entry_price=float(latest["close"]),
            fixed_stop=primary_analysis.decision.stop_loss,
            support=primary_analysis.support_resistance.support,
        )
        decision_record.update(structure_stop.to_signal_fields())
        decision_record["atr_ma"] = calculate_atr_moving_average(sliced_frames[config.primary_timeframe])
        decisions.append(decision_record)
        snapshot = simulator.process_signal(decision_record)
        decision_record["rejected_entry_reasons"] = simulator.last_rejected_entry_reasons.copy()
        if first_signal_price is None:
            first_signal_price = float(latest["close"])
        last_decision_record = decision_record
        simulator.equity_curve[-1] = snapshot
        _maybe_emit_replay_progress(config, progress_callback, row_number, len(primary_frame), decisions, simulator)

    if config.close_open_position_on_end and simulator.position_size > 0 and last_decision_record is not None:
        final_snapshot = simulator.close_open_position(
            str(last_decision_record["timestamp"]),
            float(last_decision_record["price"]),
            last_decision_record,
        )
        if simulator.equity_curve:
            simulator.equity_curve[-1] = final_snapshot
        else:
            simulator.equity_curve.append(final_snapshot)

    equity_curve = simulator.equity_curve_frame()
    if equity_curve.empty or first_signal_price is None:
        raise BacktestError("No backtest decisions were generated. Check date range and history length.")
    equity_curve["buy_and_hold_equity"] = (
        config.initial_capital * equity_curve["price"].astype(float) / first_signal_price
    )
    trades = simulator.trades_as_dicts()
    metrics = calculate_performance_metrics(
        equity_curve=equity_curve,
        trades=trades,
        initial_capital=config.initial_capital,
        start_price=first_signal_price,
        end_price=float(equity_curve.iloc[-1]["price"]),
    )
    metrics.update(simulator.execution_summary(total_decisions=len(decisions)))
    return BacktestResult(
        symbol=config.symbol,
        profile=strategy_profile.name,
        strategy_profile=strategy_profile.to_dict(),
        start_date=pd.Timestamp(equity_curve.iloc[0]["timestamp"]).date().isoformat(),
        end_date=pd.Timestamp(equity_curve.iloc[-1]["timestamp"]).date().isoformat(),
        initial_capital=config.initial_capital,
        final_equity=float(metrics["final_equity"]),
        metrics=metrics,
        decisions=decisions,
        trades=trades,
        equity_curve=equity_curve,
    )


def run_benchmark_comparison(
    config: BacktestConfig,
    cached_data: dict[str, pd.DataFrame] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run passive, trend-filter, and aggressive-profile benchmarks."""

    from backtesting.benchmarks.research import run_benchmark_suite

    return run_benchmark_suite(config, cached_data=cached_data, progress_callback=progress_callback)


def _simulator_class(
    use_trend_rider: bool,
    use_hybrid_trend_rider: bool = False,
    use_trend_holding: bool = False,
    use_regime_gated_trend_holding: bool = False,
    use_portfolio_governor: bool = False,
) -> type[TradeSimulator]:
    if use_portfolio_governor:
        from backtesting.portfolio_governor_simulator import PortfolioGovernorSimulator

        return PortfolioGovernorSimulator
    if use_regime_gated_trend_holding:
        from backtesting.regime_gated_trend_holding_simulator import RegimeGatedTrendHoldingSimulator

        return RegimeGatedTrendHoldingSimulator
    if use_trend_holding:
        from backtesting.trend_holding_simulator import TrendHoldingSimulator

        return TrendHoldingSimulator
    if use_hybrid_trend_rider:
        from backtesting.hybrid_trend_rider_simulator import HybridTrendRiderSimulator

        return HybridTrendRiderSimulator
    if use_trend_rider:
        from backtesting.trend_rider_simulator import TrendRiderSimulator

        return TrendRiderSimulator
    return TradeSimulator


def _ensure_auxiliary_timeframes(frames: dict[str, pd.DataFrame], config: BacktestConfig) -> dict[str, pd.DataFrame]:
    if "1w" not in config.auxiliary_timeframes or "1w" in frames:
        return frames
    if "1d" not in frames:
        return frames
    enriched = dict(frames)
    enriched["1w"] = _resample_daily_to_weekly(frames["1d"])
    return enriched


def _resample_daily_to_weekly(daily_frame: pd.DataFrame) -> pd.DataFrame:
    frame = daily_frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").set_index("timestamp")
    weekly = frame.resample("W-MON", label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    weekly = weekly.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return weekly[list(OHLCV_COLUMNS)]


def _latest_auxiliary_row(
    prepared_frames: dict[str, PreparedTimeframeData],
    timeframe: str,
    current_timestamp: pd.Timestamp,
) -> pd.Series | None:
    prepared = prepared_frames.get(timeframe)
    if prepared is None:
        return None
    sliced = _slice_available_history(
        prepared.frame,
        timeframe,
        current_timestamp,
        include_current=False,
    )
    if sliced.empty:
        return None
    return sliced.iloc[-1]


def _prepare_timeframe_data(timeframe: str, frame: pd.DataFrame) -> PreparedTimeframeData:
    indicator_frame = add_indicators(frame).reset_index(drop=True)
    return PreparedTimeframeData(
        timeframe=timeframe,
        frame=indicator_frame,
        support_resistance=_precompute_support_resistance(indicator_frame),
    )


def _emit_progress(progress_callback: ProgressCallback | None, event: dict[str, Any]) -> None:
    if progress_callback is not None:
        progress_callback(event)


def _maybe_emit_replay_progress(
    config: BacktestConfig,
    progress_callback: ProgressCallback | None,
    row_number: int,
    total_rows: int,
    decisions: list[dict[str, Any]],
    simulator: TradeSimulator,
) -> None:
    if progress_callback is None or config.progress_interval <= 0:
        return
    if row_number != total_rows and row_number % config.progress_interval != 0:
        return
    pct_complete = round(row_number * 100 / total_rows, 1)
    latest_decision = decisions[-1] if decisions else {}
    _emit_progress(
        progress_callback,
        {
            "phase": "replay",
            "processed_rows": row_number,
            "total_rows": total_rows,
            "pct_complete": pct_complete,
            "decisions": len(decisions),
            "trades": len(simulator.trades),
            "timestamp": latest_decision.get("timestamp"),
            "latest_decision": latest_decision.get("final_decision"),
        },
    )


def load_or_download_timeframes(config: BacktestConfig, timeframes: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for timeframe in timeframes:
        cache_path = cache_file_path(config.cache_dir, config.symbol, timeframe)
        if cache_path.exists() and not config.refresh_cache:
            frame = _load_range_aware_cache(config, timeframe, cache_path)
        else:
            frame = download_market_history(
                symbol=config.symbol,
                interval=timeframe,
                start=config.start,
                end=config.end,
                binance_base_url=config.binance_base_url,
                yahoo_base_url=config.yahoo_base_url,
                timeout_seconds=config.request_timeout_seconds,
            )
        write_cached_ohlcv(frame, cache_path)
        frames[timeframe] = frame
    return frames


def _load_range_aware_cache(config: BacktestConfig, timeframe: str, cache_path: Path) -> pd.DataFrame:
    cached = read_cached_ohlcv(cache_path)
    interval_delta = pd.to_timedelta(INTERVAL_TO_MILLISECONDS[timeframe], unit="ms")
    requested_start = _timestamp(config.start)
    requested_end = None if config.end == "latest" else _timestamp(config.end)
    frames = [cached]
    cache_start = cached["timestamp"].min()
    cache_end = cached["timestamp"].max()

    if requested_start < cache_start:
        missing_end = cache_start - interval_delta
        if requested_start <= missing_end:
            frames.append(
                download_binance_history(
                    symbol=config.symbol,
                    interval=timeframe,
                    start=requested_start.isoformat(),
                    end=missing_end.isoformat(),
                    base_url=config.binance_base_url,
                    timeout_seconds=config.request_timeout_seconds,
                    allow_empty=True,
                )
                if is_crypto_symbol(config.symbol)
                else download_equity_history(
                    symbol=config.symbol,
                    interval=timeframe,
                    start=requested_start.isoformat(),
                    end=missing_end.isoformat(),
                    base_url=config.yahoo_base_url,
                    timeout_seconds=config.request_timeout_seconds,
                    allow_empty=True,
                )
            )

    needs_append = requested_end is None or requested_end > cache_end
    if needs_append:
        missing_start = cache_end + interval_delta
        missing_end = "latest" if requested_end is None else requested_end.isoformat()
        frames.append(
            download_binance_history(
                symbol=config.symbol,
                interval=timeframe,
                start=missing_start.isoformat(),
                end=missing_end,
                base_url=config.binance_base_url,
                timeout_seconds=config.request_timeout_seconds,
                allow_empty=True,
            )
            if is_crypto_symbol(config.symbol)
            else download_equity_history(
                symbol=config.symbol,
                interval=timeframe,
                start=missing_start.isoformat(),
                end=missing_end,
                base_url=config.yahoo_base_url,
                timeout_seconds=config.request_timeout_seconds,
                allow_empty=True,
            )
        )

    return _merge_ohlcv_frames(frames)


def download_market_history(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    binance_base_url: str = "https://api.binance.com",
    yahoo_base_url: str = "https://query1.finance.yahoo.com",
    timeout_seconds: float = 10.0,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Download public historical candles for supported crypto and equity symbols."""

    if is_crypto_symbol(symbol):
        return download_binance_history(
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
            base_url=binance_base_url,
            timeout_seconds=timeout_seconds,
            allow_empty=allow_empty,
        )
    return download_equity_history(
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        base_url=yahoo_base_url,
        timeout_seconds=timeout_seconds,
        allow_empty=allow_empty,
    )


def download_binance_history(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    base_url: str = "https://api.binance.com",
    timeout_seconds: float = 10.0,
    allow_empty: bool = False,
) -> pd.DataFrame:
    if interval not in INTERVAL_TO_MILLISECONDS:
        raise BacktestError(f"Unsupported interval {interval!r}.")
    start_ms = int(_timestamp(start).timestamp() * 1000)
    end_ms = None if end == "latest" else int(_timestamp(end).timestamp() * 1000)
    current_ms = start_ms
    all_rows: list[list[Any]] = []

    while True:
        query = {"symbol": symbol.upper(), "interval": interval, "limit": 1000, "startTime": current_ms}
        if end_ms is not None:
            query["endTime"] = end_ms
        request = Request(
            f"{base_url.rstrip('/')}{BINANCE_KLINES_PATH}?{urlencode(query)}",
            headers={"User-Agent": "trading-agent-backtest/0.1"},
            method="GET",
        )
        payload = _load_json_response(request, urlopen, timeout_seconds, "Binance")
        if not isinstance(payload, list):
            raise DataLoadError("Binance returned an unexpected historical payload shape.")
        if not payload:
            break
        all_rows.extend(payload)
        last_open_time = int(payload[-1][0])
        next_ms = last_open_time + INTERVAL_TO_MILLISECONDS[interval]
        if next_ms <= current_ms:
            break
        current_ms = next_ms
        if len(payload) < 1000 or (end_ms is not None and current_ms >= end_ms):
            break

    if not all_rows and allow_empty:
        return _empty_ohlcv_frame()
    if not all_rows:
        raise BacktestError(f"No historical candles found for {symbol} {interval}.")
    return normalize_klines(all_rows).drop_duplicates(subset=["timestamp"]).reset_index(drop=True)


def download_equity_history(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    base_url: str = "https://query1.finance.yahoo.com",
    timeout_seconds: float = 10.0,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Download public equity OHLCV candles through the provider fallback adapter."""

    del base_url
    try:
        frame = download_equity_timeframe(
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
            timeout_seconds=timeout_seconds,
        )
    except DataLoadError:
        if allow_empty:
            return _empty_ohlcv_frame()
        raise
    if frame.empty and allow_empty:
        return _empty_ohlcv_frame()
    if frame.empty:
        raise BacktestError(f"No historical candles found for {symbol} {interval}.")
    return frame


def normalize_yahoo_chart(payload: dict[str, Any], symbol: str, interval: str) -> pd.DataFrame:
    return normalize_yahoo_chart_payload(payload, symbol, interval)


def validate_downloaded_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[list(OHLCV_COLUMNS)].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    return frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)


def _resample_ohlcv(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    return resample_equity_ohlcv(frame, timeframe)


def _list_value(values: list[Any], index: int) -> Any:
    return values[index] if index < len(values) else None


def is_crypto_symbol(symbol: str) -> bool:
    return symbol.upper().endswith("USDT")


def _empty_ohlcv_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(OHLCV_COLUMNS))


def _merge_ohlcv_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return _empty_ohlcv_frame()
    merged = pd.concat(non_empty, ignore_index=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
    return (
        merged.drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)[list(OHLCV_COLUMNS)]
    )


def _timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def cache_file_path(cache_dir: Path, symbol: str, timeframe: str) -> Path:
    return cache_dir / f"{symbol.upper()}_{timeframe}.csv"


def read_cached_ohlcv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values("timestamp").reset_index(drop=True)


def write_cached_ohlcv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _precompute_support_resistance(
    frame: pd.DataFrame,
    swing_window: int = 2,
    fallback_window: int = 20,
) -> list[SupportResistanceResult]:
    if frame.empty:
        return []
    swing_lows = _swing_levels(frame["low"], swing_window, "support")
    swing_highs = _swing_levels(frame["high"], swing_window, "resistance")
    support_results: list[SupportResistanceResult] = []
    known_lows: list[float] = []
    known_highs: list[float] = []
    low_cursor = 0
    high_cursor = 0

    lows = frame["low"].reset_index(drop=True)
    highs = frame["high"].reset_index(drop=True)
    closes = frame["close"].reset_index(drop=True)
    for index, current_price in enumerate(closes):
        while low_cursor < len(swing_lows) and swing_lows[low_cursor][0] <= index:
            known_lows.append(swing_lows[low_cursor][1])
            low_cursor += 1
        while high_cursor < len(swing_highs) and swing_highs[high_cursor][0] <= index:
            known_highs.append(swing_highs[high_cursor][1])
            high_cursor += 1

        price = float(current_price)
        fallback_start = max(0, index - fallback_window + 1)
        recent_lows = lows.iloc[fallback_start : index + 1]
        recent_highs = highs.iloc[fallback_start : index + 1]
        support = _latest_level(known_lows, price, "support")
        resistance = _latest_level(known_highs, price, "resistance")
        if support is None:
            support = float(recent_lows.min())
        if resistance is None:
            resistance = float(recent_highs.max())

        support_results.append(
            SupportResistanceResult(
                support=support,
                resistance=resistance,
                distance_to_support=max(0.0, price - support),
                distance_to_resistance=max(0.0, resistance - price),
                sr_score=_score_support_resistance(price, support, resistance),
            )
        )
    return support_results


def _swing_levels(series: pd.Series, window: int, side: str) -> list[tuple[int, float]]:
    values = [float(value) for value in series.reset_index(drop=True)]
    levels: list[tuple[int, float]] = []
    for index in range(window, len(values) - window):
        value = values[index]
        neighbors = values[index - window : index + window + 1]
        other_neighbors = neighbors[:window] + neighbors[window + 1 :]
        if side == "support":
            is_swing = value == min(neighbors) and value < min(other_neighbors)
        else:
            is_swing = value == max(neighbors) and value > max(other_neighbors)
        if is_swing:
            levels.append((index + window, value))
    return levels


def _latest_level(levels: list[float], current_price: float, side: str) -> float | None:
    if side == "support":
        candidates = (level for level in reversed(levels) if level <= current_price)
    else:
        candidates = (level for level in reversed(levels) if level >= current_price)
    return next(candidates, None)


def _score_support_resistance(current_price: float, support: float, resistance: float) -> int:
    if resistance <= support:
        return 5

    distance_to_support = max(0.0, current_price - support)
    distance_to_resistance = max(0.0, resistance - current_price)
    price_range = resistance - support

    near_support = distance_to_support <= price_range * 0.25
    far_from_resistance = distance_to_resistance >= price_range * 0.50
    near_resistance = distance_to_resistance <= price_range * 0.25

    if near_resistance:
        return 0
    if near_support and far_from_resistance:
        return 10
    return 5


def _ordered_timeframes(primary_timeframe: str, timeframes: tuple[str, ...]) -> tuple[str, ...]:
    ordered = [primary_timeframe]
    for timeframe in timeframes:
        if timeframe not in ordered:
            ordered.append(timeframe)
    return tuple(ordered)


def _filter_date_range(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    filtered = frame.copy()
    filtered["timestamp"] = pd.to_datetime(filtered["timestamp"], utc=True)
    start_ts = pd.Timestamp(start, tz="UTC")
    filtered = filtered[filtered["timestamp"] >= start_ts]
    if end != "latest":
        end_ts = pd.Timestamp(end, tz="UTC")
        filtered = filtered[filtered["timestamp"] <= end_ts]
    return filtered.sort_values("timestamp").reset_index(drop=True)


def _filter_history_until_end(frame: pd.DataFrame, end: str) -> pd.DataFrame:
    filtered = frame.copy()
    filtered["timestamp"] = pd.to_datetime(filtered["timestamp"], utc=True)
    if end != "latest":
        end_ts = pd.Timestamp(end, tz="UTC")
        filtered = filtered[filtered["timestamp"] <= end_ts]
    return filtered.sort_values("timestamp").reset_index(drop=True)


def _slice_available_history(
    frame: pd.DataFrame,
    timeframe: str,
    current_timestamp: pd.Timestamp,
    include_current: bool,
) -> pd.DataFrame:
    timestamp = current_timestamp.tz_localize("UTC") if current_timestamp.tzinfo is None else current_timestamp
    # Binance timestamps are candle open times. A candle is available to the
    # strategy only after its full interval has elapsed.
    completed_at = frame["timestamp"] + pd.to_timedelta(INTERVAL_TO_MILLISECONDS[timeframe], unit="ms")
    return frame[completed_at <= timestamp].copy()


def _candle_completed_at(candle_open_timestamp: pd.Timestamp, timeframe: str) -> pd.Timestamp:
    timestamp = (
        candle_open_timestamp.tz_localize("UTC")
        if candle_open_timestamp.tzinfo is None
        else candle_open_timestamp
    )
    return timestamp + pd.to_timedelta(INTERVAL_TO_MILLISECONDS[timeframe], unit="ms")


def result_summary_json(result: BacktestResult) -> str:
    return json.dumps({"symbol": result.symbol, "profile": result.profile, **result.metrics}, indent=2)
