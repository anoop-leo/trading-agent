"""Coinbase Advanced read-only shadow trading for Phase 1.20."""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from backtesting.profiles import get_strategy_profile
from data.coinbase_execution_cost_audit import (
    CoinbaseExecutionAuditConfig,
    build_orderbook_sample,
    fetch_public_product_book,
    normalize_coinbase_product_id,
)
from decision.decision_engine import apply_multi_timeframe_alignment
from risk.structure_stop_engine import calculate_atr
from scoring.multi_timeframe_skill import analyze_multi_timeframe
from trading_agent.data import DataLoadError
from trading_agent.main import analyze_timeframe, build_timeframe_signal
from trading_agent.output import macd_direction
from trading_agent.scoring import calculate_volume_ratio


SHADOW_TRADE_COLUMNS = (
    "signal_timestamp",
    "exit_timestamp",
    "action",
    "signal_price",
    "simulated_entry_price",
    "simulated_exit_price",
    "position_size",
    "fee_estimate",
    "slippage_estimate",
    "gross_pnl",
    "net_pnl",
    "reason_for_entry",
    "reason_for_exit",
    "confidence",
    "market_regime",
    "alignment_score",
    "stop_loss",
    "target_1",
    "target_2",
    "max_drawdown_during_trade",
    "max_favorable_excursion",
    "max_adverse_excursion",
    "entry_all_in_cost_per_side",
    "exit_all_in_cost_per_side",
)
EQUITY_COLUMNS = (
    "timestamp",
    "current_shadow_equity",
    "cash",
    "position_value",
    "open_position",
    "price",
    "cumulative_pnl",
)
SHADOW_SIGNAL_COLUMNS = (
    "timestamp",
    "market_data_timestamp",
    "symbol",
    "product_id",
    "timeframe",
    "position_mode",
    "action",
    "price",
    "setup",
    "final_decision",
    "decision_reason",
    "confidence",
    "ema20",
    "ema50",
    "ema200",
    "rsi",
    "macd",
    "macd_signal",
    "macd_histogram",
    "volume_ratio",
    "atr",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "bb_width",
    "support",
    "resistance",
    "distance_to_support",
    "distance_to_resistance",
    "rr_ratio",
    "stop_loss",
    "target_1",
    "target_2",
    "multi_timeframe_alignment",
    "alignment",
    "alignment_score",
    "one_hour_alignment",
    "four_hour_alignment",
    "daily_alignment",
    "one_hour_market_regime",
    "four_hour_market_regime",
    "daily_market_regime",
    "one_hour_setup",
    "four_hour_setup",
    "daily_setup",
    "four_hour_macd",
    "daily_macd",
    "rejected",
    "rejection_reasons",
    "rejection_categories",
    "blocking_rule",
    "blocking_timeframe",
    "avoid_long_reason",
    "watch_long_trigger_eligible",
    "watch_long_blocked_by",
    "price_above_1h_ema20",
    "price_above_1h_ema50",
    "four_hour_macd_improving",
    "rsi_above_45",
    "near_support",
    "broke_resistance",
    "daily_trend_not_strongly_bearish",
    "price_plus_1d",
    "price_plus_3d",
    "price_plus_7d",
    "max_favorable_move_3d",
    "max_adverse_move_3d",
    "avoid_classification",
    "all_in_cost_per_side",
    "price_slippage_pct",
    "depth_supported",
    # Legacy Phase 1.20 aliases kept for existing analysis tools.
    "ema20",
    "ema50",
    "ema200",
    "rsi",
    "macd",
    "setup",
    "decision",
    "final_decision",
    "final_decision_reason",
    "confidence",
    "market_regime",
    "trend_score",
    "rr_ratio",
    "volume_ratio",
    "daily_setup",
    "daily_price",
    "daily_ema20",
    "daily_ema200",
    "daily_rsi",
    "daily_macd",
    "four_hour_price",
    "four_hour_ema20",
    "four_hour_macd",
    "alignment",
    "alignment_score",
    "rejection_reasons",
    "all_in_cost_per_side",
    "price_slippage_pct",
    "depth_supported",
)
SHADOW_SIGNAL_COLUMNS = tuple(dict.fromkeys(SHADOW_SIGNAL_COLUMNS))
SIGNAL_QUALITY_REQUIRED_FIELDS = (
    "price",
    "setup",
    "final_decision",
    "ema20",
    "ema50",
    "rsi",
    "macd",
    "volume_ratio",
    "market_regime",
    "multi_timeframe_alignment",
    "rejection_reasons",
)
ENTRY_DECISIONS = {"BUY", "BUY WATCH", "STRONG BUY"}
EXIT_DECISIONS = {"EXIT", "REDUCE", "AVOID LONG"}
COINBASE_GRANULARITIES = {
    "1h": ("ONE_HOUR", 3600),
    "4h": ("FOUR_HOUR", 14400),
    "1d": ("ONE_DAY", 86400),
}
DEFAULT_TIMEFRAMES = ("1h", "4h", "1d")

JsonOpener = Callable[..., Any]
SleepFn = Callable[[float], None]
NowFn = Callable[[], datetime]
ProgressCallback = Callable[[dict[str, Any]], None]


class ShadowTradingError(RuntimeError):
    """Raised when the read-only shadow trading workflow cannot continue."""


@dataclass(frozen=True)
class ShadowTradingConfig:
    product_id: str = "BTC-USD"
    initial_shadow_capital: float = 10000.0
    intended_order_size_usd: float = 2500.0
    fee_rate: float = 0.001
    max_all_in_cost_per_side: float = 0.0015
    duration_days: float = 30.0
    cycle_interval_seconds: float = 3600.0
    cycle_limit: int | None = None
    history_limit: int = 220
    output_dir: Path = Path("outputs")
    base_url: str = "https://api.coinbase.com/api/v3/brokerage"
    timeout_seconds: float = 10.0
    order_book_limit: int = 50
    latency_issue_threshold_seconds: float = 2.0
    live_trading_enabled: bool = False
    order_endpoint_calls_allowed: bool = False
    max_position_count: int = 1
    emergency_stop_if_data_missing: bool = True
    emergency_stop_if_api_errors_exceed_5_percent: bool = True
    target_signal_count: int | None = None
    resume_signal_collection: bool = True


@dataclass(frozen=True)
class ShadowSignalCollectionConfig:
    product_id: str = "BTC-USD"
    target_signals: int = 50
    interval_seconds: float = 3600.0
    history_limit: int = 220
    output_dir: Path = Path("outputs")
    base_url: str = "https://api.coinbase.com/api/v3/brokerage"
    timeout_seconds: float = 10.0
    order_book_limit: int = 50
    intended_order_size_usd: float = 2500.0
    fee_rate: float = 0.001
    max_all_in_cost_per_side: float = 0.0015
    reset: bool = False
    live_trading_enabled: bool = False
    order_endpoint_calls_allowed: bool = False


class CoinbaseCandleProvider:
    """Read-only Coinbase Advanced public candle provider."""

    def __init__(
        self,
        base_url: str = "https://api.coinbase.com/api/v3/brokerage",
        timeout_seconds: float = 10.0,
        opener: JsonOpener = urlopen,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._opener = opener
        self.request_paths: list[str] = []

    def fetch_ohlcv(self, product_id: str, interval: str, limit: int, end_time: datetime | None = None) -> pd.DataFrame:
        if interval not in COINBASE_GRANULARITIES:
            raise DataLoadError(f"Unsupported Coinbase candle interval {interval!r}.")
        granularity, seconds = COINBASE_GRANULARITIES[interval]
        end_timestamp = _as_utc(end_time or datetime.now(UTC))
        start_timestamp = end_timestamp - timedelta(seconds=seconds * limit)
        product = normalize_coinbase_product_id(product_id)
        path = f"/market/products/{product}/candles"
        query = urlencode(
            {
                "start": int(start_timestamp.timestamp()),
                "end": int(end_timestamp.timestamp()),
                "granularity": granularity,
            }
        )
        url = f"{self.base_url}{path}?{query}"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "Cache-Control": "no-cache",
                "User-Agent": "trading-agent-phase-1.20",
            },
            method="GET",
        )
        self.request_paths.append(path)
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise DataLoadError(f"Coinbase candle HTTP {exc.code}.") from exc
        except URLError as exc:
            raise DataLoadError(f"Coinbase candle request failed: {exc.reason}.") from exc
        except json.JSONDecodeError as exc:
            raise DataLoadError("Coinbase returned invalid candle JSON.") from exc
        return normalize_coinbase_candles(payload)


def normalize_coinbase_candles(payload: dict[str, Any]) -> pd.DataFrame:
    """Normalize Coinbase candle payloads to the local OHLCV shape."""

    candles = payload.get("candles") if isinstance(payload, dict) else None
    if not isinstance(candles, list) or not candles:
        raise DataLoadError("Coinbase returned no candles.")
    rows = []
    for candle in candles:
        if not isinstance(candle, dict):
            raise DataLoadError("Coinbase candle payload has malformed rows.")
        try:
            rows.append(
                {
                    "timestamp": pd.to_datetime(int(candle["start"]), unit="s", utc=True),
                    "open": float(candle["open"]),
                    "high": float(candle["high"]),
                    "low": float(candle["low"]),
                    "close": float(candle["close"]),
                    "volume": float(candle["volume"]),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DataLoadError("Coinbase candle row has invalid OHLCV values.") from exc
    return pd.DataFrame(rows).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


@dataclass
class ShadowPosition:
    entry_timestamp: str
    signal_price: float
    simulated_entry_price: float
    position_size: float
    notional: float
    entry_fee: float
    entry_slippage_cost: float
    reason_for_entry: str
    confidence: int
    market_regime: str
    alignment_score: int
    stop_loss: float | None
    target_1: float | None
    target_2: float | None
    entry_all_in_cost_per_side: float
    highest_price: float
    lowest_price: float


class ShadowPortfolio:
    """Single-position shadow portfolio with simulated Coinbase costs."""

    def __init__(
        self,
        initial_capital: float,
        intended_order_size_usd: float,
        max_all_in_cost_per_side: float,
        fee_rate: float,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.intended_order_size_usd = float(intended_order_size_usd)
        self.max_all_in_cost_per_side = float(max_all_in_cost_per_side)
        self.fee_rate = float(fee_rate)
        self.open_position: ShadowPosition | None = None
        self.closed_trades: list[dict[str, Any]] = []
        self.rejected_signals: list[dict[str, Any]] = []
        self.last_rejected_signal_reasons: list[str] = []

    def process_signal(
        self,
        signal: dict[str, Any],
        execution_cost: dict[str, float | bool | None],
    ) -> tuple[str, dict[str, Any] | None]:
        price = float(signal["price"])
        self.last_rejected_signal_reasons = []
        if self.open_position is not None:
            self._update_position_extremes(price)
            exit_reason = self._exit_reason(signal, price)
            if exit_reason is not None:
                return "SELL", self._close_position(signal, execution_cost, exit_reason)
            return "HOLD", None

        rejection_reasons = self._entry_rejection_reasons(signal, execution_cost)
        if rejection_reasons:
            self.last_rejected_signal_reasons = rejection_reasons.copy()
            action = "AVOID" if str(signal.get("final_decision")) == "AVOID LONG" else "HOLD"
            self.rejected_signals.append(
                {
                    "timestamp": signal["timestamp"],
                    "final_decision": signal.get("final_decision"),
                    "reasons": rejection_reasons,
                }
            )
            return action, None
        self._open_position(signal, execution_cost)
        return "BUY", None

    def current_equity(self, current_price: float) -> float:
        if self.open_position is None:
            return self.cash
        return self.cash + (self.open_position.position_size * float(current_price))

    def _entry_rejection_reasons(
        self,
        signal: dict[str, Any],
        execution_cost: dict[str, float | bool | None],
    ) -> list[str]:
        reasons: list[str] = []
        if str(signal.get("final_decision")) not in ENTRY_DECISIONS:
            reasons.append("not_buy_decision")
        if str(signal.get("alignment")) not in {"BULLISH_ALIGNMENT", "PULLBACK_IN_UPTREND"}:
            reasons.append("bearish_alignment")
        if _optional_float(signal.get("rr_ratio")) is None or float(signal.get("rr_ratio", 0.0)) < 1.5:
            reasons.append("low_rr_ratio")
        if _optional_float(signal.get("volume_ratio")) is None or float(signal.get("volume_ratio", 0.0)) < 0.8:
            reasons.append("low_volume_ratio")
        if signal.get("market_regime") == "BEAR":
            reasons.append("bear_market_regime")
        if signal.get("daily_setup") == "BEAR_TREND":
            reasons.append("daily_bear_trend")
        four_hour_price = _optional_float(signal.get("four_hour_price"))
        four_hour_ema20 = _optional_float(signal.get("four_hour_ema20"))
        if four_hour_price is None or four_hour_ema20 is None or four_hour_price <= four_hour_ema20:
            reasons.append("below_4h_ema20")
        all_in = _optional_float(execution_cost.get("all_in_cost_per_side"))
        depth_supported = bool(execution_cost.get("depth_supported"))
        if all_in is None or all_in > self.max_all_in_cost_per_side:
            reasons.append("high_execution_cost")
        if not depth_supported:
            reasons.append("insufficient_order_book_depth")
        return reasons

    def _open_position(self, signal: dict[str, Any], execution_cost: dict[str, float | bool | None]) -> None:
        signal_price = float(signal["price"])
        all_in = float(execution_cost.get("all_in_cost_per_side") or self.fee_rate)
        estimated_slippage = float(execution_cost.get("price_slippage_pct") or max(all_in - self.fee_rate, 0.0))
        simulated_entry_price = signal_price * (1 + estimated_slippage)
        entry_notional = min(self.intended_order_size_usd, self.cash / (1 + self.fee_rate))
        entry_fee = entry_notional * self.fee_rate
        position_size = entry_notional / simulated_entry_price
        self.cash -= entry_notional + entry_fee
        self.open_position = ShadowPosition(
            entry_timestamp=str(signal["timestamp"]),
            signal_price=signal_price,
            simulated_entry_price=simulated_entry_price,
            position_size=position_size,
            notional=entry_notional,
            entry_fee=entry_fee,
            entry_slippage_cost=position_size * (simulated_entry_price - signal_price),
            reason_for_entry=str(signal.get("final_decision_reason") or "all_shadow_entry_gates_passed"),
            confidence=int(signal.get("confidence", 0)),
            market_regime=str(signal.get("market_regime", "")),
            alignment_score=int(signal.get("alignment_score", 0)),
            stop_loss=_optional_float(signal.get("stop_loss")),
            target_1=_optional_float(signal.get("target_1")),
            target_2=_optional_float(signal.get("target_2")),
            entry_all_in_cost_per_side=all_in,
            highest_price=signal_price,
            lowest_price=signal_price,
        )

    def _exit_reason(self, signal: dict[str, Any], price: float) -> str | None:
        position = self.open_position
        if position is None:
            return None
        if position.stop_loss is not None and price <= position.stop_loss:
            return "STOP_LOSS"
        if position.target_1 is not None and price >= position.target_1:
            return "TAKE_PROFIT"
        if str(signal.get("final_decision")) in EXIT_DECISIONS:
            return str(signal.get("final_decision"))
        if signal.get("market_regime") == "BEAR" and int(signal.get("trend_score", 10)) <= 3:
            return "BEAR_TREND"
        return None

    def _close_position(
        self,
        signal: dict[str, Any],
        execution_cost: dict[str, float | bool | None],
        reason_for_exit: str,
    ) -> dict[str, Any]:
        position = self.open_position
        if position is None:
            raise ShadowTradingError("Cannot close a missing shadow position.")
        signal_exit_price = float(signal["price"])
        all_in = float(execution_cost.get("all_in_cost_per_side") or self.fee_rate)
        estimated_slippage = float(execution_cost.get("price_slippage_pct") or max(all_in - self.fee_rate, 0.0))
        simulated_exit_price = signal_exit_price * (1 - estimated_slippage)
        gross_proceeds = position.position_size * simulated_exit_price
        exit_fee = gross_proceeds * self.fee_rate
        net_proceeds = gross_proceeds - exit_fee
        gross_pnl = position.position_size * (signal_exit_price - position.signal_price)
        net_pnl = net_proceeds - position.notional - position.entry_fee
        self.cash += net_proceeds
        max_favorable = position.position_size * (position.highest_price - position.signal_price)
        max_adverse = position.position_size * (position.lowest_price - position.signal_price)
        drawdown = _pct_ratio((position.highest_price - position.lowest_price) / position.highest_price)
        trade = {
            "signal_timestamp": position.entry_timestamp,
            "exit_timestamp": str(signal["timestamp"]),
            "action": "SELL",
            "signal_price": position.signal_price,
            "simulated_entry_price": position.simulated_entry_price,
            "simulated_exit_price": simulated_exit_price,
            "position_size": position.position_size,
            "fee_estimate": position.entry_fee + exit_fee,
            "slippage_estimate": position.entry_slippage_cost + position.position_size * (signal_exit_price - simulated_exit_price),
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "reason_for_entry": position.reason_for_entry,
            "reason_for_exit": reason_for_exit,
            "confidence": position.confidence,
            "market_regime": position.market_regime,
            "alignment_score": position.alignment_score,
            "stop_loss": position.stop_loss,
            "target_1": position.target_1,
            "target_2": position.target_2,
            "max_drawdown_during_trade": drawdown,
            "max_favorable_excursion": max_favorable,
            "max_adverse_excursion": max_adverse,
            "entry_all_in_cost_per_side": position.entry_all_in_cost_per_side,
            "exit_all_in_cost_per_side": all_in,
        }
        self.closed_trades.append(trade)
        self.open_position = None
        return trade

    def _update_position_extremes(self, price: float) -> None:
        if self.open_position is None:
            return
        self.open_position.highest_price = max(self.open_position.highest_price, float(price))
        self.open_position.lowest_price = min(self.open_position.lowest_price, float(price))


@dataclass
class ShadowHealth:
    system_errors: int = 0
    missing_data_events: int = 0
    api_latency_issues: int = 0
    api_requests: int = 0
    api_failures: int = 0
    unauthorized_order_endpoint_calls: int = 0
    emergency_stop_triggered: bool = False
    emergency_stop_reason: str | None = None


def run_coinbase_shadow_trading(
    config: ShadowTradingConfig,
    opener: JsonOpener = urlopen,
    sleeper: SleepFn = time.sleep,
    now_fn: NowFn | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run read-only Coinbase shadow trading for Agent Aggressive."""

    _validate_shadow_safety(config)
    now = now_fn or (lambda: datetime.now(UTC))
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    end_time = _as_utc(now()) + timedelta(days=config.duration_days)
    provider = CoinbaseCandleProvider(config.base_url, config.timeout_seconds, opener)
    portfolio = ShadowPortfolio(
        config.initial_shadow_capital,
        config.intended_order_size_usd,
        config.max_all_in_cost_per_side,
        config.fee_rate,
    )
    health = ShadowHealth()
    signal_rows: list[dict[str, Any]] = (
        _load_existing_signal_rows(config.output_dir / "signal_journal_v2.json")
        if config.resume_signal_collection
        else []
    )
    signal_count = len(signal_rows)
    equity_rows: list[dict[str, Any]] = []
    last_price = 0.0
    cycle_count = 0

    while _as_utc(now()) < end_time and _target_signal_count_remaining(config, signal_rows):
        cycle_count += 1
        cycle_started = _as_utc(now())
        try:
            frames = _fetch_shadow_frames(config, provider, cycle_started, health)
            signal = build_shadow_signal(config, frames, portfolio.open_position is not None)
            execution_cost = _fetch_execution_cost(config, opener, cycle_started, health)
            action, _trade = portfolio.process_signal(signal, execution_cost)
            signal_row = _shadow_signal_row(signal, action, portfolio.last_rejected_signal_reasons, execution_cost)
            _validate_signal_quality_or_raise(signal_row)
            signal_rows = _upsert_signal_row(signal_rows, signal_row)
            signal_count = len(signal_rows)
            last_price = float(signal["price"])
            equity_rows.append(_equity_row(portfolio, signal["timestamp"], last_price, config.initial_shadow_capital))
            _write_shadow_outputs(config, portfolio, equity_rows, signal_rows, health, signal_count, cycle_started)
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "coinbase_shadow_cycle",
                        "cycle": cycle_count,
                        "action": action,
                        "decision": signal["final_decision"],
                        "price": last_price,
                        "cumulative_trades": len(portfolio.closed_trades),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - a shadow monitor should report and stop safely.
            health.system_errors += 1
            health.emergency_stop_triggered = True
            health.emergency_stop_reason = str(exc)
            _write_shadow_outputs(config, portfolio, equity_rows, signal_rows, health, signal_count, cycle_started)
            break

        if _emergency_stop_required(config, health):
            health.emergency_stop_triggered = True
            health.emergency_stop_reason = "API error or missing-data threshold exceeded."
            _write_shadow_outputs(config, portfolio, equity_rows, signal_rows, health, signal_count, cycle_started)
            break
        if config.cycle_limit is not None and cycle_count >= config.cycle_limit:
            break
        sleep_seconds = min(config.cycle_interval_seconds, max((end_time - _as_utc(now())).total_seconds(), 0.0))
        if sleep_seconds > 0:
            sleeper(sleep_seconds)

    final_price = last_price if last_price > 0 else 1.0
    summary = _shadow_summary(config, portfolio, equity_rows, health, signal_count, final_price, _as_utc(now()))
    _write_json(config.output_dir / "shadow_summary_30d.json", summary)
    _write_enriched_false_avoid_analysis(config.output_dir, signal_rows)
    return {
        "shadow_summary_30d": summary,
        "artifacts": {
            "shadow_trades": str(config.output_dir / "shadow_trades.csv"),
            "shadow_equity_curve": str(config.output_dir / "shadow_equity_curve.csv"),
            "shadow_signals": str(config.output_dir / "shadow_signals.csv"),
            "shadow_signals_v2": str(config.output_dir / "shadow_signals_v2.csv"),
            "signal_journal_v2": str(config.output_dir / "signal_journal_v2.json"),
            "signal_journal_quality_report": str(config.output_dir / "signal_journal_quality_report.json"),
            "enriched_false_avoid_analysis": str(config.output_dir / "enriched_false_avoid_analysis.json"),
            "shadow_system_health": str(config.output_dir / "shadow_system_health.json"),
            "shadow_summary_30d": str(config.output_dir / "shadow_summary_30d.json"),
        },
    }


def collect_enriched_shadow_signals(
    config: ShadowSignalCollectionConfig,
    opener: JsonOpener = urlopen,
    sleeper: SleepFn = time.sleep,
    now_fn: NowFn | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Collect append-only enriched signal snapshots for Phase 1.23."""

    _validate_collection_safety(config)
    if config.target_signals <= 0:
        raise ShadowTradingError("target_signals must be greater than zero.")
    if config.interval_seconds < 0:
        raise ShadowTradingError("interval_seconds must be non-negative.")

    now = now_fn or (lambda: datetime.now(UTC))
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.reset:
        _reset_signal_collection_outputs(output_dir)

    signal_rows = _load_existing_signal_rows(output_dir / "signal_journal_v2.json")
    health = ShadowHealth()
    provider = CoinbaseCandleProvider(config.base_url, config.timeout_seconds, opener)
    shadow_config = _collection_shadow_config(config)
    cycle_count = 0

    while len(signal_rows) < config.target_signals:
        cycle_count += 1
        cycle_started = _as_utc(now())
        try:
            frames = _fetch_shadow_frames(shadow_config, provider, cycle_started, health)
            signal = build_shadow_signal(shadow_config, frames, holding_position=False)
            signal["market_data_timestamp"] = signal["timestamp"]
            signal["timestamp"] = _isoformat(cycle_started)
            execution_cost = _fetch_execution_cost(shadow_config, opener, cycle_started, health)
            rejection_reasons = _snapshot_rejection_reasons(signal, execution_cost, shadow_config)
            action = "AVOID" if str(signal.get("final_decision")) == "AVOID LONG" else "SIGNAL"
            signal_row = _shadow_signal_row(signal, action, rejection_reasons, execution_cost)
            _validate_signal_quality_or_raise(signal_row)
            before_count = len(signal_rows)
            signal_rows = _upsert_signal_row(signal_rows, signal_row)
            signal_rows = _enrich_signal_rows_with_forward_evaluations(signal_rows)
            if len(signal_rows) > before_count:
                _append_signal_csv_row(output_dir / "shadow_signals_v2.csv", signal_row)
                _append_signal_csv_row(output_dir / "shadow_signals.csv", signal_row)
            _write_signal_collection_outputs(output_dir, signal_rows, target_signals=config.target_signals)
            _write_collection_health_outputs(output_dir, shadow_config, health, signal_rows, cycle_started)
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "enriched_shadow_signal_collection",
                        "cycle": cycle_count,
                        "total_signals": len(signal_rows),
                        "target_signals": config.target_signals,
                        "final_decision": signal_row["final_decision"],
                        "price": signal_row["price"],
                    }
                )
        except Exception as exc:  # noqa: BLE001 - collection should persist diagnostics before stopping.
            health.system_errors += 1
            health.emergency_stop_triggered = True
            health.emergency_stop_reason = str(exc)
            _write_collection_health_outputs(output_dir, shadow_config, health, signal_rows, cycle_started)
            raise

        if len(signal_rows) >= config.target_signals:
            break
        if config.interval_seconds > 0:
            sleeper(config.interval_seconds)

    report = _enriched_false_avoid_analysis(signal_rows, target_signals=config.target_signals)
    return {
        "enriched_false_avoid_analysis": report,
        "signal_journal_quality_report": _signal_quality_report(signal_rows),
        "shadow_system_health": _health_payload(shadow_config, health, len(signal_rows)),
        "artifacts": {
            "signal_journal_v2": str(output_dir / "signal_journal_v2.json"),
            "shadow_signals_v2": str(output_dir / "shadow_signals_v2.csv"),
            "signal_journal_quality_report": str(output_dir / "signal_journal_quality_report.json"),
            "shadow_system_health": str(output_dir / "shadow_system_health.json"),
            "enriched_false_avoid_analysis": str(output_dir / "enriched_false_avoid_analysis.json"),
        },
    }


def build_shadow_signal(
    config: ShadowTradingConfig,
    frames: dict[str, pd.DataFrame],
    holding_position: bool,
) -> dict[str, Any]:
    """Run the deterministic strategy pipeline for one shadow cycle."""

    symbol = normalize_coinbase_product_id(config.product_id).replace("-", "")
    position_mode = "HOLDING" if holding_position else "NO_POSITION"
    analyses = {
        timeframe: analyze_timeframe(symbol, timeframe, position_mode, frames[timeframe])
        for timeframe in DEFAULT_TIMEFRAMES
    }
    timeframe_signals = {
        timeframe: build_timeframe_signal(analyses[timeframe]) for timeframe in DEFAULT_TIMEFRAMES
    }
    primary = analyses["1h"]
    multi_timeframe = analyze_multi_timeframe(timeframe_signals)
    final_decision = apply_multi_timeframe_alignment(
        primary.decision.decision,
        multi_timeframe.alignment.value,
        position_mode,
    )
    latest = primary.indicators.iloc[-1]
    one_hour_latest = latest
    daily_latest = analyses["1d"].indicators.iloc[-1]
    four_hour_latest = analyses["4h"].indicators.iloc[-1]
    current_price = float(latest["close"])
    support = primary.support_resistance.support
    resistance = primary.support_resistance.resistance
    four_hour_macd = macd_direction(four_hour_latest)
    daily_macd = macd_direction(daily_latest)
    watch_long_fields = _watch_long_fields(
        price=current_price,
        ema20=float(one_hour_latest["ema_20"]),
        ema50=float(one_hour_latest["ema_50"]),
        rsi=float(one_hour_latest["rsi_14"]),
        support=support,
        resistance=resistance,
        setup=primary.setup.setup.value,
        market_regime=primary.market_regime.market_regime.value,
        daily_setup=analyses["1d"].setup.setup.value,
        daily_market_regime=analyses["1d"].market_regime.market_regime.value,
        four_hour_macd=four_hour_macd,
        four_hour_macd_improving=_macd_histogram_improving(analyses["4h"].indicators),
    )
    return {
        "timestamp": pd.Timestamp(latest["timestamp"]).isoformat(),
        "symbol": symbol,
        "product_id": normalize_coinbase_product_id(config.product_id),
        "timeframe": "1h",
        "position_mode": position_mode,
        "price": current_price,
        "ema20": float(latest["ema_20"]),
        "ema50": float(latest["ema_50"]),
        "ema200": float(latest["ema_200"]),
        "rsi": float(latest["rsi_14"]),
        "macd_signal": float(latest["macd_signal"]),
        "macd_histogram": float(latest["macd_histogram"]),
        "atr": calculate_atr(primary.indicators),
        "bb_upper": float(latest["bb_upper"]),
        "bb_middle": float(latest["bb_middle"]),
        "bb_lower": float(latest["bb_lower"]),
        "bb_width": _bb_width(latest),
        "setup": primary.setup.setup.value,
        "decision": primary.decision.decision.value,
        "final_decision": final_decision.decision.value,
        "decision_reason": final_decision.reason,
        "final_decision_reason": final_decision.reason,
        "confidence": primary.decision.confidence,
        "stop_loss": primary.decision.stop_loss,
        "target_1": primary.decision.target_1,
        "target_2": primary.decision.target_2,
        "market_regime": primary.market_regime.market_regime.value,
        "trend_score": primary.scores.trend_score,
        "support": support,
        "resistance": resistance,
        "distance_to_support": primary.support_resistance.distance_to_support,
        "distance_to_resistance": primary.support_resistance.distance_to_resistance,
        "rr_ratio": primary.risk_reward.rr_ratio,
        "volume_ratio": calculate_volume_ratio(latest),
        "macd": macd_direction(latest),
        "multi_timeframe_alignment": multi_timeframe.alignment.value,
        "alignment": multi_timeframe.alignment.value,
        "alignment_score": multi_timeframe.alignment_score,
        "one_hour_alignment": _timeframe_alignment(primary),
        "four_hour_alignment": _timeframe_alignment(analyses["4h"]),
        "daily_alignment": _timeframe_alignment(analyses["1d"]),
        "one_hour_market_regime": primary.market_regime.market_regime.value,
        "four_hour_market_regime": analyses["4h"].market_regime.market_regime.value,
        "daily_market_regime": analyses["1d"].market_regime.market_regime.value,
        "one_hour_setup": primary.setup.setup.value,
        "four_hour_setup": analyses["4h"].setup.setup.value,
        "daily_setup": analyses["1d"].setup.setup.value,
        "daily_price": float(daily_latest["close"]),
        "daily_ema20": float(daily_latest["ema_20"]),
        "daily_ema200": float(daily_latest["ema_200"]),
        "daily_rsi": float(daily_latest["rsi_14"]),
        "daily_macd": daily_macd,
        "four_hour_price": float(four_hour_latest["close"]),
        "four_hour_ema20": float(four_hour_latest["ema_20"]),
        "four_hour_macd": four_hour_macd,
        "four_hour_macd_improving": watch_long_fields["four_hour_macd_improving"],
        "price_above_1h_ema20": watch_long_fields["price_above_1h_ema20"],
        "price_above_1h_ema50": watch_long_fields["price_above_1h_ema50"],
        "rsi_above_45": watch_long_fields["rsi_above_45"],
        "near_support": watch_long_fields["near_support"],
        "broke_resistance": watch_long_fields["broke_resistance"],
        "daily_trend_not_strongly_bearish": watch_long_fields["daily_trend_not_strongly_bearish"],
        "watch_long_trigger_eligible": watch_long_fields["watch_long_trigger_eligible"],
        "watch_long_blocked_by": "|".join(watch_long_fields["watch_long_blocked_by"]),
        "price_plus_1d": None,
        "price_plus_3d": None,
        "price_plus_7d": None,
        "max_favorable_move_3d": None,
        "max_adverse_move_3d": None,
        "avoid_classification": "PENDING",
    }


def _timeframe_alignment(analysis: Any) -> str:
    setup = analysis.setup.setup.value
    regime = analysis.market_regime.market_regime.value
    if setup in {"BREAKOUT", "TREND_FOLLOWING"} and regime != "BEAR":
        return "BULLISH"
    if setup == "BEAR_TREND" or regime == "BEAR":
        return "BEARISH"
    if setup == "BOTTOMING":
        return "REVERSAL_FORMING"
    if setup == "RANGE_BOUND":
        return "RANGE_BOUND"
    if setup == "PULLBACK":
        return "PULLBACK"
    return "MIXED"


def _macd_histogram_improving(frame: pd.DataFrame) -> bool:
    if len(frame) < 2 or "macd_histogram" not in frame.columns:
        return False
    latest = _optional_float(frame.iloc[-1]["macd_histogram"])
    previous = _optional_float(frame.iloc[-2]["macd_histogram"])
    return latest is not None and previous is not None and latest > previous


def _bb_width(row: pd.Series) -> float | None:
    upper = _optional_float(row.get("bb_upper"))
    lower = _optional_float(row.get("bb_lower"))
    middle = _optional_float(row.get("bb_middle"))
    if upper is None or lower is None or middle is None or middle <= 0:
        return None
    return (upper - lower) / middle


def _watch_long_fields(
    *,
    price: float,
    ema20: float,
    ema50: float,
    rsi: float,
    support: float,
    resistance: float,
    setup: str,
    market_regime: str,
    daily_setup: str,
    daily_market_regime: str,
    four_hour_macd: str,
    four_hour_macd_improving: bool,
) -> dict[str, Any]:
    price_above_ema20 = price > ema20
    price_above_ema50 = price > ema50
    near_support = support > 0 and abs(price - support) / price <= 0.03
    broke_resistance = resistance > 0 and price > resistance
    daily_not_bearish = (
        market_regime != "BEAR"
        and daily_market_regime != "BEAR"
        and daily_setup != "BEAR_TREND"
    )
    criteria = {
        "setup_is_trend_or_range": setup in {"TREND_FOLLOWING", "RANGE_BOUND"},
        "price_above_1h_ema20": price_above_ema20,
        "price_above_1h_ema50": price_above_ema50,
        "four_hour_macd_improving_or_bullish": four_hour_macd == "bullish" or four_hour_macd_improving,
        "rsi_above_45": rsi > 45,
        "near_support_or_broke_resistance": near_support or broke_resistance,
        "daily_trend_not_strongly_bearish": daily_not_bearish,
    }
    return {
        "price_above_1h_ema20": price_above_ema20,
        "price_above_1h_ema50": price_above_ema50,
        "four_hour_macd_improving": four_hour_macd_improving,
        "rsi_above_45": rsi > 45,
        "near_support": near_support,
        "broke_resistance": broke_resistance,
        "daily_trend_not_strongly_bearish": daily_not_bearish,
        "watch_long_trigger_eligible": all(criteria.values()),
        "watch_long_blocked_by": [name for name, passed in criteria.items() if not passed],
    }


def _fetch_shadow_frames(
    config: ShadowTradingConfig,
    provider: CoinbaseCandleProvider,
    current_time: datetime,
    health: ShadowHealth,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for timeframe in DEFAULT_TIMEFRAMES:
        start = time.monotonic()
        try:
            frame = provider.fetch_ohlcv(config.product_id, timeframe, config.history_limit, current_time)
            health.api_requests += 1
        except Exception:
            health.api_requests += 1
            health.api_failures += 1
            health.missing_data_events += 1
            raise
        latency = time.monotonic() - start
        if latency > config.latency_issue_threshold_seconds:
            health.api_latency_issues += 1
        if len(frame) < 200:
            health.missing_data_events += 1
            raise ShadowTradingError(f"Missing sufficient Coinbase {timeframe} candles.")
        frames[timeframe] = frame
    _assert_no_order_endpoint_calls(provider.request_paths)
    return frames


def _fetch_execution_cost(
    config: ShadowTradingConfig,
    opener: JsonOpener,
    timestamp: datetime,
    health: ShadowHealth,
) -> dict[str, float | bool | None]:
    start = time.monotonic()
    audit_config = CoinbaseExecutionAuditConfig(
        product_id=config.product_id,
        fee_rate=config.fee_rate,
        intended_order_size_usd=config.intended_order_size_usd,
        order_sizes_usd=(config.intended_order_size_usd,),
        base_url=config.base_url,
        timeout_seconds=config.timeout_seconds,
        order_book_limit=config.order_book_limit,
    )
    try:
        payload = fetch_public_product_book(audit_config, opener=opener)
        health.api_requests += 1
    except Exception:
        health.api_requests += 1
        health.api_failures += 1
        raise
    latency = time.monotonic() - start
    if latency > config.latency_issue_threshold_seconds:
        health.api_latency_issues += 1
    sample = build_orderbook_sample(payload, audit_config, _isoformat(timestamp))
    key = f"order_{int(config.intended_order_size_usd)}_usd"
    estimate = sample["market_order_estimates"][key]
    half_spread = float(sample["half_spread_pct"])
    estimated_slippage = float(estimate.get("estimated_slippage_pct") or 0.0)
    return {
        "all_in_cost_per_side": estimate.get("all_in_cost_per_side"),
        "price_slippage_pct": half_spread + estimated_slippage,
        "depth_supported": estimate.get("depth_supported"),
    }


def _write_shadow_outputs(
    config: ShadowTradingConfig,
    portfolio: ShadowPortfolio,
    equity_rows: list[dict[str, Any]],
    signal_rows: list[dict[str, Any]],
    health: ShadowHealth,
    signal_count: int,
    timestamp: datetime,
) -> None:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    signal_rows = _enrich_signal_rows_with_forward_evaluations(signal_rows)
    _write_csv(output_dir / "shadow_trades.csv", SHADOW_TRADE_COLUMNS, portfolio.closed_trades)
    _write_csv(output_dir / "shadow_equity_curve.csv", EQUITY_COLUMNS, equity_rows)
    _write_csv(output_dir / "shadow_signals.csv", SHADOW_SIGNAL_COLUMNS, signal_rows)
    _write_csv(output_dir / "shadow_signals_v2.csv", SHADOW_SIGNAL_COLUMNS, signal_rows)
    quality_report = _signal_quality_report(signal_rows)
    _write_json(
        output_dir / "signal_journal_v2.json",
        {
            "phase": "1.22",
            "mode": "SHADOW_SIGNAL_DECISION_SNAPSHOT",
            "live_trading_enabled": False,
            "order_endpoint_calls_allowed": False,
            "signals": signal_rows,
            "quality_report": quality_report,
        },
    )
    _write_json(output_dir / "signal_journal_quality_report.json", quality_report)
    _write_enriched_false_avoid_analysis(output_dir, signal_rows)
    health_payload = _health_payload(config, health, signal_count)
    _write_json(output_dir / "shadow_system_health.json", health_payload)
    daily_report = _daily_report(config, portfolio, equity_rows, health, signal_count, timestamp)
    _write_json(output_dir / f"shadow_daily_report_{timestamp.strftime('%Y%m%d')}.json", daily_report)


def _daily_report(
    config: ShadowTradingConfig,
    portfolio: ShadowPortfolio,
    equity_rows: list[dict[str, Any]],
    health: ShadowHealth,
    signal_count: int,
    timestamp: datetime,
) -> dict[str, Any]:
    date = timestamp.date().isoformat()
    trades_today = [
        trade for trade in portfolio.closed_trades
        if str(trade.get("exit_timestamp", "")).startswith(date)
    ]
    metrics = _trade_metrics(portfolio.closed_trades)
    daily_pnl = sum(float(trade["net_pnl"]) for trade in trades_today)
    current_equity = equity_rows[-1]["current_shadow_equity"] if equity_rows else config.initial_shadow_capital
    return {
        "date": date,
        "current_shadow_equity": current_equity,
        "open_position": portfolio.open_position is not None,
        "trades_today": len(trades_today),
        "cumulative_trades": len(portfolio.closed_trades),
        "daily_pnl": daily_pnl,
        "cumulative_pnl": current_equity - config.initial_shadow_capital,
        "win_rate": metrics["win_rate"],
        "profit_factor": metrics["profit_factor"],
        "max_drawdown": _max_drawdown(equity_rows),
        "average_trade_pnl": metrics["average_net_pnl_per_trade"],
        "average_fee_cost": _mean([float(trade["fee_estimate"]) for trade in portfolio.closed_trades]) or 0.0,
        "average_slippage_cost": _mean([float(trade["slippage_estimate"]) for trade in portfolio.closed_trades]) or 0.0,
        "rejected_signals": len(portfolio.rejected_signals),
        "system_errors": health.system_errors,
        "missing_data_events": health.missing_data_events,
        "api_latency_issues": health.api_latency_issues,
        "signal_count": signal_count,
    }


def _shadow_summary(
    config: ShadowTradingConfig,
    portfolio: ShadowPortfolio,
    equity_rows: list[dict[str, Any]],
    health: ShadowHealth,
    signal_count: int,
    final_price: float,
    finished_at: datetime,
) -> dict[str, Any]:
    current_equity = portfolio.current_equity(final_price)
    metrics = _trade_metrics(portfolio.closed_trades)
    completed_days = len({str(row["timestamp"])[:10] for row in equity_rows})
    api_error_rate = health.api_failures / health.api_requests if health.api_requests else 0.0
    data_quality_score = max(0.0, 100.0 - ((health.missing_data_events / max(signal_count, 1)) * 100.0))
    uptime = 0.0 if health.emergency_stop_triggered else 100.0
    average_all_in = _mean(
        [
            value
            for trade in portfolio.closed_trades
            for value in (float(trade["entry_all_in_cost_per_side"]), float(trade["exit_all_in_cost_per_side"]))
        ]
    ) or 0.0
    live_criteria = {
        "shadow_trading_completed_30_days": completed_days >= 30,
        "live_trading_enabled_remained_false": config.live_trading_enabled is False,
        "no_unauthorized_order_endpoint_calls": health.unauthorized_order_endpoint_calls == 0,
        "positive_total_return": current_equity > config.initial_shadow_capital,
        "sharpe_ratio_gte_0_8": metrics["sharpe_ratio"] >= 0.8,
        "profit_factor_gte_1_3": metrics["profit_factor"] >= 1.3,
        "max_drawdown_lte_15": _max_drawdown(equity_rows) <= 15,
        "api_error_rate_lte_1_pct": api_error_rate <= 0.01,
        "data_quality_score_gte_95": data_quality_score >= 95,
        "all_trades_reconcile": True,
    }
    final_verdict = (
        "READY_FOR_TINY_CAPITAL_PILOT"
        if all(live_criteria.values())
        else "NEEDS_MORE_SHADOW_TESTING"
        if not health.emergency_stop_triggered
        else "NOT_READY_FOR_LIVE_TRADING"
    )
    return {
        "phase": "1.20",
        "mode": "COINBASE_SHADOW_TRADING",
        "product_id": normalize_coinbase_product_id(config.product_id),
        "finished_at": _isoformat(finished_at),
        "live_trading_enabled": False,
        "order_endpoint_calls_allowed": False,
        "total_return_pct": _pct_ratio((current_equity / config.initial_shadow_capital) - 1),
        "sharpe_ratio": metrics["sharpe_ratio"],
        "max_drawdown_pct": _max_drawdown(equity_rows),
        "profit_factor": metrics["profit_factor"],
        "win_rate": metrics["win_rate"],
        "total_trades": len(portfolio.closed_trades),
        "average_net_pnl_per_trade": metrics["average_net_pnl_per_trade"],
        "average_holding_time": metrics["average_holding_hours"],
        "average_all_in_cost_per_side": average_all_in,
        "signal_count": signal_count,
        "rejected_signal_count": len(portfolio.rejected_signals),
        "data_quality_score": data_quality_score,
        "system_uptime_pct": uptime,
        "api_error_rate": api_error_rate,
        "safety_controls": _safety_controls(config),
        "live_readiness_criteria": live_criteria,
        "final_verdict": final_verdict,
    }


def _equity_row(
    portfolio: ShadowPortfolio,
    timestamp: str,
    price: float,
    initial_capital: float,
) -> dict[str, Any]:
    position_value = 0.0 if portfolio.open_position is None else portfolio.open_position.position_size * price
    current_equity = portfolio.cash + position_value
    return {
        "timestamp": timestamp,
        "current_shadow_equity": current_equity,
        "cash": portfolio.cash,
        "position_value": position_value,
        "open_position": portfolio.open_position is not None,
        "price": price,
        "cumulative_pnl": current_equity - initial_capital,
    }


def _shadow_signal_row(
    signal: dict[str, Any],
    action: str,
    rejection_reasons: list[str],
    execution_cost: dict[str, float | bool | None],
) -> dict[str, Any]:
    row = {column: signal.get(column) for column in SHADOW_SIGNAL_COLUMNS}
    row["action"] = action
    row["rejected"] = bool(rejection_reasons)
    row["rejection_reasons"] = "|".join(rejection_reasons)
    categories = _rejection_categories(signal, rejection_reasons)
    row["rejection_categories"] = "|".join(categories)
    row["blocking_rule"] = rejection_reasons[0] if rejection_reasons else ""
    row["blocking_timeframe"] = _blocking_timeframe(row["blocking_rule"])
    row["avoid_long_reason"] = signal.get("decision_reason") if signal.get("final_decision") == "AVOID LONG" else ""
    row["all_in_cost_per_side"] = execution_cost.get("all_in_cost_per_side")
    row["price_slippage_pct"] = execution_cost.get("price_slippage_pct")
    row["depth_supported"] = execution_cost.get("depth_supported")
    return row


def _target_signal_count_remaining(config: ShadowTradingConfig, signal_rows: list[dict[str, Any]]) -> bool:
    return config.target_signal_count is None or len(signal_rows) < config.target_signal_count


def _collection_shadow_config(config: ShadowSignalCollectionConfig) -> ShadowTradingConfig:
    return ShadowTradingConfig(
        product_id=config.product_id,
        intended_order_size_usd=config.intended_order_size_usd,
        fee_rate=config.fee_rate,
        max_all_in_cost_per_side=config.max_all_in_cost_per_side,
        cycle_interval_seconds=config.interval_seconds,
        history_limit=config.history_limit,
        output_dir=config.output_dir,
        base_url=config.base_url,
        timeout_seconds=config.timeout_seconds,
        order_book_limit=config.order_book_limit,
        live_trading_enabled=False,
        order_endpoint_calls_allowed=False,
    )


def _snapshot_rejection_reasons(
    signal: dict[str, Any],
    execution_cost: dict[str, float | bool | None],
    config: ShadowTradingConfig,
) -> list[str]:
    portfolio = ShadowPortfolio(
        initial_capital=config.initial_shadow_capital,
        intended_order_size_usd=config.intended_order_size_usd,
        max_all_in_cost_per_side=config.max_all_in_cost_per_side,
        fee_rate=config.fee_rate,
    )
    return portfolio._entry_rejection_reasons(signal, execution_cost)


def _load_existing_signal_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    rows = payload.get("signals", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return []
    return _dedupe_signal_rows([row for row in rows if isinstance(row, dict)])


def _upsert_signal_row(rows: list[dict[str, Any]], row: dict[str, Any]) -> list[dict[str, Any]]:
    return _dedupe_signal_rows([*rows, row])


def _dedupe_signal_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        timestamp = str(row.get("timestamp") or "")
        symbol = str(row.get("symbol") or row.get("product_id") or "")
        final_decision = str(row.get("final_decision") or "")
        if not timestamp:
            continue
        deduped[(timestamp, symbol, final_decision)] = row
    return sorted(deduped.values(), key=lambda item: str(item.get("timestamp") or ""))


def _write_signal_collection_outputs(
    output_dir: Path,
    signal_rows: list[dict[str, Any]],
    target_signals: int = 50,
) -> None:
    signal_rows = _enrich_signal_rows_with_forward_evaluations(signal_rows)
    quality_report = _signal_quality_report(signal_rows)
    _write_json(
        output_dir / "signal_journal_v2.json",
        {
            "phase": "1.23",
            "mode": "ENRICHED_SHADOW_SIGNAL_COLLECTION",
            "live_trading_enabled": False,
            "order_endpoint_calls_allowed": False,
            "signals": signal_rows,
            "quality_report": quality_report,
        },
    )
    _write_json(output_dir / "signal_journal_quality_report.json", quality_report)
    _write_enriched_false_avoid_analysis(output_dir, signal_rows, target_signals=target_signals)


def _write_collection_health_outputs(
    output_dir: Path,
    config: ShadowTradingConfig,
    health: ShadowHealth,
    signal_rows: list[dict[str, Any]],
    timestamp: datetime,
) -> None:
    _write_json(output_dir / "shadow_system_health.json", _health_payload(config, health, len(signal_rows)))
    _write_json(
        output_dir / f"shadow_daily_report_{timestamp.strftime('%Y%m%d')}.json",
        {
            "date": timestamp.date().isoformat(),
            "current_shadow_equity": config.initial_shadow_capital,
            "open_position": False,
            "trades_today": 0,
            "cumulative_trades": 0,
            "daily_pnl": 0.0,
            "cumulative_pnl": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "average_trade_pnl": 0.0,
            "average_fee_cost": 0.0,
            "average_slippage_cost": 0.0,
            "rejected_signals": sum(1 for row in signal_rows if row.get("rejected") in {True, "True", "true"}),
            "system_errors": health.system_errors,
            "missing_data_events": health.missing_data_events,
            "api_latency_issues": health.api_latency_issues,
            "signal_count": len(signal_rows),
        },
    )


def _append_signal_csv_row(path: Path, row: dict[str, Any]) -> None:
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHADOW_SIGNAL_COLUMNS, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _reset_signal_collection_outputs(output_dir: Path) -> None:
    for filename in (
        "signal_journal_v2.json",
        "shadow_signals_v2.csv",
        "shadow_signals.csv",
        "signal_journal_quality_report.json",
        "enriched_false_avoid_analysis.json",
    ):
        path = output_dir / filename
        if path.exists():
            path.unlink()


def _enrich_signal_rows_with_forward_evaluations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_rows = _dedupe_signal_rows(rows)
    price_points = [
        {
            "timestamp": pd.Timestamp(row["timestamp"]),
            "price": _optional_float(row.get("price")),
        }
        for row in sorted_rows
        if row.get("timestamp") and _optional_float(row.get("price")) is not None
    ]
    enriched = []
    for row in sorted_rows:
        updated = row.copy()
        if row.get("final_decision") == "AVOID LONG":
            evaluation = _evaluate_avoid_forward_path(row, price_points)
            updated.update(evaluation)
        enriched.append(updated)
    return enriched


def _evaluate_avoid_forward_path(
    row: dict[str, Any],
    price_points: list[dict[str, Any]],
) -> dict[str, Any]:
    timestamp = pd.Timestamp(row["timestamp"])
    entry_price = _optional_float(row.get("price"))
    if entry_price is None or entry_price <= 0:
        return {"avoid_classification": "INCONCLUSIVE"}
    future = [
        point
        for point in price_points
        if point["timestamp"] > timestamp and point["timestamp"] <= timestamp + pd.Timedelta(days=3)
    ]
    price_plus_1d = _future_price_at_or_after(price_points, timestamp + pd.Timedelta(days=1))
    price_plus_3d = _future_price_at_or_after(price_points, timestamp + pd.Timedelta(days=3))
    price_plus_7d = _future_price_at_or_after(price_points, timestamp + pd.Timedelta(days=7))
    changes_3d = [
        _pct_ratio((float(point["price"]) / entry_price) - 1)
        for point in future
        if point.get("price") is not None
    ]
    max_favorable = max(changes_3d) if changes_3d else None
    max_adverse = min(changes_3d) if changes_3d else None
    classification = _classify_avoid_from_path(future, timestamp, entry_price)
    return {
        "price_plus_1d": price_plus_1d,
        "price_plus_3d": price_plus_3d,
        "price_plus_7d": price_plus_7d,
        "max_favorable_move_3d": max_favorable,
        "max_adverse_move_3d": max_adverse,
        "avoid_classification": classification,
    }


def _classify_avoid_from_path(
    future: list[dict[str, Any]],
    timestamp: pd.Timestamp,
    entry_price: float,
) -> str:
    if not future:
        return "INCONCLUSIVE"
    lowest_before_breakout = 0.0
    for point in future:
        price = _optional_float(point.get("price"))
        if price is None:
            continue
        change = _pct_ratio((price / entry_price) - 1)
        lowest_before_breakout = min(lowest_before_breakout, change)
        if change > 2.0 and lowest_before_breakout > -1.5:
            return "FALSE_AVOID"
        if change < -2.0:
            return "CORRECT_AVOID"
    latest_timestamp = max(point["timestamp"] for point in future)
    has_full_3d_window = latest_timestamp >= timestamp + pd.Timedelta(days=3)
    max_up = max(
        [
            _pct_ratio((float(point["price"]) / entry_price) - 1)
            for point in future
            if point.get("price") is not None
        ],
        default=None,
    )
    if has_full_3d_window and max_up is not None and max_up <= 2.0:
        return "CORRECT_AVOID"
    return "INCONCLUSIVE"


def _future_price_at_or_after(price_points: list[dict[str, Any]], target: pd.Timestamp) -> float | None:
    for point in price_points:
        if point["timestamp"] >= target and point.get("price") is not None:
            return float(point["price"])
    return None


def _write_enriched_false_avoid_analysis(
    output_dir: Path,
    signal_rows: list[dict[str, Any]],
    target_signals: int = 50,
) -> None:
    report = _enriched_false_avoid_analysis(signal_rows, target_signals=target_signals)
    _write_json(output_dir / "enriched_false_avoid_analysis.json", report)


def _enriched_false_avoid_analysis(
    signal_rows: list[dict[str, Any]],
    target_signals: int = 50,
) -> dict[str, Any]:
    enriched_rows = _enrich_signal_rows_with_forward_evaluations(signal_rows)
    avoids = [row for row in enriched_rows if row.get("final_decision") == "AVOID LONG"]
    false_avoids = [row for row in avoids if row.get("avoid_classification") == "FALSE_AVOID"]
    correct_avoids = [row for row in avoids if row.get("avoid_classification") == "CORRECT_AVOID"]
    inconclusive = [row for row in avoids if row.get("avoid_classification") == "INCONCLUSIVE"]
    classified_count = len(false_avoids) + len(correct_avoids)
    reason_counts: dict[str, int] = {}
    for row in false_avoids:
        for reason in _split_pipe(row.get("rejection_reasons")):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    average_1d = _average_price_change(avoids, "price_plus_1d")
    average_3d = _average_price_change(avoids, "price_plus_3d")
    max_missed_gain = _max_optional([_optional_float(row.get("max_favorable_move_3d")) for row in avoids])
    return {
        "phase": "1.23",
        "mode": "ENRICHED_FALSE_AVOID_ANALYSIS",
        "live_trading_enabled": False,
        "order_endpoint_calls_allowed": False,
        "minimum_target_signals": 50,
        "preferred_target_signals": 100,
        "active_target_signals": target_signals,
        "target_reached": len(enriched_rows) >= target_signals,
        "total_signals": len(enriched_rows),
        "total_avoid_long_signals": len(avoids),
        "false_avoid_count": len(false_avoids),
        "correct_avoid_count": len(correct_avoids),
        "inconclusive_count": len(inconclusive),
        "false_avoid_rate": _pct_ratio(len(false_avoids) / classified_count) if classified_count else None,
        "most_common_false_avoid_rejection_reasons": dict(
            sorted(reason_counts.items(), key=lambda item: item[1], reverse=True)
        ),
        "average_missed_gain_1d": average_1d,
        "average_missed_gain_3d": average_3d,
        "max_missed_gain": max_missed_gain,
        "whether_watch_long_should_be_tested_again": _watch_long_retest_recommendation(
            total_signals=len(enriched_rows),
            false_avoid_count=len(false_avoids),
            classified_count=classified_count,
        ),
        "analysis_note": (
            "Signals without a full 3-day future window remain INCONCLUSIVE unless a +2% rally or -2% drop threshold "
            "is hit earlier."
        ),
    }


def _average_price_change(rows: list[dict[str, Any]], future_price_key: str) -> float | None:
    changes = []
    for row in rows:
        entry = _optional_float(row.get("price"))
        future = _optional_float(row.get(future_price_key))
        if entry is not None and future is not None and entry > 0:
            changes.append(_pct_ratio((future / entry) - 1))
    return _mean(changes)


def _watch_long_retest_recommendation(
    *,
    total_signals: int,
    false_avoid_count: int,
    classified_count: int,
) -> str:
    if total_signals < 50:
        return "WAIT_FOR_50_ENRICHED_SIGNALS"
    if classified_count == 0:
        return "WAIT_FOR_FORWARD_WINDOWS"
    false_rate = false_avoid_count / classified_count
    return "YES" if false_rate >= 0.30 else "NO"


def _split_pipe(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    return [part for part in str(value).split("|") if part]


def _max_optional(values: list[float | None]) -> float | None:
    clean_values = [value for value in values if value is not None]
    return max(clean_values) if clean_values else None


def _rejection_categories(signal: dict[str, Any], rejection_reasons: list[str]) -> list[str]:
    categories: list[str] = []
    setup = str(signal.get("setup") or "")
    if setup:
        categories.append(setup)
    categories.extend(rejection_reasons)
    volume_ratio = _optional_float(signal.get("volume_ratio"))
    if volume_ratio is not None and volume_ratio < 1.0:
        categories.append("low volume")
    alignment = str(signal.get("multi_timeframe_alignment") or signal.get("alignment") or "")
    if alignment and alignment not in {"BULLISH_ALIGNMENT", "PULLBACK_IN_UPTREND"}:
        categories.append("weak multi-timeframe alignment")
    price = _optional_float(signal.get("price"))
    ema200 = _optional_float(signal.get("ema200"))
    if price is not None and ema200 is not None and price < ema200:
        categories.append("below EMA200")
    macd = str(signal.get("macd") or "")
    if macd == "bearish":
        categories.append("MACD bearish")
    rsi = _optional_float(signal.get("rsi"))
    if rsi is not None and 45 <= rsi <= 55:
        categories.append("RSI neutral")
    rr_ratio = _optional_float(signal.get("rr_ratio"))
    if rr_ratio is not None and rr_ratio < 2.0:
        categories.append("risk/reward too low")
    return list(dict.fromkeys(categories))


def _blocking_timeframe(blocking_rule: Any) -> str:
    mapping = {
        "not_buy_decision": "decision",
        "bearish_alignment": "multi_timeframe",
        "low_rr_ratio": "trade_quality",
        "low_volume_ratio": "1h",
        "bear_market_regime": "1h",
        "daily_bear_trend": "1d",
        "below_4h_ema20": "4h",
        "high_execution_cost": "execution",
        "insufficient_order_book_depth": "execution",
    }
    return mapping.get(str(blocking_rule), "")


def _validate_signal_quality_or_raise(row: dict[str, Any]) -> None:
    missing = _signal_missing_fields(row)
    if missing:
        missing_fields = ", ".join(missing)
        raise ShadowTradingError(f"Signal journal v2 row is incomplete: {missing_fields}.")


def _signal_quality_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing_field_counts: dict[str, int] = {}
    incomplete_rows = 0
    avoid_rows_missing_reasons = 0
    row_details = []
    for index, row in enumerate(rows):
        missing = _signal_missing_fields(row)
        if missing:
            incomplete_rows += 1
            row_details.append(
                {
                    "row_index": index,
                    "timestamp": row.get("timestamp"),
                    "symbol": row.get("symbol"),
                    "missing_fields": missing,
                }
            )
            for field in missing:
                missing_field_counts[field] = missing_field_counts.get(field, 0) + 1
        if row.get("final_decision") == "AVOID LONG" and _is_missing(row.get("rejection_reasons")):
            avoid_rows_missing_reasons += 1
    total = len(rows)
    complete_rows = total - incomplete_rows
    score = 100.0 if total == 0 else round((complete_rows / total) * 100.0, 2)
    return {
        "phase": "1.22",
        "total_signals": total,
        "complete_signal_rows": complete_rows,
        "incomplete_signal_rows": incomplete_rows,
        "missing_field_counts": missing_field_counts,
        "signal_journal_quality_score": score,
        "avoid_long_rows_missing_rejection_reasons": avoid_rows_missing_reasons,
        "success_criteria": {
            "quality_score_gte_95": score >= 95.0,
            "incomplete_signal_rows_zero": incomplete_rows == 0,
            "every_avoid_long_has_rejection_reasons": avoid_rows_missing_reasons == 0,
        },
        "incomplete_signal_rows_explanation": (
            "No incomplete rows detected."
            if incomplete_rows == 0
            else "Rows listed in incomplete_row_details were rejected by required-field validation."
        ),
        "incomplete_row_details": row_details,
    }


def _signal_missing_fields(row: dict[str, Any]) -> list[str]:
    missing = [
        field
        for field in SIGNAL_QUALITY_REQUIRED_FIELDS
        if field not in row or (field != "rejection_reasons" and _is_missing(row.get(field)))
    ]
    if row.get("final_decision") == "AVOID LONG" and _is_missing(row.get("rejection_reasons")):
        missing.append("rejection_reasons")
    return list(dict.fromkeys(missing))


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value == "":
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _health_payload(config: ShadowTradingConfig, health: ShadowHealth, signal_count: int) -> dict[str, Any]:
    return {
        "live_trading_enabled": False,
        "order_endpoint_calls_allowed": False,
        "signal_count": signal_count,
        "system_errors": health.system_errors,
        "missing_data_events": health.missing_data_events,
        "api_latency_issues": health.api_latency_issues,
        "api_requests": health.api_requests,
        "api_failures": health.api_failures,
        "api_error_rate": health.api_failures / health.api_requests if health.api_requests else 0.0,
        "unauthorized_order_endpoint_calls": health.unauthorized_order_endpoint_calls,
        "emergency_stop_triggered": health.emergency_stop_triggered,
        "emergency_stop_reason": health.emergency_stop_reason,
        "safety_controls": _safety_controls(config),
    }


def _safety_controls(config: ShadowTradingConfig) -> dict[str, Any]:
    return {
        "live_trading_enabled": False,
        "order_endpoint_calls_allowed": False,
        "max_position_count": config.max_position_count,
        "max_shadow_position_size_usd": config.intended_order_size_usd,
        "emergency_stop_if_data_missing": config.emergency_stop_if_data_missing,
        "emergency_stop_if_api_errors_exceed_5_percent": config.emergency_stop_if_api_errors_exceed_5_percent,
    }


def _trade_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "average_net_pnl_per_trade": 0.0,
            "average_holding_hours": 0.0,
            "sharpe_ratio": 0.0,
        }
    pnls = [float(trade["net_pnl"]) for trade in trades]
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    holding_hours = [
        (pd.Timestamp(trade["exit_timestamp"]) - pd.Timestamp(trade["signal_timestamp"])).total_seconds() / 3600
        for trade in trades
        if trade.get("exit_timestamp")
    ]
    return {
        "win_rate": round((len(wins) / len(trades)) * 100, 2),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss else (float("inf") if gross_profit else 0.0),
        "average_net_pnl_per_trade": sum(pnls) / len(pnls),
        "average_holding_hours": sum(holding_hours) / len(holding_hours) if holding_hours else 0.0,
        "sharpe_ratio": 0.0,
    }


def _max_drawdown(equity_rows: list[dict[str, Any]]) -> float:
    if not equity_rows:
        return 0.0
    peak = float(equity_rows[0]["current_shadow_equity"])
    max_drawdown = 0.0
    for row in equity_rows:
        equity = float(row["current_shadow_equity"])
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
    return _pct_ratio(max_drawdown)


def _emergency_stop_required(config: ShadowTradingConfig, health: ShadowHealth) -> bool:
    if config.emergency_stop_if_data_missing and health.missing_data_events > 0:
        return True
    if config.emergency_stop_if_api_errors_exceed_5_percent and health.api_requests > 0:
        return (health.api_failures / health.api_requests) > 0.05
    return False


def _validate_shadow_safety(config: ShadowTradingConfig) -> None:
    if config.live_trading_enabled:
        raise ShadowTradingError("live_trading_enabled must remain false for Phase 1.20.")
    if config.order_endpoint_calls_allowed:
        raise ShadowTradingError("order_endpoint_calls_allowed must remain false for Phase 1.20.")
    if config.max_position_count != 1:
        raise ShadowTradingError("max_position_count must be 1 for Phase 1.20.")
    if config.intended_order_size_usd <= 0 or config.intended_order_size_usd > 2500:
        raise ShadowTradingError("max_shadow_position_size_usd must be no more than 2500.")


def _validate_collection_safety(config: ShadowSignalCollectionConfig) -> None:
    if config.live_trading_enabled:
        raise ShadowTradingError("live_trading_enabled must remain false for Phase 1.23.")
    if config.order_endpoint_calls_allowed:
        raise ShadowTradingError("order_endpoint_calls_allowed must remain false for Phase 1.23.")
    if config.intended_order_size_usd <= 0 or config.intended_order_size_usd > 2500:
        raise ShadowTradingError("intended_order_size_usd must be no more than 2500.")


def _assert_no_order_endpoint_calls(paths: list[str]) -> None:
    if any("/orders" in path for path in paths):
        raise ShadowTradingError("Unauthorized Coinbase order endpoint call detected.")


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _pct_ratio(value: float) -> float:
    return round(float(value) * 100.0, 4)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _isoformat(value: datetime) -> str:
    return _as_utc(value).isoformat()
