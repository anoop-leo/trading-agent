"""Trend Holding Engine for Phase 1.13 deterministic backtests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd


class TrendState(str, Enum):
    STRONG_BULL = "STRONG_BULL"
    BULL = "BULL"
    RANGE = "RANGE"
    BEAR = "BEAR"


class MacroRegime(str, Enum):
    STRONG_BULL = "STRONG_BULL"
    BULL = "BULL"
    RANGE = "RANGE"
    BEAR = "BEAR"


@dataclass(frozen=True)
class TrendStateInput:
    daily_close: float
    daily_ema20: float
    daily_ema50: float
    daily_ema200: float
    daily_macd: str
    daily_rsi: float


@dataclass(frozen=True)
class MacroRegimeInput:
    daily_close: float
    daily_ema20: float
    daily_ema50: float
    daily_ema200: float
    daily_macd: str
    daily_rsi: float
    weekly_close: float | None = None
    weekly_ema20: float | None = None
    ema_cross_count: int = 0


@dataclass(frozen=True)
class TrendHoldingConfig:
    tp1_r_multiple: float = 2.0
    tp1_close_fraction: float = 0.50
    tp2_r_multiple: float = 4.0
    tp2_close_fraction: float = 0.25
    runner_fraction: float = 0.25
    runner_trailing_stop_pct: float = 0.20


def determine_trend_state(values: TrendStateInput | dict[str, Any]) -> TrendState:
    """Classify daily trend context from EMA stack, RSI, and MACD."""

    data = _coerce_trend_state_input(values)
    if data is None:
        return TrendState.RANGE
    if data.daily_close < data.daily_ema50 or data.daily_ema20 < data.daily_ema50 or data.daily_macd == "bearish":
        return TrendState.BEAR
    if (
        data.daily_ema20 > data.daily_ema50 > data.daily_ema200
        and data.daily_rsi > 55
        and data.daily_macd == "bullish"
    ):
        return TrendState.STRONG_BULL
    if data.daily_close > data.daily_ema200 and data.daily_ema20 >= data.daily_ema50 and data.daily_rsi >= 50:
        return TrendState.BULL
    return TrendState.RANGE


def determine_macro_regime(values: MacroRegimeInput | dict[str, Any]) -> MacroRegime:
    """Classify macro regime from daily trend context plus weekly confirmation."""

    data = _coerce_macro_regime_input(values)
    if data is None:
        return MacroRegime.RANGE
    weekly_below_ema20 = (
        data.weekly_close is not None
        and data.weekly_ema20 is not None
        and data.weekly_close < data.weekly_ema20
    )
    if (data.daily_ema20 < data.daily_ema50 < data.daily_ema200) or weekly_below_ema20:
        return MacroRegime.BEAR
    if (
        data.daily_ema20 > data.daily_ema50 > data.daily_ema200
        and data.daily_rsi > 55
        and data.daily_macd == "bullish"
        and data.weekly_close is not None
        and data.weekly_ema20 is not None
        and data.weekly_close > data.weekly_ema20
    ):
        return MacroRegime.STRONG_BULL
    if data.ema_cross_count >= 2 or 45 <= data.daily_rsi <= 55:
        return MacroRegime.RANGE
    if data.daily_ema20 > data.daily_ema50 and data.daily_rsi > 50:
        return MacroRegime.BULL
    return MacroRegime.RANGE


class TrendHoldingEngine:
    """Pure rule helper for partial exits and runner management."""

    def __init__(self, config: TrendHoldingConfig | None = None) -> None:
        self.config = config or TrendHoldingConfig()

    def tp1_price(self, entry_price: float, initial_risk: float) -> float | None:
        if initial_risk <= 0:
            return None
        return float(entry_price) + (self.config.tp1_r_multiple * float(initial_risk))

    def tp2_price(self, entry_price: float, initial_risk: float) -> float | None:
        if initial_risk <= 0:
            return None
        return float(entry_price) + (self.config.tp2_r_multiple * float(initial_risk))

    def runner_trailing_stop(self, highest_close: float) -> float | None:
        if highest_close <= 0:
            return None
        return float(highest_close) * (1 - self.config.runner_trailing_stop_pct)

    def runner_exit_reason(self, signal: dict[str, Any], highest_close: float) -> str | None:
        daily_close = _optional_float(signal.get("daily_price"))
        daily_ema50 = _optional_float(signal.get("daily_ema50"))
        if daily_close is not None and daily_ema50 is not None and daily_close < daily_ema50:
            return "RUNNER_DAILY_EMA50_EXIT"
        if str(signal.get("daily_macd") or signal.get("macd") or "").lower() == "bearish":
            return "RUNNER_MACD_EXIT"
        trailing_stop = self.runner_trailing_stop(highest_close)
        price = _optional_float(signal.get("price"))
        if trailing_stop is not None and price is not None and price <= trailing_stop:
            return "RUNNER_TRAILING_STOP"
        return None


def _coerce_trend_state_input(values: TrendStateInput | dict[str, Any]) -> TrendStateInput | None:
    if isinstance(values, TrendStateInput):
        return values
    try:
        return TrendStateInput(
            daily_close=float(values["daily_close"] if "daily_close" in values else values["daily_price"]),
            daily_ema20=float(values["daily_ema20"]),
            daily_ema50=float(values["daily_ema50"]),
            daily_ema200=float(values["daily_ema200"]),
            daily_macd=str(values["daily_macd"]),
            daily_rsi=float(values["daily_rsi"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _coerce_macro_regime_input(values: MacroRegimeInput | dict[str, Any]) -> MacroRegimeInput | None:
    if isinstance(values, MacroRegimeInput):
        return values
    try:
        return MacroRegimeInput(
            daily_close=float(values["daily_close"] if "daily_close" in values else values["daily_price"]),
            daily_ema20=float(values["daily_ema20"]),
            daily_ema50=float(values["daily_ema50"]),
            daily_ema200=float(values["daily_ema200"]),
            daily_macd=str(values["daily_macd"]),
            daily_rsi=float(values["daily_rsi"]),
            weekly_close=_optional_float(values.get("weekly_close", values.get("weekly_price"))),
            weekly_ema20=_optional_float(values.get("weekly_ema20")),
            ema_cross_count=int(values.get("ema_cross_count", 0) or 0),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
