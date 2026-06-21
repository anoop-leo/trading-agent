"""BTC long-term cycle phase overlay for the Investor Agent."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from trading_agent.indicators import calculate_ema, calculate_rsi


CYCLE_PHASE_ANALOGS = {
    "DEEP_BEAR": ["2015 bear-market lows", "2018 capitulation zone"],
    "LATE_BEAR": ["2019 recovery base", "2022 post-capitulation recovery"],
    "EARLY_BULL": ["2016 early bull recovery", "2020 post-halving recovery"],
    "MID_BULL": ["2017 trend expansion", "2021 mid-cycle trend"],
    "LATE_BULL": ["2017 late-cycle distribution", "2021 late-cycle risk zone"],
    "EUPHORIA": ["2013 blow-off top", "2017 cycle peak"],
    "UNKNOWN": [],
}

ACCUMULATION_BIAS = {
    "DEEP_BEAR": "EXTREME",
    "LATE_BEAR": "HIGH",
    "EARLY_BULL": "HIGH",
    "MID_BULL": "MEDIUM",
    "LATE_BULL": "LOW",
    "EUPHORIA": "LOW",
    "UNKNOWN": "MEDIUM",
}

PROFIT_TAKING_BIAS = {
    "DEEP_BEAR": "LOW",
    "LATE_BEAR": "LOW",
    "EARLY_BULL": "LOW",
    "MID_BULL": "MEDIUM",
    "LATE_BULL": "HIGH",
    "EUPHORIA": "HIGH",
    "UNKNOWN": "LOW",
}

PHASE_SCORE = {
    "DEEP_BEAR": 95,
    "LATE_BEAR": 80,
    "EARLY_BULL": 65,
    "MID_BULL": 55,
    "LATE_BULL": 30,
    "EUPHORIA": 10,
    "UNKNOWN": 50,
}


@dataclass(frozen=True)
class CyclePhaseInput:
    monthly_candles: pd.DataFrame | None = None
    weekly_candles: pd.DataFrame | None = None
    daily_candles: pd.DataFrame | None = None
    price: float | None = None
    ma200: float | None = None
    ma200w: float | None = None
    monthly_ema20: float | None = None
    weekly_rsi: float | None = None
    mvrv: float | None = None
    drawdown_from_cycle_high_pct: float | None = None
    fear_and_greed: float | None = None
    halving_date: date | None = None
    months_since_halving: float | None = None


def classify_btc_cycle_phase(inputs: CyclePhaseInput) -> dict[str, Any]:
    values = _derive_values(inputs)
    missing_fields = _missing_fields(values)
    phase, rationale = _classify(values)
    confidence = _phase_confidence(phase, missing_fields)
    return {
        "agent": "BTC_CYCLE_PHASE",
        "cycle_phase": phase,
        "phase_score": PHASE_SCORE[phase],
        "phase_confidence": confidence,
        "historical_analogs": CYCLE_PHASE_ANALOGS[phase],
        "accumulation_bias": ACCUMULATION_BIAS[phase],
        "profit_taking_bias": PROFIT_TAKING_BIAS[phase],
        "inputs": {
            "price": values["price"],
            "ma200": values["ma200"],
            "ma200w": values["ma200w"],
            "monthly_ema20": values["monthly_ema20"],
            "weekly_rsi": values["weekly_rsi"],
            "mvrv": values["mvrv"],
            "drawdown_from_cycle_high_pct": values["drawdown_from_cycle_high_pct"],
            "fear_and_greed": values["fear_and_greed"],
            "months_since_halving": values["months_since_halving"],
            "missing_fields": missing_fields,
        },
        "rationale": rationale,
    }


def cycle_adjusted_dca_multiplier(base_multiplier: str, cycle_phase: str) -> str:
    if cycle_phase == "EUPHORIA":
        return _cap_dca_multiplier_at_0_25(base_multiplier)
    if cycle_phase == "LATE_BULL":
        return _cap_dca_multiplier_at_0_5(base_multiplier)
    if cycle_phase == "MID_BULL":
        return _cap_dca_multiplier_at_1_0(base_multiplier)
    return base_multiplier


def cycle_adjusted_action(base_action: str, cycle_phase: str) -> str:
    if cycle_phase == "EUPHORIA":
        return "PROFIT_TAKING_REVIEW"
    if cycle_phase == "LATE_BULL":
        return "PREPARE_STAGED_SELL_PLAN"
    if cycle_phase == "MID_BULL":
        return "NORMAL_DCA"
    if cycle_phase in {"DEEP_BEAR", "LATE_BEAR", "EARLY_BULL"}:
        return base_action
    return base_action if base_action else "INSUFFICIENT_DATA"


def _derive_values(inputs: CyclePhaseInput) -> dict[str, float | None]:
    price = inputs.price
    ma200 = inputs.ma200
    ma200w = inputs.ma200w
    monthly_ema20 = inputs.monthly_ema20
    weekly_rsi = inputs.weekly_rsi

    if inputs.daily_candles is not None and len(inputs.daily_candles) >= 200:
        close = inputs.daily_candles["close"].astype(float)
        price = price if price is not None else float(close.iloc[-1])
        ma200 = ma200 if ma200 is not None else float(close.rolling(200, min_periods=200).mean().iloc[-1])

    if inputs.weekly_candles is not None and len(inputs.weekly_candles) >= 20:
        close = inputs.weekly_candles["close"].astype(float)
        weekly_rsi = weekly_rsi if weekly_rsi is not None else float(calculate_rsi(close, 14).iloc[-1])
        if len(close) >= 200:
            ma200w = ma200w if ma200w is not None else float(close.rolling(200, min_periods=200).mean().iloc[-1])

    if inputs.monthly_candles is not None and len(inputs.monthly_candles) >= 20:
        close = inputs.monthly_candles["close"].astype(float)
        monthly_ema20 = monthly_ema20 if monthly_ema20 is not None else float(calculate_ema(close, 20).iloc[-1])

    return {
        "price": price,
        "ma200": ma200,
        "ma200w": ma200w,
        "monthly_ema20": monthly_ema20,
        "weekly_rsi": weekly_rsi,
        "mvrv": inputs.mvrv,
        "drawdown_from_cycle_high_pct": inputs.drawdown_from_cycle_high_pct,
        "fear_and_greed": inputs.fear_and_greed,
        "months_since_halving": inputs.months_since_halving,
    }


def _classify(values: dict[str, float | None]) -> tuple[str, list[str]]:
    mvrv = values["mvrv"]
    drawdown_abs = _drawdown_abs(values["drawdown_from_cycle_high_pct"])
    weekly_rsi = values["weekly_rsi"]
    fear = values["fear_and_greed"]
    price = values["price"]
    ma200 = values["ma200"]
    ma200w = values["ma200w"]
    monthly_ema20 = values["monthly_ema20"]
    distance_from_200d_ma_pct = _distance_pct(price, ma200)

    if mvrv is None:
        return "UNKNOWN", ["MVRV is unavailable, so cycle phase confidence is limited."]

    if (
        mvrv > 4.0
        and weekly_rsi is not None
        and weekly_rsi > 80
        and fear is not None
        and fear > 80
        and distance_from_200d_ma_pct is not None
        and distance_from_200d_ma_pct >= 100
    ):
        return "EUPHORIA", [
            "MVRV is above 4.0.",
            "Weekly RSI is above 80 and sentiment is euphoric.",
            "Price is far above the 200D moving average.",
        ]

    if (
        mvrv < 1.0
        and drawdown_abs is not None
        and drawdown_abs > 70
        and weekly_rsi is not None
        and weekly_rsi < 35
        and price is not None
        and ma200w is not None
        and price < ma200w
    ):
        return "DEEP_BEAR", [
            "MVRV is below 1.0.",
            "Drawdown from cycle high is greater than 70%.",
            "Weekly RSI is below 35 and price is below the 200W moving average.",
        ]

    if (
        1.0 <= mvrv < 1.5
        and drawdown_abs is not None
        and drawdown_abs > 45
        and fear is not None
        and fear < 30
        and weekly_rsi is not None
        and 30 <= weekly_rsi <= 45
    ):
        return "LATE_BEAR", [
            "MVRV is between 1.0 and 1.5.",
            "Drawdown from cycle high is greater than 45%.",
            "Fear & Greed is below 30 and weekly RSI is in the 30-45 range.",
        ]

    if (
        3.0 <= mvrv <= 4.0
        and weekly_rsi is not None
        and weekly_rsi > 70
        and fear is not None
        and fear > 70
    ):
        return "LATE_BULL", [
            "MVRV is in the 3.0-4.0 late-cycle zone.",
            "Weekly RSI and Fear & Greed are elevated.",
        ]

    if (
        2.0 <= mvrv <= 3.0
        and _above(price, ma200)
        and _above(price, monthly_ema20)
        and weekly_rsi is not None
        and 55 <= weekly_rsi <= 70
    ):
        return "MID_BULL", [
            "Price is above the 200D moving average and monthly EMA20.",
            "Weekly RSI is in the 55-70 trend expansion range.",
            "MVRV is between 2.0 and 3.0.",
        ]

    if (
        1.5 <= mvrv <= 2.2
        and _above(price, ma200)
        and weekly_rsi is not None
        and 45 <= weekly_rsi <= 60
    ):
        return "EARLY_BULL", [
            "Price is recovering above the 200D moving average.",
            "Weekly RSI is in the 45-60 recovery range.",
            "MVRV is between 1.5 and 2.2.",
        ]

    return "UNKNOWN", ["Cycle inputs do not match a supported deterministic phase rule."]


def _phase_confidence(phase: str, missing_fields: list[str]) -> str:
    if phase == "UNKNOWN":
        return "LOW"
    if len(missing_fields) == 0:
        return "HIGH"
    if len(missing_fields) <= 2:
        return "MEDIUM"
    return "LOW"


def _missing_fields(values: dict[str, float | None]) -> list[str]:
    required = (
        "price",
        "ma200",
        "monthly_ema20",
        "weekly_rsi",
        "mvrv",
        "drawdown_from_cycle_high_pct",
        "fear_and_greed",
    )
    return [name for name in required if values.get(name) is None]


def _drawdown_abs(value: float | None) -> float | None:
    if value is None:
        return None
    return abs(float(value))


def _distance_pct(price: float | None, baseline: float | None) -> float | None:
    if price is None or baseline is None or baseline <= 0:
        return None
    return ((price - baseline) / baseline) * 100


def _above(left: float | None, right: float | None) -> bool:
    return left is not None and right is not None and left > right


def _cap_dca_multiplier_at_1_0(multiplier: str) -> str:
    if multiplier in {"1.0x to 1.25x normal DCA", "1.0x to 1.5x normal DCA", "1.5x to 2.5x normal DCA"}:
        return "0.5x to 1.0x normal DCA"
    return multiplier


def _cap_dca_multiplier_at_0_5(multiplier: str) -> str:
    if multiplier == "0.0x to 0.25x normal DCA":
        return multiplier
    return "0.0x to 0.5x normal DCA"


def _cap_dca_multiplier_at_0_25(multiplier: str) -> str:
    del multiplier
    return "0.0x to 0.25x normal DCA"
