"""Portfolio drawdown and position-sizing controls for Phase 1.15."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import pandas as pd


class RiskState(str, Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    DEFENSIVE = "DEFENSIVE"
    CAPITAL_PRESERVATION = "CAPITAL_PRESERVATION"


@dataclass(frozen=True)
class RiskSnapshot:
    current_equity: float
    peak_equity: float
    current_drawdown_pct: float
    risk_state: str
    allocation_multiplier: float
    runner_enabled: bool
    trend_holding_enabled: bool
    portfolio_stop_active: bool
    portfolio_stop_triggered: bool

    def to_dict(self) -> dict[str, float | str | bool]:
        return asdict(self)


@dataclass(frozen=True)
class PositionSizeResult:
    position_size: float
    notional: float
    risk_amount: float
    allocation_multiplier: float
    volatility_factor: float
    position_size_fraction: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class PortfolioRiskGovernor:
    """Track portfolio drawdown and convert risk limits into position size."""

    def __init__(
        self,
        initial_equity: float = 10000.0,
        risk_per_trade: float = 0.01,
        caution_drawdown_pct: float = 10.0,
        defensive_drawdown_pct: float = 15.0,
        capital_preservation_drawdown_pct: float = 20.0,
        portfolio_stop_drawdown_pct: float = 25.0,
        recovery_drawdown_pct: float = 15.0,
        min_volatility_factor: float = 0.25,
        max_volatility_factor: float = 1.0,
    ) -> None:
        if initial_equity <= 0:
            raise ValueError("initial_equity must be greater than zero.")
        if risk_per_trade <= 0:
            raise ValueError("risk_per_trade must be greater than zero.")
        if not 0 < min_volatility_factor <= max_volatility_factor <= 1.0:
            raise ValueError("volatility factor bounds must be within 0 and 1.")

        self.initial_equity = float(initial_equity)
        self.current_equity = float(initial_equity)
        self.peak_equity = float(initial_equity)
        self.current_drawdown_pct = 0.0
        self.risk_per_trade = float(risk_per_trade)
        self.caution_drawdown_pct = float(caution_drawdown_pct)
        self.defensive_drawdown_pct = float(defensive_drawdown_pct)
        self.capital_preservation_drawdown_pct = float(capital_preservation_drawdown_pct)
        self.portfolio_stop_drawdown_pct = float(portfolio_stop_drawdown_pct)
        self.recovery_drawdown_pct = float(recovery_drawdown_pct)
        self.min_volatility_factor = float(min_volatility_factor)
        self.max_volatility_factor = float(max_volatility_factor)
        self.risk_state = RiskState.NORMAL
        self.portfolio_stop_active = False
        self.portfolio_stop_count = 0

    def update_equity(self, current_equity: float) -> RiskSnapshot:
        """Update equity, drawdown, risk state, and portfolio-stop latch."""

        equity = float(current_equity)
        if equity < 0:
            raise ValueError("current_equity must not be negative.")
        self.current_equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        self.current_drawdown_pct = calculate_drawdown_pct(equity, self.peak_equity)
        triggered = False
        if (
            self.current_drawdown_pct > self.portfolio_stop_drawdown_pct
            and not self.portfolio_stop_active
        ):
            self.portfolio_stop_active = True
            self.portfolio_stop_count += 1
            triggered = True
        elif self.portfolio_stop_active and self.current_drawdown_pct < self.recovery_drawdown_pct:
            self.portfolio_stop_active = False

        self.risk_state = self._state_for_drawdown(self.current_drawdown_pct)
        return self.snapshot(portfolio_stop_triggered=triggered)

    def snapshot(self, portfolio_stop_triggered: bool = False) -> RiskSnapshot:
        return RiskSnapshot(
            current_equity=round(self.current_equity, 4),
            peak_equity=round(self.peak_equity, 4),
            current_drawdown_pct=round(self.current_drawdown_pct, 4),
            risk_state=self.risk_state.value,
            allocation_multiplier=self.allocation_multiplier,
            runner_enabled=self.runner_enabled,
            trend_holding_enabled=self.trend_holding_enabled,
            portfolio_stop_active=self.portfolio_stop_active,
            portfolio_stop_triggered=portfolio_stop_triggered,
        )

    @property
    def allocation_multiplier(self) -> float:
        if self.risk_state == RiskState.CAPITAL_PRESERVATION:
            return 0.25
        if self.risk_state == RiskState.DEFENSIVE:
            return 0.50
        if self.risk_state == RiskState.CAUTION:
            return 0.75
        return 1.0

    @property
    def runner_enabled(self) -> bool:
        return (
            self.risk_state in {RiskState.NORMAL, RiskState.CAUTION}
            and not self.portfolio_stop_active
        )

    @property
    def trend_holding_enabled(self) -> bool:
        return self.runner_enabled

    def volatility_factor(self, atr: float | None, atr_ma: float | None) -> float:
        """Return an ATR-based size multiplier clamped to [0.25, 1.0]."""

        if atr is None or atr_ma is None or atr <= 0 or atr_ma <= 0 or atr <= atr_ma:
            return 1.0
        factor = atr_ma / atr
        return min(self.max_volatility_factor, max(self.min_volatility_factor, factor))

    def position_size_details(
        self,
        *,
        entry_price: float,
        stop_price: float | None,
        available_cash: float,
        fee_rate: float = 0.0,
        atr: float | None = None,
        atr_ma: float | None = None,
    ) -> PositionSizeResult:
        """Calculate units to buy using 1% risk, drawdown state, and volatility."""

        entry = float(entry_price)
        cash = float(available_cash)
        stop = None if stop_price is None else float(stop_price)
        if entry <= 0 or cash <= 0 or stop is None or stop >= entry:
            return PositionSizeResult(0.0, 0.0, 0.0, self.allocation_multiplier, 1.0, 0.0)

        risk_per_unit = entry - stop
        volatility_factor = self.volatility_factor(atr, atr_ma)
        risk_amount = self.current_equity * self.risk_per_trade * self.allocation_multiplier
        risk_sized_units = (risk_amount / risk_per_unit) * volatility_factor
        max_affordable_units = cash / (entry * (1 + max(0.0, float(fee_rate))))
        position_size = max(0.0, min(risk_sized_units, max_affordable_units))
        notional = position_size * entry
        position_size_fraction = notional / self.current_equity if self.current_equity > 0 else 0.0
        return PositionSizeResult(
            position_size=position_size,
            notional=notional,
            risk_amount=risk_amount,
            allocation_multiplier=self.allocation_multiplier,
            volatility_factor=volatility_factor,
            position_size_fraction=position_size_fraction,
        )

    def calculate_position_size(
        self,
        *,
        entry_price: float,
        stop_price: float | None,
        available_cash: float,
        fee_rate: float = 0.0,
        atr: float | None = None,
        atr_ma: float | None = None,
    ) -> float:
        return self.position_size_details(
            entry_price=entry_price,
            stop_price=stop_price,
            available_cash=available_cash,
            fee_rate=fee_rate,
            atr=atr,
            atr_ma=atr_ma,
        ).position_size

    def _state_for_drawdown(self, drawdown_pct: float) -> RiskState:
        if drawdown_pct >= self.capital_preservation_drawdown_pct:
            return RiskState.CAPITAL_PRESERVATION
        if drawdown_pct >= self.defensive_drawdown_pct:
            return RiskState.DEFENSIVE
        if drawdown_pct >= self.caution_drawdown_pct:
            return RiskState.CAUTION
        return RiskState.NORMAL


def calculate_drawdown_pct(current_equity: float, peak_equity: float) -> float:
    if peak_equity <= 0:
        return 0.0
    return max(0.0, ((float(peak_equity) - float(current_equity)) / float(peak_equity)) * 100)


def calculate_atr_moving_average(
    candles: pd.DataFrame,
    atr_period: int = 14,
    ma_period: int = 20,
) -> float | None:
    """Calculate a short ATR moving average from recent candles only."""

    if candles.empty or atr_period <= 0 or ma_period <= 0:
        return None
    required = {"high", "low", "close"}
    if not required.issubset(candles.columns):
        return None
    tail = candles[list(required)].astype(float).tail(atr_period + ma_period + 1).reset_index(drop=True)
    if len(tail) < atr_period + 1:
        return None

    true_ranges: list[float] = []
    previous_close: float | None = None
    for row in tail.itertuples(index=False):
        high = float(row.high)
        low = float(row.low)
        close = float(row.close)
        if previous_close is None:
            previous_close = close
            continue
        true_ranges.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))
        previous_close = close

    if len(true_ranges) < atr_period:
        return None
    atr_values = pd.Series(true_ranges).rolling(atr_period, min_periods=atr_period).mean().dropna()
    if atr_values.empty:
        return None
    return round(float(atr_values.tail(ma_period).mean()), 4)
