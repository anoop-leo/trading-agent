"""Long-only trade simulator for Phase 1.5 backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from backtesting.profiles import StrategyProfile, get_strategy_profile


ENTRY_DECISIONS = {"BUY", "BUY WATCH", "STRONG BUY"}
EXIT_REASONS = (
    "STOP_LOSS",
    "TAKE_PROFIT",
    "TRAILING_STOP",
    "MOMENTUM_EXIT",
    "TIME_EXIT",
    "BEAR_TREND",
    "RUNNER_RSI_EXIT",
    "RUNNER_EMA_EXIT",
    "RUNNER_DAILY_EMA50_EXIT",
    "RUNNER_MACD_EXIT",
    "RUNNER_TRAILING_STOP",
    "RUNNER_MAX_DRAWDOWN_EXIT",
    "END_OF_BACKTEST",
)
REJECTION_REASONS = (
    "not_buy_decision",
    "bearish_alignment",
    "low_rr_ratio",
    "low_volume_ratio",
    "bear_market_regime",
    "daily_bear_trend",
    "below_4h_ema20",
    "four_hour_macd_not_bullish",
    "below_1h_ema20",
    "already_holding",
    "cooldown_active",
)
TRADE_COLUMNS = (
    "entry_timestamp",
    "exit_timestamp",
    "entry_price",
    "exit_price",
    "position_size",
    "entry_fee",
    "exit_fee",
    "pnl",
    "return_pct",
    "exit_reason",
    "open_reason",
    "close_reason",
    "entry_decision",
    "exit_decision",
    "entry_alignment",
    "exit_alignment",
    "entry_rr_ratio",
    "entry_volume_ratio",
    "entry_market_regime",
    "entry_regime_tag",
    "exit_regime_tag",
    "entry_bull_mode_active",
    "entry_daily_setup",
    "entry_4h_price",
    "entry_4h_ema20",
    "entry_stop_loss",
    "entry_target_1",
    "initial_risk",
    "entry_stop_type",
    "entry_stop_distance_pct",
    "entry_stop_distance_atr",
    "entry_atr",
    "entry_stop_candidates",
    "r_multiple",
    "max_price",
    "trailing_stop",
    "runner_return_pct",
    "runner_holding_hours",
    "profit_capture_ratio",
    "tp1_price",
    "tp2_price",
    "runner_exit_price",
    "entry_trend_state",
    "exit_trend_state",
    "partial_exits",
    "runner_max_drawdown_pct",
    "rejected_entry_reasons",
)


@dataclass(frozen=True)
class Trade:
    entry_timestamp: str
    exit_timestamp: str
    entry_price: float
    exit_price: float
    position_size: float
    entry_fee: float
    exit_fee: float
    pnl: float
    return_pct: float
    exit_reason: str
    open_reason: str
    close_reason: str
    entry_decision: str
    exit_decision: str
    entry_alignment: str
    exit_alignment: str
    entry_rr_ratio: float
    entry_volume_ratio: float
    entry_market_regime: str
    entry_regime_tag: str
    exit_regime_tag: str
    entry_bull_mode_active: bool
    entry_daily_setup: str
    entry_4h_price: float
    entry_4h_ema20: float
    entry_stop_loss: float | None
    entry_target_1: float | None
    initial_risk: float
    r_multiple: float
    max_price: float
    trailing_stop: float | None
    rejected_entry_reasons: list[str]
    runner_return_pct: float | None = None
    runner_holding_hours: float | None = None
    profit_capture_ratio: float | None = None
    tp1_price: float | None = None
    tp2_price: float | None = None
    runner_exit_price: float | None = None
    entry_trend_state: str | None = None
    exit_trend_state: str | None = None
    partial_exits: list[dict[str, Any]] | None = None
    runner_max_drawdown_pct: float | None = None
    entry_stop_type: str = "FIXED"
    entry_stop_distance_pct: float | None = None
    entry_stop_distance_atr: float | None = None
    entry_atr: float | None = None
    entry_stop_candidates: dict[str, Any] | None = None


@dataclass(frozen=True)
class PortfolioSnapshot:
    timestamp: str
    price: float
    cash: float
    position_size: float
    current_equity: float
    realized_pnl: float
    unrealized_pnl: float


class TradeSimulator:
    """Simulate a single long position using deterministic decision rules."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        cooldown_hours: float = 0.0,
        minimum_hold_hours: float = 48.0,
        momentum_exit_minimum_hours: float = 24.0,
        strategy_profile: StrategyProfile | None = None,
    ) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be greater than zero.")
        if fee_rate < 0 or slippage_rate < 0:
            raise ValueError("fee_rate and slippage_rate must not be negative.")
        if cooldown_hours < 0:
            raise ValueError("cooldown_hours must not be negative.")
        if minimum_hold_hours < 0 or momentum_exit_minimum_hours < 0:
            raise ValueError("minimum hold settings must not be negative.")
        self.strategy_profile = strategy_profile or get_strategy_profile("balanced")
        if not 0 < self.strategy_profile.allocation_per_trade <= 1:
            raise ValueError("allocation_per_trade must be greater than zero and no more than one.")

        self.initial_capital = float(initial_capital)
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.cooldown_hours = float(cooldown_hours)
        self.minimum_hold_hours = float(minimum_hold_hours)
        self.momentum_exit_minimum_hours = float(momentum_exit_minimum_hours)
        self.cash = float(initial_capital)
        self.position_size = 0.0
        self.entry_price: float | None = None
        self.entry_timestamp: str | None = None
        self.entry_fee = 0.0
        self.entry_total_cost = 0.0
        self.entry_audit: dict[str, Any] = {}
        self.entry_stop_loss: float | None = None
        self.entry_target_1: float | None = None
        self.initial_risk = 0.0
        self.highest_price = 0.0
        self.trailing_stop: float | None = None
        self.trailing_active = False
        self.cooldown_until: pd.Timestamp | None = None
        self.realized_pnl = 0.0
        self.trades: list[Trade] = []
        self.equity_curve: list[PortfolioSnapshot] = []
        self.entry_attempts = 0
        self.accepted_entries = 0
        self.rejected_entries = 0
        self.rejected_entry_reasons = {reason: 0 for reason in REJECTION_REASONS}
        self.exit_reasons: dict[str, int] = {reason: 0 for reason in EXIT_REASONS}
        self.exits_before_minimum_hold = 0
        self.momentum_exits_blocked = 0
        self.last_rejected_entry_reasons: list[str] = []

    @property
    def position_mode(self) -> str:
        return "HOLDING" if self.position_size > 0 else "NO_POSITION"

    def process_signal(self, signal: dict[str, Any]) -> PortfolioSnapshot:
        """Apply one signal to the simulated portfolio."""

        timestamp = _format_timestamp(signal["timestamp"])
        timestamp_value = pd.Timestamp(timestamp)
        price = float(signal["price"])
        final_decision = str(signal["final_decision"])
        market_regime = signal.get("market_regime")
        trend_score = int(signal.get("trend_score", 10))
        self.last_rejected_entry_reasons = []

        if self._should_audit_entry(final_decision):
            rejected_reasons = self._entry_rejection_reasons(signal, timestamp_value)
            self.entry_attempts += 1
            if rejected_reasons:
                self._record_rejected_entry(rejected_reasons)
            else:
                self._open_long(timestamp, price, signal)

        if self.position_size > 0:
            exit_reason = self._exit_reason(signal, timestamp_value, price, market_regime, trend_score)
            if exit_reason is not None:
                self._close_long(timestamp, price, exit_reason, signal)

        snapshot = self.snapshot(timestamp, price)
        self.equity_curve.append(snapshot)
        return snapshot

    def snapshot(self, timestamp: str, price: float) -> PortfolioSnapshot:
        position_value = self.position_size * float(price)
        current_equity = self.cash + position_value
        unrealized_pnl = position_value - self.entry_total_cost if self.position_size > 0 else 0.0
        return PortfolioSnapshot(
            timestamp=timestamp,
            price=float(price),
            cash=self.cash,
            position_size=self.position_size,
            current_equity=current_equity,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=unrealized_pnl,
        )

    def close_open_position(
        self,
        timestamp: str,
        price: float,
        signal: dict[str, Any],
        exit_reason: str = "END_OF_BACKTEST",
    ) -> PortfolioSnapshot:
        if self.position_size > 0:
            self._close_long(timestamp, price, exit_reason, signal)
        return self.snapshot(timestamp, price)

    def _should_audit_entry(self, final_decision: str) -> bool:
        return self.position_size <= 0 or final_decision in ENTRY_DECISIONS or final_decision == "ADD"

    def _entry_rejection_reasons(self, signal: dict[str, Any], timestamp: pd.Timestamp) -> list[str]:
        reasons: list[str] = []
        final_decision = str(signal["final_decision"])
        alignment = str(signal.get("alignment", ""))
        rr_ratio = _optional_float(signal.get("rr_ratio"))
        volume_ratio = _optional_float(signal.get("volume_ratio"))
        market_regime = signal.get("market_regime")
        daily_setup = signal.get("daily_setup")
        four_hour_price = _optional_float(signal.get("four_hour_price"))
        four_hour_ema20 = _optional_float(signal.get("four_hour_ema20"))
        one_hour_ema20 = _optional_float(signal.get("ema20"))
        position_mode = signal.get("position_mode", self.position_mode)
        bull_mode_active = self._bull_market_mode_active(signal)
        min_rr_ratio = self._effective_min_rr_ratio(bull_mode_active)
        min_volume_ratio = self._effective_min_volume_ratio(bull_mode_active)

        if final_decision not in ENTRY_DECISIONS:
            reasons.append("not_buy_decision")
        if self.strategy_profile.require_alignment and not self._alignment_allowed(alignment, bull_mode_active):
            reasons.append("bearish_alignment")
        if self.strategy_profile.require_rr_ratio and (
            rr_ratio is None or rr_ratio < min_rr_ratio
        ):
            reasons.append("low_rr_ratio")
        if self.strategy_profile.require_volume_ratio and (
            volume_ratio is None or volume_ratio < min_volume_ratio
        ):
            reasons.append("low_volume_ratio")
        if market_regime == "BEAR":
            reasons.append("bear_market_regime")
        if daily_setup == "BEAR_TREND":
            reasons.append("daily_bear_trend")
        if four_hour_price is None or four_hour_ema20 is None or four_hour_price <= four_hour_ema20:
            reasons.append("below_4h_ema20")
        if self.strategy_profile.require_4h_macd_bullish and signal.get("four_hour_macd") != "bullish":
            reasons.append("four_hour_macd_not_bullish")
        if (
            self.strategy_profile.require_price_above_1h_ema20
            and (one_hour_ema20 is None or float(signal["price"]) <= one_hour_ema20)
        ):
            reasons.append("below_1h_ema20")
        if self.position_size > 0 or position_mode != "NO_POSITION":
            reasons.append("already_holding")
        if self._cooldown_active(timestamp):
            reasons.append("cooldown_active")
        return reasons

    def _record_rejected_entry(self, reasons: list[str]) -> None:
        self.rejected_entries += 1
        self.last_rejected_entry_reasons = reasons.copy()
        for reason in reasons:
            self.rejected_entry_reasons[reason] += 1

    def _open_long(self, timestamp: str, price: float, signal: dict[str, Any]) -> None:
        adjusted_entry_price = price * (1 + self.slippage_rate)
        entry_budget = self.cash * self.strategy_profile.allocation_per_trade
        entry_notional = entry_budget / (1 + self.fee_rate)
        entry_fee = entry_notional * self.fee_rate
        self.position_size = entry_notional / adjusted_entry_price
        self.entry_price = adjusted_entry_price
        self.entry_timestamp = timestamp
        self.entry_fee = entry_fee
        self.entry_total_cost = entry_notional + entry_fee
        self.entry_stop_loss = _optional_float(signal.get("stop_loss"))
        self.initial_risk = (
            adjusted_entry_price - self.entry_stop_loss
            if self.entry_stop_loss is not None and self.entry_stop_loss < adjusted_entry_price
            else 0.0
        )
        self.entry_target_1 = self._entry_target(signal)
        self.highest_price = price
        self.trailing_stop = None
        self.trailing_active = False
        self.entry_audit = {
            "open_reason": "all_entry_gates_passed",
            "entry_decision": str(signal["final_decision"]),
            "entry_alignment": str(signal.get("alignment", "")),
            "entry_rr_ratio": float(signal.get("rr_ratio", 0.0)),
            "entry_volume_ratio": float(signal.get("volume_ratio", 0.0)),
            "entry_market_regime": str(signal.get("market_regime", "")),
            "entry_regime_tag": _regime_tag(signal),
            "entry_bull_mode_active": self._bull_market_mode_active(signal),
            "entry_daily_setup": str(signal.get("daily_setup", "")),
            "entry_4h_price": float(signal.get("four_hour_price", 0.0)),
            "entry_4h_ema20": float(signal.get("four_hour_ema20", 0.0)),
            "entry_stop_loss": self.entry_stop_loss,
            "entry_target_1": self.entry_target_1,
            "initial_risk": self.initial_risk,
            "entry_stop_type": str(signal.get("stop_type", "FIXED") or "FIXED"),
            "entry_stop_distance_pct": _optional_float(signal.get("stop_distance_pct")),
            "entry_stop_distance_atr": _optional_float(signal.get("stop_distance_atr")),
            "entry_atr": _optional_float(signal.get("atr")),
            "entry_stop_candidates": signal.get("stop_candidates") or {},
            "rejected_entry_reasons": [],
        }
        self.cash -= self.entry_total_cost
        self.accepted_entries += 1

    def _close_long(self, timestamp: str, price: float, exit_reason: str, signal: dict[str, Any]) -> None:
        if self.entry_price is None or self.entry_timestamp is None:
            return
        adjusted_exit_price = price * (1 - self.slippage_rate)
        gross_proceeds = self.position_size * adjusted_exit_price
        exit_fee = gross_proceeds * self.fee_rate
        net_proceeds = gross_proceeds - exit_fee
        pnl = net_proceeds - self.entry_total_cost
        return_pct = (pnl / self.entry_total_cost) * 100
        r_multiple = self._r_multiple(adjusted_exit_price)

        self.trades.append(
            Trade(
                entry_timestamp=self.entry_timestamp,
                exit_timestamp=timestamp,
                entry_price=self.entry_price,
                exit_price=adjusted_exit_price,
                position_size=self.position_size,
                entry_fee=self.entry_fee,
                exit_fee=exit_fee,
                pnl=pnl,
                return_pct=return_pct,
                exit_reason=exit_reason,
                open_reason=str(self.entry_audit.get("open_reason", "")),
                close_reason=exit_reason,
                entry_decision=str(self.entry_audit.get("entry_decision", "")),
                exit_decision=str(signal.get("final_decision", "")),
                entry_alignment=str(self.entry_audit.get("entry_alignment", "")),
                exit_alignment=str(signal.get("alignment", "")),
                entry_rr_ratio=float(self.entry_audit.get("entry_rr_ratio", 0.0)),
                entry_volume_ratio=float(self.entry_audit.get("entry_volume_ratio", 0.0)),
                entry_market_regime=str(self.entry_audit.get("entry_market_regime", "")),
                entry_regime_tag=str(self.entry_audit.get("entry_regime_tag", "neutral")),
                exit_regime_tag=_regime_tag(signal),
                entry_bull_mode_active=bool(self.entry_audit.get("entry_bull_mode_active", False)),
                entry_daily_setup=str(self.entry_audit.get("entry_daily_setup", "")),
                entry_4h_price=float(self.entry_audit.get("entry_4h_price", 0.0)),
                entry_4h_ema20=float(self.entry_audit.get("entry_4h_ema20", 0.0)),
                entry_stop_loss=self.entry_audit.get("entry_stop_loss"),
                entry_target_1=self.entry_audit.get("entry_target_1"),
                initial_risk=float(self.entry_audit.get("initial_risk", 0.0)),
                entry_stop_type=str(self.entry_audit.get("entry_stop_type", "FIXED")),
                entry_stop_distance_pct=self.entry_audit.get("entry_stop_distance_pct"),
                entry_stop_distance_atr=self.entry_audit.get("entry_stop_distance_atr"),
                entry_atr=self.entry_audit.get("entry_atr"),
                entry_stop_candidates=self.entry_audit.get("entry_stop_candidates"),
                r_multiple=r_multiple,
                max_price=self.highest_price,
                trailing_stop=self.trailing_stop,
                rejected_entry_reasons=list(self.entry_audit.get("rejected_entry_reasons", [])),
            )
        )
        self.cash += net_proceeds
        self.realized_pnl += pnl
        self.exit_reasons[exit_reason] = self.exit_reasons.get(exit_reason, 0) + 1
        self._activate_cooldown(timestamp)
        self.position_size = 0.0
        self.entry_price = None
        self.entry_timestamp = None
        self.entry_fee = 0.0
        self.entry_total_cost = 0.0
        self.entry_audit = {}
        self.entry_stop_loss = None
        self.entry_target_1 = None
        self.initial_risk = 0.0
        self.highest_price = 0.0
        self.trailing_stop = None
        self.trailing_active = False

    def _exit_reason(
        self,
        signal: dict[str, Any],
        timestamp: pd.Timestamp,
        price: float,
        market_regime: object,
        trend_score: int,
    ) -> str | None:
        self._update_trade_progress(price)
        if self.entry_stop_loss is not None and price <= self.entry_stop_loss:
            return "STOP_LOSS"

        holding_hours = self._holding_hours(timestamp)
        daily_bear_breakdown = self._daily_bear_breakdown(signal)
        if holding_hours < self.minimum_hold_hours and daily_bear_breakdown:
            return "BEAR_TREND"

        take_profit = self.entry_target_1 is not None and price >= self.entry_target_1
        trailing_stop = self.trailing_active and self.trailing_stop is not None and price <= self.trailing_stop
        momentum_exit = self._momentum_exit_condition(signal, price, holding_hours)
        time_exit = holding_hours > 168 and not self._is_profitable(price)
        bear_trend = market_regime == "BEAR" and trend_score <= 3

        if holding_hours < self.minimum_hold_hours:
            momentum_signal_present = self._momentum_signal_present(signal)
            if take_profit or trailing_stop or time_exit or bear_trend or momentum_signal_present:
                self.exits_before_minimum_hold += 1
            if momentum_signal_present:
                self.momentum_exits_blocked += 1
            return None

        if take_profit:
            return "TAKE_PROFIT"
        if trailing_stop:
            return "TRAILING_STOP"
        if momentum_exit:
            return "MOMENTUM_EXIT"
        if self._blocked_momentum_exit(signal, price, holding_hours):
            self.momentum_exits_blocked += 1
        if time_exit:
            return "TIME_EXIT"
        if bear_trend or daily_bear_breakdown:
            return "BEAR_TREND"
        return None

    def _entry_target(self, signal: dict[str, Any]) -> float | None:
        explicit_target = _optional_float(signal.get("target_1"))
        if explicit_target is not None:
            return explicit_target
        if self.entry_price is None or self.initial_risk <= 0:
            return None
        return self.entry_price + (2 * self.initial_risk)

    def _bull_market_mode_active(self, signal: dict[str, Any]) -> bool:
        daily_rsi = _optional_float(signal.get("daily_rsi"))
        daily_price = _optional_float(signal.get("daily_price"))
        daily_ema200 = _optional_float(signal.get("daily_ema200"))
        return (
            self.strategy_profile.enable_bull_market_mode
            and daily_rsi is not None
            and daily_price is not None
            and daily_ema200 is not None
            and daily_rsi > 55
            and daily_price > daily_ema200
        )

    def _effective_min_rr_ratio(self, bull_mode_active: bool) -> float:
        if bull_mode_active:
            return self.strategy_profile.bull_min_rr_ratio
        return self.strategy_profile.min_rr_ratio

    def _effective_min_volume_ratio(self, bull_mode_active: bool) -> float:
        if bull_mode_active:
            return self.strategy_profile.bull_min_volume_ratio
        return self.strategy_profile.min_volume_ratio

    def _alignment_allowed(self, alignment: str, bull_mode_active: bool) -> bool:
        if alignment == "BULLISH_ALIGNMENT":
            return True
        return (
            bull_mode_active
            and self.strategy_profile.bull_allow_pullback_alignment
            and alignment == "PULLBACK_IN_UPTREND"
        )

    def _update_trade_progress(self, price: float) -> None:
        self.highest_price = max(self.highest_price, price)
        if self.initial_risk <= 0 or self.entry_price is None:
            return
        gain_r = (self.highest_price - self.entry_price) / self.initial_risk
        if gain_r >= 2.0:
            self.trailing_active = True
            self.trailing_stop = self.highest_price - self.initial_risk

    def _is_profitable(self, price: float) -> bool:
        return self.entry_price is not None and price > self.entry_price

    def _momentum_exit_condition(self, signal: dict[str, Any], price: float, holding_hours: float) -> bool:
        four_hour_price = _optional_float(signal.get("four_hour_price"))
        four_hour_ema20 = _optional_float(signal.get("four_hour_ema20"))
        return (
            self._is_profitable(price)
            and holding_hours >= self.momentum_exit_minimum_hours
            and signal.get("four_hour_macd") == "bearish"
            and four_hour_price is not None
            and four_hour_ema20 is not None
            and four_hour_price < four_hour_ema20
        )

    def _blocked_momentum_exit(self, signal: dict[str, Any], price: float, holding_hours: float) -> bool:
        return self._momentum_signal_present(signal) and not self._momentum_exit_condition(signal, price, holding_hours)

    def _momentum_signal_present(self, signal: dict[str, Any]) -> bool:
        return signal.get("macd") == "bearish" or signal.get("four_hour_macd") == "bearish"

    def _daily_bear_breakdown(self, signal: dict[str, Any]) -> bool:
        daily_price = _optional_float(signal.get("daily_price"))
        daily_ema20 = _optional_float(signal.get("daily_ema20"))
        return (
            signal.get("daily_setup") == "BEAR_TREND"
            and daily_price is not None
            and daily_ema20 is not None
            and daily_price < daily_ema20
        )

    def _holding_hours(self, timestamp: pd.Timestamp) -> float:
        if self.entry_timestamp is None:
            return 0.0
        return (timestamp - pd.Timestamp(self.entry_timestamp)).total_seconds() / 3600

    def _r_multiple(self, exit_price: float) -> float:
        if self.entry_price is None or self.initial_risk <= 0:
            return 0.0
        return round((exit_price - self.entry_price) / self.initial_risk, 2)

    def trades_as_dicts(self) -> list[dict[str, Any]]:
        return [trade.__dict__.copy() for trade in self.trades]

    def equity_curve_frame(self) -> pd.DataFrame:
        return pd.DataFrame([snapshot.__dict__.copy() for snapshot in self.equity_curve])

    def execution_summary(self, total_decisions: int) -> dict[str, Any]:
        return {
            "total_decisions": total_decisions,
            "entry_attempts": self.entry_attempts,
            "accepted_entries": self.accepted_entries,
            "rejected_entries": self.rejected_entries,
            "rejected_entry_reasons": self.rejected_entry_reasons.copy(),
            "exit_reasons": self.exit_reasons.copy(),
            "exits_before_minimum_hold": self.exits_before_minimum_hold,
            "momentum_exits_blocked": self.momentum_exits_blocked,
            "stop_out_count": self.exit_reasons.get("STOP_LOSS", 0),
            "stop_type_usage": _count_stop_types(self.trades),
            "average_stop_distance_pct": _rounded_mean(
                [
                    trade.entry_stop_distance_pct
                    for trade in self.trades
                    if trade.entry_stop_distance_pct is not None
                ]
            ),
            "average_stop_distance_atr": _rounded_mean(
                [
                    trade.entry_stop_distance_atr
                    for trade in self.trades
                    if trade.entry_stop_distance_atr is not None
                ]
            ),
            "strategy_profile": self.strategy_profile.to_dict(),
        }

    def _cooldown_active(self, timestamp: pd.Timestamp) -> bool:
        return self.cooldown_until is not None and timestamp < self.cooldown_until

    def _activate_cooldown(self, timestamp: str) -> None:
        if self.cooldown_hours <= 0:
            return
        self.cooldown_until = pd.Timestamp(timestamp) + pd.Timedelta(hours=self.cooldown_hours)


def _format_timestamp(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.isoformat()


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _regime_tag(signal: dict[str, Any]) -> str:
    daily_rsi = _optional_float(signal.get("daily_rsi"))
    daily_price = _optional_float(signal.get("daily_price"))
    daily_ema200 = _optional_float(signal.get("daily_ema200"))
    if daily_rsi is None or daily_price is None or daily_ema200 is None:
        market_regime = str(signal.get("market_regime", "")).lower()
        return market_regime if market_regime in {"bull", "neutral", "bear"} else "neutral"
    if daily_rsi > 55 and daily_price > daily_ema200:
        return "bull"
    if daily_rsi < 45 and daily_price < daily_ema200:
        return "bear"
    return "neutral"


def _count_stop_types(trades: list[Trade]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trade in trades:
        stop_type = str(trade.entry_stop_type or "UNKNOWN")
        counts[stop_type] = counts.get(stop_type, 0) + 1
    return counts


def _rounded_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(float(value) for value in values) / len(values), 4)
