"""Hybrid Trend Rider position manager for Phase 1.10 backtests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from backtesting.trade_simulator import Trade, _optional_float, _regime_tag
from backtesting.trend_rider_simulator import TrendRiderSimulator


HYBRID_RUNNER_EXIT_REASONS = (
    "RUNNER_RSI_EXIT",
    "RUNNER_EMA_EXIT",
    "RUNNER_DAILY_EMA50_EXIT",
    "RUNNER_MAX_DRAWDOWN_EXIT",
    "RUNNER_TRAILING_STOP",
    "END_OF_BACKTEST",
)


@dataclass(frozen=True)
class HybridRunnerProfile:
    name: str
    runner_size: float
    tp1_close_fraction: float
    tp2_close_fraction: float
    trailing_stop_pct: float
    rsi_field: str
    rsi_exit_below: float
    ema_fast_field: str
    ema_slow_field: str
    price_below_field: str | None = None
    price_below_ema_field: str | None = None
    max_runner_drawdown_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


HYBRID_RUNNER_PROFILES: dict[str, HybridRunnerProfile] = {
    "hybrid_trend_rider": HybridRunnerProfile(
        name="hybrid_trend_rider",
        runner_size=0.25,
        tp1_close_fraction=0.50,
        tp2_close_fraction=0.25,
        trailing_stop_pct=0.15,
        rsi_field="daily_rsi",
        rsi_exit_below=50.0,
        ema_fast_field="daily_ema20",
        ema_slow_field="daily_ema50",
        price_below_field="daily_price",
        price_below_ema_field="daily_ema50",
        max_runner_drawdown_pct=25.0,
    ),
    "hybrid_conservative": HybridRunnerProfile(
        name="hybrid_conservative",
        runner_size=0.25,
        tp1_close_fraction=0.50,
        tp2_close_fraction=0.25,
        trailing_stop_pct=0.15,
        rsi_field="daily_rsi",
        rsi_exit_below=50.0,
        ema_fast_field="daily_ema20",
        ema_slow_field="daily_ema50",
    ),
    "hybrid_balanced": HybridRunnerProfile(
        name="hybrid_balanced",
        runner_size=0.40,
        tp1_close_fraction=0.40,
        tp2_close_fraction=0.20,
        trailing_stop_pct=0.20,
        rsi_field="daily_rsi",
        rsi_exit_below=45.0,
        ema_fast_field="daily_ema50",
        ema_slow_field="daily_ema100",
    ),
    "hybrid_aggressive": HybridRunnerProfile(
        name="hybrid_aggressive",
        runner_size=0.50,
        tp1_close_fraction=0.25,
        tp2_close_fraction=0.25,
        trailing_stop_pct=0.25,
        rsi_field="weekly_rsi",
        rsi_exit_below=45.0,
        ema_fast_field="weekly_ema20",
        ema_slow_field="weekly_ema50",
    ),
}


class HybridTrendRiderSimulator(TrendRiderSimulator):
    """Partial-profit trend rider with breakeven and runner drawdown controls."""

    def __init__(self, *args: Any, hybrid_profile_name: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        profile_name = hybrid_profile_name or "hybrid_trend_rider"
        if profile_name not in HYBRID_RUNNER_PROFILES:
            supported = ", ".join(sorted(HYBRID_RUNNER_PROFILES))
            raise ValueError(f"Unsupported hybrid runner profile {profile_name!r}. Supported profiles: {supported}.")
        self.runner_profile = HYBRID_RUNNER_PROFILES[profile_name]
        self.tp1_hit_count = 0
        self.tp2_hit_count = 0
        self.runner_activation_count = 0
        self.runner_exit_reasons = {reason: 0 for reason in HYBRID_RUNNER_EXIT_REASONS}
        self.runner_drawdowns: list[float] = []

    def _open_long(self, timestamp: str, price: float, signal: dict[str, Any]) -> None:
        super()._open_long(timestamp, price, signal)
        self._apply_support_stop_fallback(signal)

    def _process_profit_targets(self, timestamp: str, price: float) -> None:
        if self.entry_price is None or self.initial_risk <= 0 or self.initial_position_size <= 0:
            return
        tp1_price = self.entry_price + (2 * self.initial_risk)
        tp2_price = self.entry_price + (4 * self.initial_risk)
        if not self.tp1_done and price >= tp1_price:
            self._partial_close(timestamp, price, self.runner_profile.tp1_close_fraction, "TP1_2R")
            self.tp1_done = True
            self.tp1_hit_count += 1
            self.entry_stop_loss = self.entry_price
        if not self.tp2_done and price >= tp2_price:
            self._partial_close(timestamp, price, self.runner_profile.tp2_close_fraction, "TP2_4R")
            self.tp2_done = True
            self.tp2_hit_count += 1
            self.runner_activation_count += 1
            self.runner_activation_timestamp = timestamp
            self.runner_highest_price = price
            self.runner_max_drawdown_pct = 0.0

    def _runner_exit_reason(self, signal: dict[str, Any], price: float) -> str | None:
        if not self.tp2_done:
            return None
        self._update_runner_progress(price)

        runner_rsi = _optional_float(signal.get(self.runner_profile.rsi_field))
        if runner_rsi is not None and runner_rsi < self.runner_profile.rsi_exit_below:
            return "RUNNER_RSI_EXIT"

        ema_fast = _optional_float(signal.get(self.runner_profile.ema_fast_field))
        ema_slow = _optional_float(signal.get(self.runner_profile.ema_slow_field))
        if ema_fast is not None and ema_slow is not None and ema_fast < ema_slow:
            return "RUNNER_EMA_EXIT"

        price_value = _optional_float(signal.get(self.runner_profile.price_below_field))
        ema_value = _optional_float(signal.get(self.runner_profile.price_below_ema_field))
        if price_value is not None and ema_value is not None and price_value < ema_value:
            return "RUNNER_DAILY_EMA50_EXIT"

        if (
            self.runner_profile.max_runner_drawdown_pct is not None
            and self.runner_max_drawdown_pct > self.runner_profile.max_runner_drawdown_pct
        ):
            return "RUNNER_MAX_DRAWDOWN_EXIT"

        if self.runner_highest_price > 0 and price <= self.runner_highest_price * (1 - self.runner_profile.trailing_stop_pct):
            return "RUNNER_TRAILING_STOP"
        return None

    def _update_runner_high(self, price: float) -> None:
        self.highest_price = max(self.highest_price, price)
        if self.tp2_done:
            self._update_runner_progress(price)

    def _close_long(self, timestamp: str, price: float, exit_reason: str, signal: dict[str, Any]) -> None:
        if self.entry_price is None or self.entry_timestamp is None:
            return
        adjusted_exit_price = price * (1 - self.slippage_rate)
        gross_proceeds = self.position_size * adjusted_exit_price
        exit_fee = gross_proceeds * self.fee_rate
        net_proceeds = gross_proceeds - exit_fee
        final_pnl = net_proceeds - self.entry_total_cost
        cost_fields = self._trade_cost_fields(price, adjusted_exit_price, self.position_size, exit_fee)
        total_pnl = cost_fields["net_pnl"]
        original_cost = self.original_entry_total_cost if self.original_entry_total_cost > 0 else self.entry_total_cost
        return_pct = (total_pnl / original_cost) * 100 if original_cost > 0 else 0.0
        runner_active = self.tp2_done
        runner_return_pct = ((adjusted_exit_price / self.entry_price) - 1) * 100 if runner_active else None
        runner_holding_hours = self._runner_holding_hours(pd.Timestamp(timestamp)) if runner_active else None
        max_available_gain = max(0.0, ((self.highest_price / self.entry_price) - 1) * 100)
        captured_gain = self._weighted_captured_gain_pct(adjusted_exit_price)
        profit_capture_ratio = captured_gain / max_available_gain if max_available_gain > 0 else None
        runner_drawdown = self.runner_max_drawdown_pct if runner_active else None

        self.trades.append(
            Trade(
                entry_timestamp=self.entry_timestamp,
                exit_timestamp=timestamp,
                signal_entry_price=cost_fields["signal_entry_price"],
                actual_entry_price=cost_fields["actual_entry_price"],
                signal_exit_price=cost_fields["signal_exit_price"],
                actual_exit_price=cost_fields["actual_exit_price"],
                entry_price=cost_fields["actual_entry_price"],
                exit_price=cost_fields["actual_exit_price"],
                position_size=cost_fields["position_size"],
                entry_slippage_cost=cost_fields["entry_slippage_cost"],
                exit_slippage_cost=cost_fields["exit_slippage_cost"],
                total_slippage_cost=cost_fields["total_slippage_cost"],
                gross_pnl_before_fees_and_slippage=cost_fields["gross_pnl_before_fees_and_slippage"],
                gross_pnl_after_slippage_before_fees=cost_fields["gross_pnl_after_slippage_before_fees"],
                entry_fee=cost_fields["entry_fee"],
                exit_fee=cost_fields["exit_fee"],
                total_fee=cost_fields["total_fee"],
                net_pnl=cost_fields["net_pnl"],
                pnl=total_pnl,
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
                r_multiple=self._r_multiple(adjusted_exit_price),
                max_price=self.highest_price,
                trailing_stop=self.runner_highest_price * (1 - self.runner_profile.trailing_stop_pct)
                if self.runner_highest_price > 0
                else None,
                rejected_entry_reasons=list(self.entry_audit.get("rejected_entry_reasons", [])),
                runner_return_pct=round(runner_return_pct, 2) if runner_return_pct is not None else None,
                runner_holding_hours=round(runner_holding_hours, 2) if runner_holding_hours is not None else None,
                profit_capture_ratio=round(profit_capture_ratio, 4) if profit_capture_ratio is not None else None,
                partial_exits=self.partial_exits.copy(),
                runner_max_drawdown_pct=round(runner_drawdown, 2) if runner_drawdown is not None else None,
            )
        )
        if runner_active:
            self.runner_drawdowns.append(self.runner_max_drawdown_pct)
            if exit_reason in self.runner_exit_reasons:
                self.runner_exit_reasons[exit_reason] += 1
        self.cash += net_proceeds
        self.realized_pnl += final_pnl
        self.exit_reasons[exit_reason] = self.exit_reasons.get(exit_reason, 0) + 1
        self._activate_cooldown(timestamp)
        self.position_size = 0.0
        self.entry_signal_price = None
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
        self._reset_trend_rider_state()

    def execution_summary(self, total_decisions: int) -> dict[str, Any]:
        summary = super().execution_summary(total_decisions)
        summary.update(
            {
                "tp1_hit_count": self.tp1_hit_count,
                "tp2_hit_count": self.tp2_hit_count,
                "runner_activation_count": self.runner_activation_count,
                "runner_exit_reasons": self.runner_exit_reasons.copy(),
                "average_runner_drawdown_pct": _rounded_mean(self.runner_drawdowns),
                "max_runner_drawdown_pct": round(max(self.runner_drawdowns), 2) if self.runner_drawdowns else 0.0,
                "hybrid_runner_profile": self.runner_profile.to_dict(),
            }
        )
        return summary

    def _update_runner_progress(self, price: float) -> None:
        if self.runner_highest_price <= 0:
            self.runner_highest_price = price
        self.runner_highest_price = max(self.runner_highest_price, price)
        if self.runner_highest_price <= 0:
            return
        drawdown = max(0.0, (self.runner_highest_price - price) / self.runner_highest_price * 100)
        self.runner_max_drawdown_pct = max(self.runner_max_drawdown_pct, drawdown)

    def _apply_support_stop_fallback(self, signal: dict[str, Any]) -> None:
        if self.entry_price is None or self.initial_risk > 0:
            return
        support = _optional_float(signal.get("support"))
        if support is None or support >= self.entry_price:
            return
        self.entry_stop_loss = support
        self.initial_risk = self.entry_price - support
        self.entry_target_1 = self.entry_price + (2 * self.initial_risk)
        self.entry_audit["entry_stop_loss"] = self.entry_stop_loss
        self.entry_audit["entry_target_1"] = self.entry_target_1
        self.entry_audit["initial_risk"] = self.initial_risk

    def _runner_holding_hours(self, timestamp: pd.Timestamp) -> float:
        if self.runner_activation_timestamp is None:
            return self._holding_hours(timestamp)
        return (timestamp - pd.Timestamp(self.runner_activation_timestamp)).total_seconds() / 3600

    def _weighted_captured_gain_pct(self, adjusted_exit_price: float) -> float:
        if self.entry_price is None:
            return 0.0
        captured = 0.0
        closed_fraction = 0.0
        for partial_exit in self.partial_exits:
            fraction = max(0.0, float(partial_exit.get("position_fraction", 0.0)))
            partial_price = float(partial_exit.get("price", adjusted_exit_price))
            captured += fraction * max(0.0, ((partial_price / self.entry_price) - 1) * 100)
            closed_fraction += fraction
        runner_fraction = max(0.0, 1.0 - closed_fraction)
        captured += runner_fraction * max(0.0, ((adjusted_exit_price / self.entry_price) - 1) * 100)
        return captured

    def _reset_trend_rider_state(self) -> None:
        super()._reset_trend_rider_state()
        self.runner_activation_timestamp: str | None = None
        self.runner_max_drawdown_pct = 0.0


def _rounded_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)
