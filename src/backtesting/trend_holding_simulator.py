"""Trend Holding Engine simulator for Phase 1.13 backtests."""

from __future__ import annotations

from typing import Any

import pandas as pd

from backtesting.trade_simulator import PortfolioSnapshot, Trade, TradeSimulator, _optional_float, _regime_tag
from strategy.trend_holding_engine import TrendHoldingConfig, TrendHoldingEngine, determine_trend_state


TREND_HOLDING_RUNNER_EXIT_REASONS = (
    "RUNNER_DAILY_EMA50_EXIT",
    "RUNNER_MACD_EXIT",
    "RUNNER_TRAILING_STOP",
    "END_OF_BACKTEST",
)


class TrendHoldingSimulator(TradeSimulator):
    """Aggressive entries with partial profits and a long-running trend runner."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.engine = TrendHoldingEngine(TrendHoldingConfig())
        self.tp1_hit_count = 0
        self.tp2_hit_count = 0
        self.runner_activation_count = 0
        self.runner_exit_reasons = {reason: 0 for reason in TREND_HOLDING_RUNNER_EXIT_REASONS}
        self._reset_trend_holding_state()

    def process_signal(self, signal: dict[str, Any]) -> PortfolioSnapshot:
        timestamp = self._format_signal_timestamp(signal)
        timestamp_value = pd.Timestamp(timestamp)
        price = float(signal["price"])
        final_decision = str(signal["final_decision"])
        self.last_rejected_entry_reasons = []

        if self._should_audit_entry(final_decision):
            rejected_reasons = self._entry_rejection_reasons(signal, timestamp_value)
            self.entry_attempts += 1
            if rejected_reasons:
                self._record_rejected_entry(rejected_reasons)
            else:
                self._open_long(timestamp, price, signal)

        if self.position_size > 0:
            self._update_runner_high(price)
            if self.entry_stop_loss is not None and price <= self.entry_stop_loss:
                self._close_long(timestamp, price, "STOP_LOSS", signal)
            else:
                self._process_profit_targets(timestamp, price)
                runner_exit_reason = self._runner_exit_reason(signal) if self.runner_active else None
                if runner_exit_reason is not None and self.position_size > 0:
                    self._close_long(timestamp, price, runner_exit_reason, signal)

        snapshot = self.snapshot(timestamp, price)
        self.equity_curve.append(snapshot)
        return snapshot

    def _open_long(self, timestamp: str, price: float, signal: dict[str, Any]) -> None:
        super()._open_long(timestamp, price, signal)
        self.initial_position_size = self.position_size
        self.original_entry_total_cost = self.entry_total_cost
        self.partial_realized_pnl = 0.0
        self.partial_exits = []
        self.tp1_done = False
        self.tp2_done = False
        self.runner_active = False
        self.runner_activation_timestamp = None
        self.runner_highest_close = price
        self.tp1_price = self.engine.tp1_price(self.entry_price, self.initial_risk)
        self.tp2_price = self.engine.tp2_price(self.entry_price, self.initial_risk)
        self.entry_audit["tp1_price"] = self.tp1_price
        self.entry_audit["tp2_price"] = self.tp2_price
        self.entry_audit["entry_trend_state"] = determine_trend_state(signal).value

    def _process_profit_targets(self, timestamp: str, price: float) -> None:
        if self.entry_price is None or self.initial_risk <= 0 or self.initial_position_size <= 0:
            return
        if not self.tp1_done and self.tp1_price is not None and price >= self.tp1_price:
            self._partial_close(timestamp, price, self.engine.config.tp1_close_fraction, "TP1_2R")
            self.tp1_done = True
            self.tp1_hit_count += 1
            self.entry_stop_loss = self.entry_price
            self.entry_audit["entry_stop_loss"] = self.entry_stop_loss
        if not self.tp2_done and self.tp2_price is not None and price >= self.tp2_price:
            self._partial_close(timestamp, price, self.engine.config.tp2_close_fraction, "TP2_4R")
            self.tp2_done = True
            self.tp2_hit_count += 1
            self.runner_active = True
            self.runner_activation_count += 1
            self.runner_activation_timestamp = timestamp
            self.runner_highest_close = price

    def _partial_close(self, timestamp: str, price: float, fraction_of_initial: float, reason: str) -> None:
        if self.initial_position_size <= 0:
            return
        close_size = min(self.initial_position_size * fraction_of_initial, self.position_size)
        if close_size <= 0:
            return
        adjusted_exit_price = price * (1 - self.slippage_rate)
        gross_proceeds = close_size * adjusted_exit_price
        exit_fee = gross_proceeds * self.fee_rate
        net_proceeds = gross_proceeds - exit_fee
        cost_basis = self.original_entry_total_cost * (close_size / self.initial_position_size)
        pnl = net_proceeds - cost_basis
        self.cash += net_proceeds
        self.realized_pnl += pnl
        self.partial_realized_pnl += pnl
        self.position_size -= close_size
        self.entry_total_cost = max(0.0, self.entry_total_cost - cost_basis)
        self.partial_exits.append(
            {
                "timestamp": timestamp,
                "reason": reason,
                "price": round(adjusted_exit_price, 2),
                "position_fraction": round(fraction_of_initial, 2),
                "position_size": close_size,
                "pnl": round(pnl, 2),
                "return_pct": round((pnl / cost_basis) * 100, 2) if cost_basis > 0 else 0.0,
            }
        )

    def _runner_exit_reason(self, signal: dict[str, Any]) -> str | None:
        return self.engine.runner_exit_reason(signal, self.runner_highest_close)

    def _update_runner_high(self, price: float) -> None:
        self.highest_price = max(self.highest_price, price)
        if self.runner_active:
            self.runner_highest_close = max(self.runner_highest_close, price)

    def _close_long(self, timestamp: str, price: float, exit_reason: str, signal: dict[str, Any]) -> None:
        if self.entry_price is None or self.entry_timestamp is None:
            return
        adjusted_exit_price = price * (1 - self.slippage_rate)
        gross_proceeds = self.position_size * adjusted_exit_price
        exit_fee = gross_proceeds * self.fee_rate
        net_proceeds = gross_proceeds - exit_fee
        final_pnl = net_proceeds - self.entry_total_cost
        total_pnl = self.partial_realized_pnl + final_pnl
        original_cost = self.original_entry_total_cost if self.original_entry_total_cost > 0 else self.entry_total_cost
        return_pct = (total_pnl / original_cost) * 100 if original_cost > 0 else 0.0
        runner_return_pct = ((adjusted_exit_price / self.entry_price) - 1) * 100 if self.runner_active else None
        runner_holding_hours = self._runner_holding_hours(pd.Timestamp(timestamp)) if self.runner_active else None
        max_available_gain = max(0.0, ((self.highest_price / self.entry_price) - 1) * 100)
        captured_gain = self._weighted_captured_gain_pct(adjusted_exit_price)
        profit_capture_ratio = captured_gain / max_available_gain if max_available_gain > 0 else None
        trend_state = determine_trend_state(signal).value

        self.trades.append(
            Trade(
                entry_timestamp=self.entry_timestamp,
                exit_timestamp=timestamp,
                entry_price=self.entry_price,
                exit_price=adjusted_exit_price,
                position_size=self.initial_position_size,
                entry_fee=self.entry_fee,
                exit_fee=exit_fee,
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
                entry_stop_type=str(self.entry_audit.get("entry_stop_type", "FIXED")),
                entry_stop_distance_pct=self.entry_audit.get("entry_stop_distance_pct"),
                entry_stop_distance_atr=self.entry_audit.get("entry_stop_distance_atr"),
                entry_atr=self.entry_audit.get("entry_atr"),
                entry_stop_candidates=self.entry_audit.get("entry_stop_candidates"),
                r_multiple=self._r_multiple(adjusted_exit_price),
                max_price=self.highest_price,
                trailing_stop=self.engine.runner_trailing_stop(self.runner_highest_close),
                rejected_entry_reasons=list(self.entry_audit.get("rejected_entry_reasons", [])),
                runner_return_pct=round(runner_return_pct, 2) if runner_return_pct is not None else None,
                runner_holding_hours=round(runner_holding_hours, 2) if runner_holding_hours is not None else None,
                profit_capture_ratio=round(profit_capture_ratio, 4) if profit_capture_ratio is not None else None,
                tp1_price=_round_or_none(self.tp1_price),
                tp2_price=_round_or_none(self.tp2_price),
                runner_exit_price=round(adjusted_exit_price, 2) if self.runner_active else None,
                entry_trend_state=str(self.entry_audit.get("entry_trend_state", "")),
                exit_trend_state=trend_state,
                partial_exits=self.partial_exits.copy(),
            )
        )
        if self.runner_active and exit_reason in self.runner_exit_reasons:
            self.runner_exit_reasons[exit_reason] += 1
        self.cash += net_proceeds
        self.realized_pnl += final_pnl
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
        self._reset_trend_holding_state()

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

    def execution_summary(self, total_decisions: int) -> dict[str, Any]:
        summary = super().execution_summary(total_decisions)
        runner_returns = [
            float(trade.runner_return_pct)
            for trade in self.trades
            if trade.runner_return_pct is not None
        ]
        runner_hours = [
            float(trade.runner_holding_hours)
            for trade in self.trades
            if trade.runner_holding_hours is not None
        ]
        summary.update(
            {
                "tp1_hits": self.tp1_hit_count,
                "tp2_hits": self.tp2_hit_count,
                "runner_activations": self.runner_activation_count,
                "tp1_hit_count": self.tp1_hit_count,
                "tp2_hit_count": self.tp2_hit_count,
                "runner_activation_count": self.runner_activation_count,
                "runner_exit_reasons": self.runner_exit_reasons.copy(),
                "average_runner_return_pct": _round_or_none(_mean(runner_returns)),
                "max_runner_return_pct": _round_or_none(max(runner_returns) if runner_returns else None),
                "average_runner_holding_hours": _round_or_none(_mean(runner_hours)),
                "median_runner_holding_hours": _round_or_none(_median(runner_hours)),
            }
        )
        return summary

    def _runner_holding_hours(self, timestamp: pd.Timestamp) -> float | None:
        if self.runner_activation_timestamp is None:
            return None
        return (timestamp - pd.Timestamp(self.runner_activation_timestamp)).total_seconds() / 3600

    def _weighted_captured_gain_pct(self, final_exit_price: float) -> float:
        if self.entry_price is None:
            return 0.0
        captured = 0.0
        closed_fraction = 0.0
        for partial_exit in self.partial_exits:
            fraction = max(0.0, float(partial_exit.get("position_fraction", 0.0)))
            partial_price = float(partial_exit.get("price", final_exit_price))
            captured += fraction * max(0.0, ((partial_price / self.entry_price) - 1) * 100)
            closed_fraction += fraction
        runner_fraction = max(0.0, 1.0 - closed_fraction)
        captured += runner_fraction * max(0.0, ((final_exit_price / self.entry_price) - 1) * 100)
        return captured

    def _reset_trend_holding_state(self) -> None:
        self.initial_position_size = 0.0
        self.original_entry_total_cost = 0.0
        self.partial_realized_pnl = 0.0
        self.partial_exits: list[dict[str, Any]] = []
        self.tp1_done = False
        self.tp2_done = False
        self.runner_active = False
        self.runner_activation_timestamp: str | None = None
        self.runner_highest_close = 0.0
        self.tp1_price: float | None = None
        self.tp2_price: float | None = None

    @staticmethod
    def _format_signal_timestamp(signal: dict[str, Any]) -> str:
        timestamp = pd.Timestamp(signal["timestamp"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        return timestamp.isoformat()


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)
