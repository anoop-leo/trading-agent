"""Trend Rider exit engine for Phase 1.9 backtests."""

from __future__ import annotations

from typing import Any

import pandas as pd

from backtesting.trade_simulator import PortfolioSnapshot, Trade, TradeSimulator, _optional_float, _regime_tag


class TrendRiderSimulator(TradeSimulator):
    """Use existing entry gates with partial exits and a 25% trend runner."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._reset_trend_rider_state()

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
                runner_exit_reason = self._runner_exit_reason(signal, price) if self.tp2_done else None
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
        self.runner_highest_price = price

    def _process_profit_targets(self, timestamp: str, price: float) -> None:
        if self.entry_price is None or self.initial_risk <= 0 or self.initial_position_size <= 0:
            return
        tp1_price = self.entry_price + (2 * self.initial_risk)
        tp2_price = self.entry_price + (4 * self.initial_risk)
        if not self.tp1_done and price >= tp1_price:
            self._partial_close(timestamp, price, 0.50, "TP1_2R")
            self.tp1_done = True
        if not self.tp2_done and price >= tp2_price:
            self._partial_close(timestamp, price, 0.25, "TP2_4R")
            self.tp2_done = True

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

    def _runner_exit_reason(self, signal: dict[str, Any], price: float) -> str | None:
        daily_rsi = _optional_float(signal.get("daily_rsi"))
        if daily_rsi is not None and daily_rsi < 50:
            return "RUNNER_RSI_EXIT"
        daily_ema20 = _optional_float(signal.get("daily_ema20"))
        daily_ema50 = _optional_float(signal.get("daily_ema50"))
        if daily_ema20 is not None and daily_ema50 is not None and daily_ema20 < daily_ema50:
            return "RUNNER_EMA_EXIT"
        if self.runner_highest_price > 0 and price <= self.runner_highest_price * 0.90:
            return "RUNNER_TRAILING_STOP"
        return None

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
        runner_active = self.tp2_done
        runner_return_pct = ((adjusted_exit_price / self.entry_price) - 1) * 100 if runner_active else None
        runner_holding_hours = self._holding_hours(pd.Timestamp(timestamp)) if runner_active else None
        max_available_gain = max(0.0, ((self.highest_price / self.entry_price) - 1) * 100)
        captured_gain = max(0.0, ((adjusted_exit_price / self.entry_price) - 1) * 100)
        profit_capture_ratio = captured_gain / max_available_gain if max_available_gain > 0 else None

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
                r_multiple=self._r_multiple(adjusted_exit_price),
                max_price=self.highest_price,
                trailing_stop=self.runner_highest_price * 0.90 if self.runner_highest_price > 0 else None,
                rejected_entry_reasons=list(self.entry_audit.get("rejected_entry_reasons", [])),
                runner_return_pct=round(runner_return_pct, 2) if runner_return_pct is not None else None,
                runner_holding_hours=round(runner_holding_hours, 2) if runner_holding_hours is not None else None,
                profit_capture_ratio=round(profit_capture_ratio, 4) if profit_capture_ratio is not None else None,
                partial_exits=self.partial_exits.copy(),
            )
        )
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
        self._reset_trend_rider_state()

    def _update_runner_high(self, price: float) -> None:
        self.highest_price = max(self.highest_price, price)
        self.runner_highest_price = max(self.runner_highest_price, price)

    def _reset_trend_rider_state(self) -> None:
        self.initial_position_size = 0.0
        self.original_entry_total_cost = 0.0
        self.partial_realized_pnl = 0.0
        self.partial_exits: list[dict[str, Any]] = []
        self.tp1_done = False
        self.tp2_done = False
        self.runner_highest_price = 0.0

    @staticmethod
    def _format_signal_timestamp(signal: dict[str, Any]) -> str:
        timestamp = pd.Timestamp(signal["timestamp"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        return timestamp.isoformat()
