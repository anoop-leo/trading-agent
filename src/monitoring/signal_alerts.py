"""Daily job: surface short-term technical signals to Telegram, READ-ONLY.

This is an observation channel for the Phase-1 entry-quality gates -- it posts
entry_decision / trend_bias / watch_levels / alert_summary so the new gated output
can be watched in Telegram WITHOUT touching any execution path. It places no
orders, proposes no sizing, and runs the signal engine via compute_signal() so it
has no file/chart/journal side effects.

Cadence: daily, actionable-only -- a symbol is alerted only when its
entry_decision is BUY or WAIT_FOR_PULLBACK, or an entry-quality gate fired. Alerts
are de-duped so an unchanged state does not re-fire every day.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from monitoring.monitoring_config import load_monitoring_config
from notify.telegram import send_telegram_message
from risk.alert_state import (
    DEFAULT_ALERT_STATE_PATH,
    evaluate_signal_state_alert,
    load_alert_state,
    save_alert_state,
)


DEFAULT_SIGNAL_ALERTS_PATH = Path("data/signal_alerts.json")
ACTIONABLE_ENTRY_DECISIONS = {"BUY", "WAIT_FOR_PULLBACK"}
DEFAULT_BETWEEN_SYMBOL_SLEEP_SECONDS = 1.0


def is_actionable(payload: dict[str, Any]) -> bool:
    """Actionable = a tradeable entry decision, or a gate downgraded a BUY."""

    return payload.get("entry_decision") in ACTIONABLE_ENTRY_DECISIONS or payload.get("gate_triggered") is not None


def signal_state_value(payload: dict[str, Any]) -> str:
    return f"{payload.get('entry_decision')}|{payload.get('gate_triggered')}"


def format_signal_alert(payload: dict[str, Any]) -> str:
    symbol = payload.get("symbol")
    entry_decision = payload.get("entry_decision")
    trend_bias = payload.get("trend_bias")
    gate = payload.get("gate_triggered")
    summary = payload.get("alert_summary")
    if summary:
        head = summary
    else:
        reason = payload.get("final_decision_reason") or ""
        head = f"{symbol} — {trend_bias} trend, entry {entry_decision}. {reason}".strip()
    if gate:
        head += f" (gate: {gate})"
    return head


def _signal_record(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": payload.get("symbol"),
        "pre_gate_decision": payload.get("pre_gate_decision"),
        "final_decision": payload.get("final_decision"),
        "entry_decision": payload.get("entry_decision"),
        "gate_triggered": payload.get("gate_triggered"),
        "trend_bias": payload.get("trend_bias"),
        "watch_levels": payload.get("watch_levels"),
        "alert_summary": payload.get("alert_summary"),
        "actionable": is_actionable(payload),
    }


def _default_signal_fn(symbol: str) -> dict[str, Any]:
    from trading_agent.config import AgentConfig
    from trading_agent.main import compute_signal

    return compute_signal(AgentConfig(symbol=symbol))


def run_signal_alerts_job(
    monitoring_config_path: Path | None = None,
    alert_state_path: Path = DEFAULT_ALERT_STATE_PATH,
    signal_alerts_path: Path = DEFAULT_SIGNAL_ALERTS_PATH,
    signal_fn: Callable[[str], dict[str, Any]] = _default_signal_fn,
    sleep_fn: Callable[[float], None] = time.sleep,
    between_symbol_sleep_seconds: float = DEFAULT_BETWEEN_SYMBOL_SLEEP_SECONDS,
    send_alerts: bool = True,
) -> dict[str, Any]:
    config = load_monitoring_config(monitoring_config_path) if monitoring_config_path else load_monitoring_config()
    alert_state = load_alert_state(alert_state_path)
    state = dict(alert_state)

    signals: dict[str, Any] = {}
    errors: dict[str, str] = {}
    alerts: list[str] = []

    for index, symbol in enumerate(config.signal_alert_symbols):
        if index > 0 and between_symbol_sleep_seconds > 0:
            sleep_fn(between_symbol_sleep_seconds)
        try:
            payload = signal_fn(symbol)
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not kill the scan
            errors[symbol] = str(exc)
            continue
        signals[symbol] = _signal_record(payload)
        actionable = is_actionable(payload)
        fired, state = evaluate_signal_state_alert(state, symbol, signal_state_value(payload), actionable)
        if fired:
            alerts.append(format_signal_alert(payload))

    save_alert_state(state, alert_state_path)

    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "signals": signals,
        "alerts": alerts,
        "errors": errors,
    }
    signal_alerts_path = Path(signal_alerts_path)
    signal_alerts_path.parent.mkdir(parents=True, exist_ok=True)
    signal_alerts_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    sent = []
    if send_alerts:
        for alert in alerts:
            sent.append(send_telegram_message(f"[Signal] {alert}"))
    result["alerts_sent"] = sent
    return result


if __name__ == "__main__":
    print(json.dumps(run_signal_alerts_job(), indent=2, default=str))
