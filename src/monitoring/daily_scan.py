"""Daily job: re-score the equity watchlist (rate-limited, spaced Alpha Vantage
calls) and fire an alert for any name that crosses into the accumulation zone.

Read-only: proposes nothing, places no orders.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from agents.equity_investor_agent import EquityInvestorConfig, run_equity_investor_agent
from monitoring.monitoring_config import MonitoringConfig, load_monitoring_config
from notify.telegram import send_telegram_message
from risk.alert_state import (
    DEFAULT_ALERT_STATE_PATH,
    evaluate_watchlist_accumulation_alert,
    load_alert_state,
    save_alert_state,
)


DEFAULT_WATCHLIST_SCORES_PATH = Path("data/watchlist_scores.json")
DEFAULT_BETWEEN_SYMBOL_SLEEP_SECONDS = 5.0


def compute_watchlist_alerts(
    scores: dict[str, int],
    alert_state: dict[str, Any],
    accumulation_zone_threshold: int = 70,
) -> tuple[list[str], dict[str, Any]]:
    """Pure: no network or filesystem access."""

    alerts: list[str] = []
    state = dict(alert_state)
    for symbol, score in scores.items():
        message, state = evaluate_watchlist_accumulation_alert(state, symbol, score, accumulation_zone_threshold)
        if message:
            alerts.append(message)
    return alerts, state


def run_daily_scan_job(
    monitoring_config_path: Path | None = None,
    alert_state_path: Path = DEFAULT_ALERT_STATE_PATH,
    watchlist_scores_path: Path = DEFAULT_WATCHLIST_SCORES_PATH,
    sleep_fn: Callable[[float], None] = time.sleep,
    between_symbol_sleep_seconds: float = DEFAULT_BETWEEN_SYMBOL_SLEEP_SECONDS,
    send_alerts: bool = True,
) -> dict[str, Any]:
    config = load_monitoring_config(monitoring_config_path) if monitoring_config_path else load_monitoring_config()
    alert_state = load_alert_state(alert_state_path)

    scores: dict[str, int] = {}
    bands: dict[str, str] = {}
    errors: dict[str, str] = {}
    for index, symbol in enumerate(config.watchlist_symbols):
        if index > 0 and between_symbol_sleep_seconds > 0:
            sleep_fn(between_symbol_sleep_seconds)
        try:
            payload = run_equity_investor_agent(EquityInvestorConfig(symbol=symbol, bucket="growth"))
            scores[symbol] = payload["investor_score"]
            bands[symbol] = payload["investor_band"]
        except Exception as exc:  # noqa: BLE001 - one bad symbol should not kill the scan
            errors[symbol] = str(exc)

    alerts, new_alert_state = compute_watchlist_alerts(scores, alert_state, config.accumulation_zone_threshold)
    save_alert_state(new_alert_state, alert_state_path)

    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "scores": scores,
        "bands": bands,
        "errors": errors,
        "alerts": alerts,
    }
    watchlist_scores_path = Path(watchlist_scores_path)
    watchlist_scores_path.parent.mkdir(parents=True, exist_ok=True)
    watchlist_scores_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    sent = []
    if send_alerts:
        for alert in alerts:
            sent.append(send_telegram_message(f"[Watchlist] {alert}"))
    result["alerts_sent"] = sent
    return result


if __name__ == "__main__":
    print(json.dumps(run_daily_scan_job(), indent=2, default=str))
