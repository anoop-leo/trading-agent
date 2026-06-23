"""Hourly job: regenerate portfolio value from live crypto prices, append to
equity history (the real drawdown peak source), and fire any event alerts.

Read-only: places no orders, writes no buy/sell to portfolio_state.json.
The only writes are the regenerated valuation snapshot, the equity-history
append, and de-dupe alert state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from monitoring.monitoring_config import MonitoringConfig, load_monitoring_config
from notify.telegram import send_telegram_message
from risk.alert_state import (
    DEFAULT_ALERT_STATE_PATH,
    evaluate_bucket_cap_alert,
    evaluate_drawdown_alert,
    evaluate_position_move_alert,
    load_alert_state,
    save_alert_state,
)
from risk.build_portfolio_state import (
    DEFAULT_HOLDINGS_PATH,
    build_portfolio_state_from_holdings,
    load_holdings,
)
from risk.equity_history import (
    DEFAULT_EQUITY_HISTORY_PATH,
    append_equity_history_point,
    compute_real_peak_value_usd,
    find_point_near,
    load_equity_history,
)
from risk.portfolio_state import DEFAULT_PORTFOLIO_STATE_PATH, PortfolioState, save_portfolio_state
from risk.risk_config import DEFAULT_RISK_CONFIG_PATH, RiskEngineConfig, load_risk_config
from trading_agent.data import BinanceKlineProvider


@dataclass(frozen=True)
class HourlySnapshotResult:
    state: PortfolioState
    drawdown_pct: float
    alerts: list[str]
    history_point: dict[str, Any]


def compute_hourly_update(
    raw_state: PortfolioState,
    risk_config: RiskEngineConfig,
    monitoring_config: MonitoringConfig,
    history: list[dict[str, Any]],
    alert_state: dict[str, Any],
    prices: dict[str, float],
    now_iso: str,
) -> tuple[HourlySnapshotResult, dict[str, Any]]:
    """Pure: no network or filesystem access. Returns (result, new_alert_state)."""

    real_peak = compute_real_peak_value_usd(history, raw_state.total_value_usd)
    state = PortfolioState(
        total_value_usd=raw_state.total_value_usd,
        peak_value_usd=real_peak,
        cash_usd=raw_state.cash_usd,
        core_usd=raw_state.core_usd,
        growth_usd=raw_state.growth_usd,
        speculative_usd=raw_state.speculative_usd,
    )
    drawdown_pct = state.drawdown_pct
    working_alert_state = dict(alert_state)
    alerts: list[str] = []

    message, working_alert_state = evaluate_drawdown_alert(working_alert_state, drawdown_pct)
    if message:
        alerts.append(message)

    speculative_cap_usd = state.total_value_usd * risk_config.speculative_max_pct / 100
    message, working_alert_state = evaluate_bucket_cap_alert(
        working_alert_state, "speculative", state.speculative_usd, speculative_cap_usd,
        monitoring_config.bucket_near_cap_fraction,
    )
    if message:
        alerts.append(message)

    target_time = (datetime.fromisoformat(now_iso) - timedelta(hours=24)).isoformat()
    yesterday_point = find_point_near(history, target_time, tolerance_seconds=3600 * 2)
    if yesterday_point is not None:
        yesterday_prices = yesterday_point.get("prices", {})
        today_date = now_iso[:10]
        for symbol, price in prices.items():
            old_price = yesterday_prices.get(symbol)
            if not old_price:
                continue
            move_pct = (price - old_price) / old_price * 100
            message, working_alert_state = evaluate_position_move_alert(
                working_alert_state, symbol, move_pct, monitoring_config.position_daily_move_pct_threshold, today_date,
            )
            if message:
                alerts.append(message)

    history_point = {
        "timestamp": now_iso,
        "total_value_usd": state.total_value_usd,
        "core_usd": state.core_usd,
        "growth_usd": state.growth_usd,
        "speculative_usd": state.speculative_usd,
        "cash_usd": state.cash_usd,
        "prices": prices,
    }
    return HourlySnapshotResult(state=state, drawdown_pct=drawdown_pct, alerts=alerts, history_point=history_point), working_alert_state


def run_hourly_snapshot_job(
    holdings_path: Path = DEFAULT_HOLDINGS_PATH,
    portfolio_state_path: Path = DEFAULT_PORTFOLIO_STATE_PATH,
    equity_history_path: Path = DEFAULT_EQUITY_HISTORY_PATH,
    alert_state_path: Path = DEFAULT_ALERT_STATE_PATH,
    risk_config_path: Path = DEFAULT_RISK_CONFIG_PATH,
    monitoring_config_path: Path | None = None,
    market_data_provider: BinanceKlineProvider | None = None,
    send_alerts: bool = True,
) -> dict[str, Any]:
    risk_config = load_risk_config(risk_config_path)
    monitoring_config = load_monitoring_config(monitoring_config_path) if monitoring_config_path else load_monitoring_config()
    holdings = load_holdings(holdings_path)
    provider = market_data_provider or BinanceKlineProvider()

    raw_state, diagnostics = build_portfolio_state_from_holdings(holdings, provider)
    prices = {item["symbol"]: item["price_usd"] for item in diagnostics["priced_holdings"]}
    history = load_equity_history(equity_history_path)
    alert_state = load_alert_state(alert_state_path)
    now_iso = datetime.now(UTC).isoformat()

    result, new_alert_state = compute_hourly_update(
        raw_state, risk_config, monitoring_config, history, alert_state, prices, now_iso,
    )

    save_portfolio_state(result.state, portfolio_state_path)
    append_equity_history_point(result.history_point, equity_history_path)
    save_alert_state(new_alert_state, alert_state_path)

    sent = []
    if send_alerts:
        for alert in result.alerts:
            sent.append(send_telegram_message(f"[Hourly] {alert}"))

    return {
        "total_value_usd": result.state.total_value_usd,
        "peak_value_usd": result.state.peak_value_usd,
        "drawdown_pct": result.drawdown_pct,
        "alerts": result.alerts,
        "alerts_sent": sent,
        "excluded": diagnostics.get("excluded", []),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_hourly_snapshot_job(), indent=2, default=str))
