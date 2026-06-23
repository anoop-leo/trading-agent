"""Builds and sends the one daily digest message. Read-only summary -- no
recommendation to buy or sell anything, just the current readout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from monitoring.daily_scan import DEFAULT_WATCHLIST_SCORES_PATH
from monitoring.monitoring_config import MonitoringConfig, load_monitoring_config
from notify.telegram import send_telegram_message
from risk.build_portfolio_state import DEFAULT_HOLDINGS_PATH, load_holdings
from risk.equity_history import DEFAULT_EQUITY_HISTORY_PATH, find_point_near, load_equity_history
from risk.portfolio_state import DEFAULT_PORTFOLIO_STATE_PATH, PortfolioState, load_portfolio_state
from risk.risk_config import DEFAULT_RISK_CONFIG_PATH, RiskEngineConfig, load_risk_config


CRO_BLIND_SPOT_NOTE = (
    "Note: speculative bucket excludes ~4,000 CRO (unfetchable on Binance/Bybit) -- "
    "true speculative value is understated, not overstated."
)


def build_daily_digest_text(
    state: PortfolioState,
    risk_config: RiskEngineConfig,
    monitoring_config: MonitoringConfig,
    watchlist: dict[str, Any] | None,
    yesterday_total_value_usd: float | None,
    current_btc_quantity: float | None,
) -> str:
    """Pure: no network or filesystem access."""

    total = state.total_value_usd
    targets = risk_config.bucket_targets
    lines = ["DAILY PORTFOLIO DIGEST", ""]

    lines.append(f"Total value: ${total:,.2f}")
    if yesterday_total_value_usd:
        change_usd = total - yesterday_total_value_usd
        change_pct = change_usd / yesterday_total_value_usd * 100
        lines.append(f"24h change: {change_usd:+,.2f} ({change_pct:+.2f}%)")
    else:
        lines.append("24h change: not enough history yet")
    lines.append("")

    lines.append("Buckets (value | % of total | target | room to cap):")
    speculative_cap_usd = total * risk_config.speculative_max_pct / 100
    bucket_rows = [
        ("core", state.core_usd, targets.core_pct, None),
        ("growth", state.growth_usd, targets.growth_pct, None),
        ("speculative", state.speculative_usd, targets.speculative_pct, speculative_cap_usd),
        ("cash", state.cash_usd, targets.cash_pct, None),
    ]
    for name, value, target_pct, cap_usd in bucket_rows:
        pct = value / total * 100 if total else 0.0
        room = f"${cap_usd - value:,.2f} to cap" if cap_usd is not None else "no bucket cap"
        lines.append(f"  {name:<12} ${value:>12,.2f}  {pct:>6.2f}%  (target {target_pct:.0f}%)  {room}")
    lines.append("")

    lines.append(
        f"Drawdown: {state.drawdown_pct:.2f}% vs -{risk_config.portfolio_drawdown_circuit_breaker_pct:.0f}% breaker"
        f"{' -- TRIPPED' if state.drawdown_pct >= risk_config.portfolio_drawdown_circuit_breaker_pct else ''}"
    )
    lines.append("")

    if current_btc_quantity is not None:
        progress_pct = min(100.0, current_btc_quantity / monitoring_config.btc_core_target * 100)
        lines.append(
            f"BTC core: {current_btc_quantity} / {monitoring_config.btc_core_target} BTC ({progress_pct:.1f}% of target)"
        )
        lines.append("")

    if watchlist and watchlist.get("scores"):
        lines.append(f"Watchlist (accumulation zone >= {monitoring_config.accumulation_zone_threshold}):")
        for symbol, score in sorted(watchlist["scores"].items(), key=lambda item: item[1], reverse=True):
            distance = monitoring_config.accumulation_zone_threshold - score
            status = "IN ZONE" if distance <= 0 else f"{distance} pts away"
            band = watchlist.get("bands", {}).get(symbol, "")
            lines.append(f"  {symbol:<6} score {score:>3}  {band:<24} {status}")
        lines.append("")

    lines.append(CRO_BLIND_SPOT_NOTE)
    return "\n".join(lines)


def run_daily_digest_job(
    portfolio_state_path: Path = DEFAULT_PORTFOLIO_STATE_PATH,
    risk_config_path: Path = DEFAULT_RISK_CONFIG_PATH,
    monitoring_config_path: Path | None = None,
    holdings_path: Path = DEFAULT_HOLDINGS_PATH,
    equity_history_path: Path = DEFAULT_EQUITY_HISTORY_PATH,
    watchlist_scores_path: Path = DEFAULT_WATCHLIST_SCORES_PATH,
    send: bool = True,
) -> dict[str, Any]:
    risk_config = load_risk_config(risk_config_path)
    monitoring_config = load_monitoring_config(monitoring_config_path) if monitoring_config_path else load_monitoring_config()
    state = load_portfolio_state(portfolio_state_path, risk_config)
    history = load_equity_history(equity_history_path)

    from datetime import UTC, datetime, timedelta

    yesterday_target = (datetime.now(UTC) - timedelta(hours=24)).isoformat()
    yesterday_point = find_point_near(history, yesterday_target, tolerance_seconds=3600 * 2)
    yesterday_total = float(yesterday_point["total_value_usd"]) if yesterday_point else None

    current_btc_quantity = None
    try:
        holdings = load_holdings(holdings_path)
        for holding in holdings["holdings"]:
            if holding["symbol"] == "BTC":
                current_btc_quantity = holding["quantity"]
                break
    except (OSError, FileNotFoundError, KeyError):
        pass

    watchlist = None
    watchlist_scores_path = Path(watchlist_scores_path)
    if watchlist_scores_path.exists():
        watchlist = json.loads(watchlist_scores_path.read_text())

    text = build_daily_digest_text(state, risk_config, monitoring_config, watchlist, yesterday_total, current_btc_quantity)

    sent = False
    if send:
        sent = send_telegram_message(text)

    return {"text": text, "sent": sent}


if __name__ == "__main__":
    result = run_daily_digest_job()
    print(result["text"])
    print()
    print(f"Telegram sent: {result['sent']}")
