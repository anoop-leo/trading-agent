"""Builds and sends the one daily digest message. Read-only summary -- no
recommendation to buy or sell anything, just the current readout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from monitoring.crypto_scan import DEFAULT_CRYPTO_SCORES_PATH
from monitoring.daily_scan import DEFAULT_WATCHLIST_SCORES_PATH
from monitoring.equity_news import DEFAULT_EQUITY_NEWS_PATH
from monitoring.monitoring_config import MonitoringConfig, load_monitoring_config
from notify.telegram import send_telegram_message
from risk.build_portfolio_state import DEFAULT_HOLDINGS_PATH, load_holdings
from risk.equity_history import DEFAULT_EQUITY_HISTORY_PATH, find_point_near, load_equity_history
from risk.portfolio_state import DEFAULT_PORTFOLIO_STATE_PATH, PortfolioState, load_portfolio_state
from risk.risk_config import DEFAULT_RISK_CONFIG_PATH, RiskEngineConfig, load_risk_config


CRYPTO_ROLE_LABELS = (("core", "Core"), ("held", "Held alts"), ("watchlist", "Watchlist (not held)"))


def build_crypto_accumulation_lines(crypto: dict[str, Any] | None, accumulation_zone_threshold: int) -> list[str]:
    """Pure: render the crypto accumulation block, parallel to the equity watchlist."""

    if not crypto or not crypto.get("scores"):
        return []
    scores = crypto["scores"]
    threshold = crypto.get("accumulation_zone_threshold", accumulation_zone_threshold)
    lines = [f"Crypto accumulation (zone >= {threshold}):"]
    for role, label in CRYPTO_ROLE_LABELS:
        members = [entry for entry in scores.values() if entry.get("role") == role]
        if not members:
            continue
        lines.append(f"  {label}:")
        for entry in sorted(members, key=lambda item: item.get("score", 0), reverse=True):
            distance = entry.get("distance_to_zone", max(0, threshold - entry.get("score", 0)))
            status = "IN ZONE" if entry.get("in_zone") else f"{distance} pts away"
            drivers = entry.get("drivers")
            extra = ""
            if drivers:
                extra = f"  [MVRV {drivers.get('mvrv')} | F&G {drivers.get('fear_greed')} | {drivers.get('cycle_phase')}]"
            elif entry.get("in_zone") and entry.get("cap_note"):
                extra = f"  [{entry['cap_note']}]"
            lines.append(
                f"    {entry.get('symbol', ''):<6} score {entry.get('score', 0):>3}  "
                f"{entry.get('band', ''):<26} {status}{extra}"
            )
    for excluded in crypto.get("excluded", []) or []:
        lines.append(f"  Note: {excluded.get('reason', 'holding excluded (no price provider).')}")
    lines.append("")
    return lines


_TIER1_MATERIAL = (("m_and_a", "M&A"), ("regulatory", "Regulatory"), ("sec_filing", "Filing"),
                   ("guidance", "Guidance"), ("earnings_news", "Earnings news"))
_NEWS_MAX_PER_SECTION = 5


def _news_line(record: dict[str, Any], prefix: str = "") -> str:
    symbols = "/".join(record.get("symbols", [])) or "?"
    title = (record.get("title") or "").strip()
    if len(title) > 96:
        title = title[:93] + "..."
    source = record.get("source") or ""
    tail = f" ({source})" if source else ""
    return f"    [{symbols}] {prefix}{title}{tail}"


def build_equity_news_lines(equity_news: dict[str, Any] | None) -> list[str]:
    """Pure: render the equity news + earnings block for the digest."""

    if not equity_news:
        return []
    lines = ["EQUITY NEWS & EARNINGS:"]

    window = equity_news.get("upcoming_earnings_window_days", 14)
    lead = equity_news.get("earnings_alert_lead_days", 3)
    lines.append(f"  Upcoming earnings (next {window} days):")
    if not equity_news.get("earnings_available", False):
        lines.append("    earnings calendar UNAVAILABLE — timing unknown (not all-clear)")
    else:
        all_upcoming = equity_news.get("upcoming_earnings", [])
        soon = [e for e in all_upcoming if e.get("days_until", 999) <= window]
        if not soon:
            lines.append(f"    none in the next {window} days")
            for entry in all_upcoming[:3]:  # still surface the populated calendar
                lines.append(f"    next up: {entry.get('symbol', '?'):<6} {entry.get('report_date', '?')} (in {entry.get('days_until')}d)")
        for entry in soon:
            days = entry.get("days_until")
            flag = "  <- within alert window: a tranche now = a binary event" if days is not None and days <= lead else ""
            lines.append(f"    {entry.get('symbol', '?'):<6} {entry.get('report_date', '?')} (in {days}d){flag}")

    tier1 = equity_news.get("news_tier1", {}) or {}
    ratings = tier1.get("analyst_rating", [])
    if ratings:
        lines.append("  Rating / price-target changes:")
        for record in ratings[:_NEWS_MAX_PER_SECTION]:
            lines.append(_news_line(record))

    material = [(label, rec) for key, label in _TIER1_MATERIAL for rec in tier1.get(key, [])]
    if material:
        lines.append("  Material events (M&A / regulatory / filings / guidance):")
        for label, record in material[:_NEWS_MAX_PER_SECTION]:
            lines.append(_news_line(record, prefix=f"{label}: "))

    tier2 = equity_news.get("news_tier2_sentiment", [])
    if tier2:
        lines.append("  --- UNVERIFIED SENTIMENT (do not trade on this alone) ---")
        for record in tier2[:_NEWS_MAX_PER_SECTION]:
            sentiment = record.get("sentiment") or "n/a"
            lines.append(_news_line(record, prefix=f"[{sentiment}] "))

    for note in equity_news.get("coverage_notes", []) or []:
        lines.append(f"  {note}")
    lines.append("")
    return lines


def build_daily_digest_text(
    state: PortfolioState,
    risk_config: RiskEngineConfig,
    monitoring_config: MonitoringConfig,
    watchlist: dict[str, Any] | None,
    yesterday_total_value_usd: float | None,
    current_btc_quantity: float | None,
    crypto: dict[str, Any] | None = None,
    equity_news: dict[str, Any] | None = None,
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

    earnings_days = _earnings_days_lookup(equity_news)
    caveat_days = equity_news.get("earnings_caveat_days", 10) if equity_news else 10
    if watchlist and watchlist.get("scores"):
        lines.append(f"Watchlist (accumulation zone >= {monitoring_config.accumulation_zone_threshold}):")
        for symbol, score in sorted(watchlist["scores"].items(), key=lambda item: item[1], reverse=True):
            distance = monitoring_config.accumulation_zone_threshold - score
            in_zone = distance <= 0
            status = "IN ZONE" if in_zone else f"{distance} pts away"
            band = watchlist.get("bands", {}).get(symbol, "")
            caveat = _earnings_caveat(symbol, score, in_zone, earnings_days, caveat_days)
            lines.append(f"  {symbol:<6} score {score:>3}  {band:<24} {status}{caveat}")
        lines.append("")

    lines.extend(build_crypto_accumulation_lines(crypto, monitoring_config.accumulation_zone_threshold))
    lines.extend(build_equity_news_lines(equity_news))

    return "\n".join(lines).rstrip() + "\n"


def _earnings_days_lookup(equity_news: dict[str, Any] | None) -> dict[str, int]:
    """Soonest days-until-earnings per symbol, from the equity-news payload."""

    lookup: dict[str, int] = {}
    if not equity_news:
        return lookup
    for entry in equity_news.get("upcoming_earnings", []) or []:
        symbol = entry.get("symbol")
        days = entry.get("days_until")
        if symbol is None or days is None:
            continue
        if symbol not in lookup or days < lookup[symbol]:
            lookup[symbol] = days
    return lookup


def _earnings_caveat(symbol: str, score: int, in_zone: bool, earnings_days: dict[str, int], caveat_days: int) -> str:
    """Earnings proximity SUPPRESSES/caveats a signal -- it never amplifies one."""

    days = earnings_days.get(symbol)
    if days is None or days > caveat_days:
        return ""
    if in_zone:
        return f"  <- BUT earnings in {days}d — consider waiting until after (binary event)"
    return f"  (earnings in {days}d)"


def run_daily_digest_job(
    portfolio_state_path: Path = DEFAULT_PORTFOLIO_STATE_PATH,
    risk_config_path: Path = DEFAULT_RISK_CONFIG_PATH,
    monitoring_config_path: Path | None = None,
    holdings_path: Path = DEFAULT_HOLDINGS_PATH,
    equity_history_path: Path = DEFAULT_EQUITY_HISTORY_PATH,
    watchlist_scores_path: Path = DEFAULT_WATCHLIST_SCORES_PATH,
    crypto_scores_path: Path = DEFAULT_CRYPTO_SCORES_PATH,
    equity_news_path: Path = DEFAULT_EQUITY_NEWS_PATH,
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

    crypto = None
    crypto_scores_path = Path(crypto_scores_path)
    if crypto_scores_path.exists():
        crypto = json.loads(crypto_scores_path.read_text())

    equity_news = None
    equity_news_path = Path(equity_news_path)
    if equity_news_path.exists():
        equity_news = json.loads(equity_news_path.read_text())

    text = build_daily_digest_text(
        state, risk_config, monitoring_config, watchlist, yesterday_total, current_btc_quantity, crypto, equity_news
    )

    sent = False
    if send:
        sent = send_telegram_message(text)

    return {"text": text, "sent": sent}


if __name__ == "__main__":
    result = run_daily_digest_job()
    print(result["text"])
    print()
    print(f"Telegram sent: {result['sent']}")
