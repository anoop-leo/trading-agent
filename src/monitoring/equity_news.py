"""Daily job: equity news + earnings for the watchlist.

Self-contained, read-only, lives in the daily job (equity data is rate-limited).
It NEVER creates a buy. Its outputs are:
  - the earnings calendar + a T-minus-lead_days "when NOT to act" alert,
  - Tier 1 hard-news items (ratings / M&A / regulatory / filings / guidance),
  - a clearly separated Tier 2 "unverified sentiment" stream.

Earnings proximity is used elsewhere (the digest) only to SUPPRESS/caveat an
existing accumulation signal, never to amplify one.

Invariant carried over from the airdrop screener: if a source is unavailable it is
reported as unavailable -- never silently treated as an all-clear.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Callable

from data.equity_news_provider import EquityNewsProvider
from monitoring.monitoring_config import load_monitoring_config
from notify.telegram import send_telegram_message
from risk.alert_state import (
    DEFAULT_ALERT_STATE_PATH,
    evaluate_earnings_proximity_alert,
    load_alert_state,
    save_alert_state,
)


DEFAULT_EQUITY_NEWS_PATH = Path("data/equity_news.json")

TIER1_CATEGORIES = ("analyst_rating", "m_and_a", "regulatory", "sec_filing", "guidance", "earnings_news")

_RATING_KEYWORDS = (
    "price target", "upgrade", "downgrade", "initiates coverage", "reiterate", "raised to",
    "cut to", "outperform", "underperform", "overweight", "underweight", "buy rating",
    "sell rating", "neutral rating", "hold rating", "analyst",
)
_MA_KEYWORDS = ("acquire", "acquisition", "merger", "takeover", "to buy", "buyout", "stake in")
_REG_KEYWORDS = ("lawsuit", "antitrust", "ftc", "doj", "regulator", "probe", "investigation",
                 "fine", "settlement", "sanction", "subpoena")
_FILING_KEYWORDS = ("8-k", "form 4", "13d", "13g", "insider", "s-1", "files for", "prospectus")
_GUIDANCE_KEYWORDS = ("guidance", "forecast", "outlook", "warns", "preannounce", "profit warning")
_EARNINGS_KEYWORDS = ("earnings", "quarterly results", "reports results", "beats", "misses", "eps")

# We can fully serve the earnings calendar and headline-derived news. Structured
# analyst ratings and SEC Form 4/8-K feeds are NOT on Alpha Vantage's free tier --
# proposed alternatives below are NOT added (per the "propose before adding" rule).
THIN_COVERAGE_NOTE = (
    "Coverage note: structured analyst rating/price-target changes and SEC Form 4 / 8-K filings "
    "are not available on the Alpha Vantage free tier; rating/filing items here are detected from "
    "news headlines only and may be incomplete. Candidate providers if you want full coverage "
    "(NOT added — proposing first): Finnhub for analyst ratings/price targets, and SEC EDGAR "
    "(keyless) for 8-K / Form 4. Say the word and I'll wire one in."
)


def _parse_report_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def compute_upcoming_earnings(rows: list[dict[str, Any]], today: date) -> list[dict[str, Any]]:
    """Pure: attach days_until and keep future-dated reports, soonest first."""

    upcoming = []
    for row in rows:
        report = _parse_report_date(row.get("report_date"))
        if report is None:
            continue
        days_until = (report - today).days
        if days_until < 0:
            continue
        upcoming.append({
            "symbol": row.get("symbol"),
            "report_date": row.get("report_date"),
            "days_until": days_until,
            "estimate": row.get("estimate"),
        })
    return sorted(upcoming, key=lambda item: item["days_until"])


def compute_earnings_alerts(
    upcoming: list[dict[str, Any]], alert_state: dict[str, Any], lead_days: int
) -> tuple[list[str], dict[str, Any]]:
    """Pure: de-duped T-minus-lead_days earnings alerts."""

    state = dict(alert_state)
    alerts: list[str] = []
    for entry in upcoming:
        message, state = evaluate_earnings_proximity_alert(
            state, entry["symbol"], entry["report_date"], entry["days_until"], lead_days
        )
        if message:
            alerts.append(message)
    return alerts, state


def classify_news_category(title: str, topics: list[dict[str, Any]] | None) -> str | None:
    text = (title or "").lower()
    topic_names = {str(topic.get("topic", "")).lower() for topic in (topics or [])}
    if any(keyword in text for keyword in _RATING_KEYWORDS):
        return "analyst_rating"
    if "mergers & acquisitions" in topic_names or any(keyword in text for keyword in _MA_KEYWORDS):
        return "m_and_a"
    if any(keyword in text for keyword in _FILING_KEYWORDS):
        return "sec_filing"
    if any(keyword in text for keyword in _REG_KEYWORDS):
        return "regulatory"
    if any(keyword in text for keyword in _GUIDANCE_KEYWORDS):
        return "guidance"
    if "earnings" in topic_names or any(keyword in text for keyword in _EARNINGS_KEYWORDS):
        return "earnings_news"
    return None


def _relevant_symbols(item: dict[str, Any], watchlist: set[str], min_relevance: float = 0.1) -> list[str]:
    symbols = []
    for ticker_sentiment in item.get("ticker_sentiment", []):
        ticker = str(ticker_sentiment.get("ticker", "")).upper()
        if ticker not in watchlist:
            continue
        try:
            relevance = float(ticker_sentiment.get("relevance_score", 0))
        except (TypeError, ValueError):
            relevance = 0.0
        if relevance >= min_relevance:
            symbols.append(ticker)
    return symbols


def classify_news(feed: list[dict[str, Any]], watchlist: set[str]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Pure: split the feed into Tier 1 (hard news by category) and Tier 2 (sentiment)."""

    tier1: dict[str, list[dict[str, Any]]] = {category: [] for category in TIER1_CATEGORIES}
    tier2: list[dict[str, Any]] = []
    for item in feed:
        symbols = _relevant_symbols(item, watchlist)
        if not symbols:
            continue
        record = {
            "symbols": symbols,
            "title": item.get("title"),
            "source": item.get("source"),
            "url": item.get("url"),
            "time_published": item.get("time_published"),
            "sentiment": item.get("overall_sentiment_label"),
        }
        category = classify_news_category(item.get("title", ""), item.get("topics"))
        if category:
            record["category"] = category
            tier1[category].append(record)
        else:
            record["label"] = "unverified sentiment — do not trade on this alone"
            tier2.append(record)
    return tier1, tier2


def run_equity_news_job(
    monitoring_config_path: Path | None = None,
    alert_state_path: Path = DEFAULT_ALERT_STATE_PATH,
    equity_news_path: Path = DEFAULT_EQUITY_NEWS_PATH,
    provider: EquityNewsProvider | None = None,
    today: date | None = None,
    offline: bool = False,
    send_alerts: bool = True,
    between_call_sleep_seconds: float = 15.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    config = load_monitoring_config(monitoring_config_path) if monitoring_config_path else load_monitoring_config()
    provider = provider or EquityNewsProvider()
    today = today or datetime.now(UTC).date()
    watchlist = {symbol.upper() for symbol in config.watchlist_symbols}

    earnings_env = provider.fetch_earnings_calendar(config.watchlist_symbols, config.earnings_horizon, offline=offline)
    if between_call_sleep_seconds > 0:
        sleep_fn(between_call_sleep_seconds)
    news_env = provider.fetch_news_sentiment(config.watchlist_symbols, config.equity_news_limit, offline=offline)

    upcoming = compute_upcoming_earnings(earnings_env["data"] if earnings_env["available"] else [], today)
    alert_state = load_alert_state(alert_state_path)
    alerts, new_alert_state = compute_earnings_alerts(upcoming, alert_state, config.earnings_alert_lead_days)
    save_alert_state(new_alert_state, alert_state_path)

    tier1, tier2 = classify_news(news_env["data"] if news_env["available"] else [], watchlist)

    coverage_notes: list[str] = []
    if not earnings_env["available"]:
        coverage_notes.append(
            f"Earnings calendar UNAVAILABLE ({earnings_env['error']}) — cannot confirm whether a "
            "watchlist name reports soon; treat earnings timing as unknown, not all-clear."
        )
    if not news_env["available"]:
        coverage_notes.append(f"News feed UNAVAILABLE ({news_env['error']}) — no rating/filing/sentiment items this run.")
    coverage_notes.append(THIN_COVERAGE_NOTE)

    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "as_of_date": today.isoformat(),
        "earnings_available": earnings_env["available"],
        "news_available": news_env["available"],
        "upcoming_earnings_window_days": config.upcoming_earnings_window_days,
        "earnings_alert_lead_days": config.earnings_alert_lead_days,
        "earnings_caveat_days": config.earnings_caveat_days,
        "upcoming_earnings": upcoming,
        "earnings_alerts": alerts,
        "news_tier1": tier1,
        "news_tier2_sentiment": tier2,
        "coverage_notes": coverage_notes,
        "errors": {"earnings": earnings_env["error"], "news": news_env["error"]},
    }
    equity_news_path = Path(equity_news_path)
    equity_news_path.parent.mkdir(parents=True, exist_ok=True)
    equity_news_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    sent = []
    if send_alerts:
        for alert in alerts:
            sent.append(send_telegram_message(f"[Earnings] {alert}"))
    result["alerts_sent"] = sent
    return result


if __name__ == "__main__":
    print(json.dumps(run_equity_news_job(), indent=2, default=str))
