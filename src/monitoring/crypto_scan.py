"""Daily job: re-score the crypto book for long-term accumulation and fire an
alert when a name crosses into its accumulation zone.

This reuses the existing investor agents wholesale -- it does NOT reimplement any
scoring. BTC goes through investor_agent.py (MVRV / fear-greed / cycle phase /
institutional overlay); every other name goes through crypto_investor_agent.py.
The monitor only reads their existing score + band output.

Cadence: MVRV (CoinMetrics community API) and fear/greed (alternative.me) are
daily-resolution metrics, so this belongs in the daily job, not the hourly one.

Read-only: it scores and notifies. It places/simulates nothing and never writes
a buy to portfolio_state.json. The only write is data/crypto_scores.json plus the
de-dupe alert state. Cap headroom for any in-zone name is checked by routing a
probe buy through the live risk engine -- the same gate real recommendations use.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from decision.investor_recommendation import bucket_for_crypto_symbol
from decision.recommendation import BUCKETS, PositionRecommendation
from monitoring.monitoring_config import load_monitoring_config
from notify.telegram import send_telegram_message
from risk.alert_state import (
    DEFAULT_ALERT_STATE_PATH,
    evaluate_crypto_accumulation_crossing,
    load_alert_state,
    save_alert_state,
)
from risk.build_portfolio_state import DEFAULT_HOLDINGS_PATH, load_holdings
from risk.live_risk_engine import LiveRiskEngine
from risk.portfolio_state import DEFAULT_PORTFOLIO_STATE_PATH, PortfolioState, load_portfolio_state
from risk.risk_config import DEFAULT_RISK_CONFIG_PATH, RiskEngineConfig, load_risk_config


DEFAULT_CRYPTO_SCORES_PATH = Path("data/crypto_scores.json")
DEFAULT_BETWEEN_SYMBOL_SLEEP_SECONDS = 1.0

DATA_SOURCE_NOTES = (
    "BTC valuation uses CoinMetrics community MVRV (keyless, rate-limited ~10 req/6s) and "
    "alternative.me fear/greed (keyless); both are daily-resolution, fetched once per day."
)


@dataclass(frozen=True)
class CryptoScoreInput:
    symbol: str
    role: str  # "core" | "held" | "watchlist"
    bucket: str
    held: bool
    score: int
    band: str
    confidence: str
    drivers: dict[str, Any] | None  # BTC only (MVRV / fear-greed / cycle phase)


def _resolve_bucket(symbol: str, role: str, holdings_buckets: dict[str, Any]) -> str:
    """Bucket a name actually belongs to. Held coins use the bucket recorded in
    holdings.json (the source of truth); BTC is core; not-held watchlist names
    fall back to the generic symbol->bucket map."""

    if role == "core":
        return "core"
    held_bucket = holdings_buckets.get(symbol.upper())
    if held_bucket in BUCKETS:
        return str(held_bucket)
    return bucket_for_crypto_symbol(symbol)


def _cap_headroom(engine: LiveRiskEngine, state: PortfolioState, bucket: str, symbol: str) -> tuple[float, str]:
    """Ask the live risk engine how much could be added to ``bucket`` right now by
    probing a deliberately oversized buy and reading what it would approve. This
    routes the cap/circuit-breaker check through the one gate that owns it rather
    than recomputing cap math here."""

    probe = PositionRecommendation(
        symbol=symbol,
        asset_class="crypto",
        bucket=bucket,
        action="buy",
        conviction_score=1.0,
        suggested_size_usd=state.total_value_usd,
        rationale="cap headroom probe (no order placed)",
        source_agent="crypto_monitor",
    )
    decision = engine.evaluate(probe, state)
    room = decision.approved_size_usd
    if room <= 0:
        return 0.0, f"{bucket} bucket cannot take a buy now -- {decision.reason}"
    return room, f"{bucket} bucket has ${room:,.0f} of room under caps"


def _format_crypto_alert(item: CryptoScoreInput, cap_note: str) -> str:
    if item.role == "watchlist":
        head = (
            f"{item.symbol} (NOT HELD) entry zone looks good: investor score {item.score} "
            f"(>= zone), band {item.band}."
        )
    else:
        head = (
            f"{item.symbol} crossed into the accumulation zone: investor score {item.score}, "
            f"band {item.band}."
        )
    parts = [head]
    if item.drivers:
        drivers = item.drivers
        parts.append(
            f"Drivers: MVRV {drivers.get('mvrv')}, fear/greed {drivers.get('fear_greed')}, "
            f"cycle {drivers.get('cycle_phase')}."
        )
    parts.append(cap_note + ".")
    return " ".join(parts)


def compute_crypto_accumulation(
    inputs: list[CryptoScoreInput],
    alert_state: dict[str, Any],
    risk_config: RiskEngineConfig,
    portfolio_state: PortfolioState,
    accumulation_zone_threshold: int = 70,
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, Any]]:
    """Pure: no network or filesystem access. Returns (entries, alerts, new_state)."""

    engine = LiveRiskEngine(risk_config)
    entries: dict[str, dict[str, Any]] = {}
    alerts: list[str] = []
    state = dict(alert_state)

    for item in inputs:
        in_zone = item.score >= accumulation_zone_threshold
        entry: dict[str, Any] = {
            "symbol": item.symbol,
            "role": item.role,
            "bucket": item.bucket,
            "held": item.held,
            "score": item.score,
            "band": item.band,
            "confidence": item.confidence,
            "distance_to_zone": max(0, accumulation_zone_threshold - item.score),
            "in_zone": in_zone,
            "drivers": item.drivers,
        }
        if in_zone:
            room_usd, cap_note = _cap_headroom(engine, portfolio_state, item.bucket, item.symbol)
            entry["cap_room_usd"] = room_usd
            entry["cap_note"] = cap_note

        crossed, state = evaluate_crypto_accumulation_crossing(
            state, item.symbol, item.score, accumulation_zone_threshold
        )
        if crossed:
            alerts.append(_format_crypto_alert(item, entry["cap_note"]))

        entries[item.symbol] = entry

    return entries, alerts, state


def _default_btc_scorer(symbol: str, offline: bool) -> dict[str, Any]:
    from agents.investor_agent import InvestorAgentConfig, run_investor_agent

    payload = run_investor_agent(InvestorAgentConfig(symbol="BTC", offline=offline))
    factor_scores = payload.get("factor_scores", {})
    mvrv = (factor_scores.get("mvrv") or {}).get("value")
    return {
        "score": int(payload.get("accumulation_score", 0)),
        "band": str(payload.get("accumulation_band", "")),
        "confidence": str(payload.get("data_quality", {}).get("confidence", "")),
        "drivers": {
            "mvrv": round(mvrv, 2) if isinstance(mvrv, (int, float)) else mvrv,
            "fear_greed": (factor_scores.get("fear_and_greed") or {}).get("value"),
            "cycle_phase": (payload.get("cycle_overlay") or {}).get("cycle_phase"),
        },
    }


def _default_alt_scorer(symbol: str, offline: bool) -> dict[str, Any]:
    from agents.crypto_investor_agent import CryptoInvestorConfig, run_crypto_investor_agent

    payload = run_crypto_investor_agent(CryptoInvestorConfig(symbol=symbol, offline=offline))
    return {
        "score": int(payload.get("investor_score", 0)),
        "band": str(payload.get("investor_band", "")),
        "confidence": str(payload.get("data_quality", {}).get("confidence", "")),
        "drivers": None,
    }


def run_crypto_accumulation_scan_job(
    monitoring_config_path: Path | None = None,
    alert_state_path: Path = DEFAULT_ALERT_STATE_PATH,
    risk_config_path: Path = DEFAULT_RISK_CONFIG_PATH,
    portfolio_state_path: Path = DEFAULT_PORTFOLIO_STATE_PATH,
    holdings_path: Path = DEFAULT_HOLDINGS_PATH,
    crypto_scores_path: Path = DEFAULT_CRYPTO_SCORES_PATH,
    btc_scorer: Callable[[str, bool], dict[str, Any]] = _default_btc_scorer,
    alt_scorer: Callable[[str, bool], dict[str, Any]] = _default_alt_scorer,
    sleep_fn: Callable[[float], None] = time.sleep,
    between_symbol_sleep_seconds: float = DEFAULT_BETWEEN_SYMBOL_SLEEP_SECONDS,
    offline: bool = False,
    send_alerts: bool = True,
) -> dict[str, Any]:
    config = load_monitoring_config(monitoring_config_path) if monitoring_config_path else load_monitoring_config()
    risk_config = load_risk_config(risk_config_path)
    portfolio_state = load_portfolio_state(portfolio_state_path, risk_config)
    alert_state = load_alert_state(alert_state_path)

    # data/holdings.json is the single source of truth for two things here: the
    # blind-spot "excluded" note, and the bucket each held coin actually sits in
    # (LINK/XRP are growth there, not speculative as the generic symbol map assumes).
    try:
        holdings = load_holdings(holdings_path)
    except (OSError, FileNotFoundError, KeyError, ValueError):
        holdings = {}
    excluded = holdings.get("excluded", [])
    holdings_buckets = {
        str(item["symbol"]).upper(): item.get("bucket")
        for item in holdings.get("holdings", [])
    }

    plan: list[tuple[str, str, bool, Callable[[str, bool], dict[str, Any]]]] = [
        (config.crypto_core_symbol, "core", True, btc_scorer)
    ]
    plan.extend((symbol, "held", True, alt_scorer) for symbol in config.crypto_held_symbols)
    plan.extend((symbol, "watchlist", False, alt_scorer) for symbol in config.crypto_watchlist_symbols)

    inputs: list[CryptoScoreInput] = []
    errors: dict[str, str] = {}
    for index, (symbol, role, held, scorer) in enumerate(plan):
        if index > 0 and between_symbol_sleep_seconds > 0:
            sleep_fn(between_symbol_sleep_seconds)
        try:
            scored = scorer(symbol, offline)
            inputs.append(
                CryptoScoreInput(
                    symbol=symbol,
                    role=role,
                    bucket=_resolve_bucket(symbol, role, holdings_buckets),
                    held=held,
                    score=int(scored["score"]),
                    band=str(scored.get("band", "")),
                    confidence=str(scored.get("confidence", "")),
                    drivers=scored.get("drivers"),
                )
            )
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not kill the scan
            errors[symbol] = str(exc)

    entries, alerts, new_alert_state = compute_crypto_accumulation(
        inputs, alert_state, risk_config, portfolio_state, config.accumulation_zone_threshold
    )
    save_alert_state(new_alert_state, alert_state_path)

    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "accumulation_zone_threshold": config.accumulation_zone_threshold,
        "scores": entries,
        "errors": errors,
        "alerts": alerts,
        "excluded": excluded,
        "data_source_notes": DATA_SOURCE_NOTES,
    }
    crypto_scores_path = Path(crypto_scores_path)
    crypto_scores_path.parent.mkdir(parents=True, exist_ok=True)
    crypto_scores_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    sent = []
    if send_alerts:
        for alert in alerts:
            sent.append(send_telegram_message(f"[Crypto] {alert}"))
    result["alerts_sent"] = sent
    return result


if __name__ == "__main__":
    print(json.dumps(run_crypto_accumulation_scan_job(), indent=2, default=str))
