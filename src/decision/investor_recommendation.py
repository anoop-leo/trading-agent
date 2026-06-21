"""Build a PositionRecommendation from existing BTC/crypto investor agent payloads.

The BTC and crypto investor agents predate the PositionRecommendation contract
and don't size positions or assign portfolio buckets themselves. This module
is CLI-facing glue: it reads their existing output fields and proposes a
recommendation, exactly like the Equity Investor Agent does internally.
"""

from __future__ import annotations

import re
from typing import Any

from decision.recommendation import PositionRecommendation


CRYPTO_BUCKET_BY_SYMBOL = {
    "BTC": "core",
    "ETH": "growth",
    "SOL": "growth",
    "XRP": "speculative",
    "AVAX": "speculative",
    "LINK": "speculative",
    "ONDO": "speculative",
    "HYPE": "speculative",
}


def bucket_for_crypto_symbol(symbol: str) -> str:
    return CRYPTO_BUCKET_BY_SYMBOL.get(symbol.upper(), "speculative")


def build_crypto_position_recommendation(
    payload: dict[str, Any],
    default_position_usd: float = 2_000.0,
) -> PositionRecommendation:
    symbol = str(payload["symbol"]).upper()
    bucket = bucket_for_crypto_symbol(symbol)
    score = payload.get("accumulation_score", payload.get("investor_score", 0))
    conviction_score = round(max(0.0, min(100.0, float(score))) / 100, 4)
    final_action = str(payload.get("final_investor_action", ""))
    action = "hold" if "DO_NOT_ACCUMULATE" in final_action else "buy"
    multiplier_text = str(
        payload.get("confidence_adjusted_dca_multiplier")
        or payload.get("cycle_adjusted_dca_multiplier")
        or payload.get("suggested_dca_multiplier")
        or "0x normal DCA"
    )
    multiplier = _parse_dca_multiplier_midpoint(multiplier_text)
    suggested_size_usd = round(default_position_usd * multiplier, 2) if action == "buy" else 0.0

    return PositionRecommendation(
        symbol=symbol,
        asset_class="crypto",
        bucket=bucket,
        action=action,
        conviction_score=conviction_score,
        suggested_size_usd=suggested_size_usd,
        rationale=f"{symbol} {payload.get('agent', 'CRYPTO_INVESTOR')} action is {final_action or 'UNKNOWN'} ({multiplier_text}).",
        source_agent=str(payload.get("agent", "crypto_investor_agent")).lower(),
    )


def _parse_dca_multiplier_midpoint(text: str) -> float:
    values = [float(match) for match in re.findall(r"(\d+(?:\.\d+)?)x", text)]
    if not values:
        return 0.0
    return sum(values) / len(values)
