"""De-duped event-alert state: fire on crossing a threshold, not on every run
while a condition persists. State is a plain JSON dict, persisted between runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_ALERT_STATE_PATH = Path("data/alert_state.json")

_DRAWDOWN_TIER_RANK = {None: 0, "drawdown_10": 1, "drawdown_20": 2, "drawdown_25_approaching": 3, "drawdown_25_tripped": 4}
_DRAWDOWN_TIER_TEXT = {
    "drawdown_10": "-10%",
    "drawdown_20": "-20%",
    "drawdown_25_approaching": "approaching -25% (circuit breaker threshold)",
    "drawdown_25_tripped": "-25% -- CIRCUIT BREAKER TRIPPED",
}


def load_alert_state(path: Path = DEFAULT_ALERT_STATE_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_alert_state(state: dict[str, Any], path: Path = DEFAULT_ALERT_STATE_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    return path


def drawdown_tier_label(drawdown_pct: float) -> str | None:
    if drawdown_pct >= 25.0:
        return "drawdown_25_tripped"
    if drawdown_pct >= 23.0:
        return "drawdown_25_approaching"
    if drawdown_pct >= 20.0:
        return "drawdown_20"
    if drawdown_pct >= 10.0:
        return "drawdown_10"
    return None


def evaluate_drawdown_alert(state: dict[str, Any], drawdown_pct: float) -> tuple[str | None, dict[str, Any]]:
    current = drawdown_tier_label(drawdown_pct)
    previous = state.get("drawdown_tier")
    message = None
    if _DRAWDOWN_TIER_RANK[current] > _DRAWDOWN_TIER_RANK[previous]:
        message = f"Portfolio drawdown crossed {_DRAWDOWN_TIER_TEXT[current]}: currently {drawdown_pct:.2f}%."
    new_state = dict(state)
    new_state["drawdown_tier"] = current
    return message, new_state


def evaluate_bucket_cap_alert(
    state: dict[str, Any],
    bucket_name: str,
    current_value_usd: float,
    cap_usd: float,
    near_cap_fraction: float = 0.90,
) -> tuple[str | None, dict[str, Any]]:
    pct_of_cap = current_value_usd / cap_usd if cap_usd > 0 else 0.0
    near = pct_of_cap >= near_cap_fraction
    key = f"{bucket_name}_near_cap"
    was_near = bool(state.get(key, False))
    message = None
    if near and not was_near:
        message = f"{bucket_name} bucket reached {pct_of_cap * 100:.1f}% of its cap (${current_value_usd:,.2f} of ${cap_usd:,.2f})."
    new_state = dict(state)
    new_state[key] = near
    return message, new_state


def _evaluate_zone_crossing(state: dict[str, Any], key: str, in_zone: bool) -> tuple[bool, dict[str, Any]]:
    """Track an in/out-of-zone boolean under ``key`` and report whether this run
    is a fresh crossing into the zone (de-duped while it stays in)."""

    was_in_zone = bool(state.get(key, False))
    new_state = dict(state)
    new_state[key] = in_zone
    return (in_zone and not was_in_zone), new_state


def evaluate_watchlist_accumulation_alert(
    state: dict[str, Any],
    symbol: str,
    score: int,
    accumulation_zone_threshold: int = 70,
) -> tuple[str | None, dict[str, Any]]:
    crossed, new_state = _evaluate_zone_crossing(
        state, f"watchlist_{symbol}_in_zone", score >= accumulation_zone_threshold
    )
    message = None
    if crossed:
        message = f"{symbol} crossed into the accumulation zone: score {score} (threshold {accumulation_zone_threshold})."
    return message, new_state


def evaluate_crypto_accumulation_crossing(
    state: dict[str, Any],
    symbol: str,
    score: int,
    accumulation_zone_threshold: int = 70,
) -> tuple[bool, dict[str, Any]]:
    """De-duped crypto accumulation-zone crossing tracker. Namespaced separately
    from the equity watchlist so the two never collide. Returns (crossed, state);
    the caller composes the message (it needs driver metrics + cap context)."""

    return _evaluate_zone_crossing(
        state, f"crypto_{symbol}_in_zone", score >= accumulation_zone_threshold
    )


def evaluate_btc_target_alert(
    state: dict[str, Any],
    current_btc: float,
    target_btc: float,
) -> tuple[str | None, dict[str, Any]]:
    key = "btc_target_reached"
    was_reached = bool(state.get(key, False))
    reached = current_btc >= target_btc
    message = None
    if reached and not was_reached:
        message = f"BTC core position reached target: {current_btc} >= {target_btc} BTC."
    new_state = dict(state)
    new_state[key] = reached
    return message, new_state


def evaluate_earnings_proximity_alert(
    state: dict[str, Any],
    symbol: str,
    report_date: str,
    days_until: int | None,
    lead_days: int = 3,
) -> tuple[str | None, dict[str, Any]]:
    """Fire once when a watchlist name enters the T-minus-lead_days earnings window.
    De-duped per symbol+report_date so it does not re-alert every daily run."""

    key = f"earnings_alerted_{symbol}_{report_date}"
    if days_until is None or days_until < 0 or days_until > lead_days:
        return None, state
    if state.get(key):
        return None, state
    new_state = dict(state)
    new_state[key] = True
    message = (
        f"{symbol} reports earnings on {report_date} (in {days_until} day(s)). "
        "Initiating or adding a tranche right before earnings is taking a binary event — "
        "the job of this alert is to say WHEN NOT TO ACT, not to buy."
    )
    return message, new_state


def evaluate_position_move_alert(
    state: dict[str, Any],
    symbol: str,
    move_pct: float,
    threshold_pct: float,
    today_date: str,
) -> tuple[str | None, dict[str, Any]]:
    """De-duped per symbol per calendar day -- fires at most once/day even if
    checked hourly, since the underlying 24h move doesn't reset until the next day."""

    key = f"position_move_{symbol}_last_alert_date"
    last_alert_date = state.get(key)
    if abs(move_pct) < threshold_pct or last_alert_date == today_date:
        return None, state
    direction = "up" if move_pct > 0 else "down"
    message = f"{symbol} moved {move_pct:+.1f}% in 24h ({direction}, threshold {threshold_pct:.1f}%)."
    new_state = dict(state)
    new_state[key] = today_date
    return message, new_state
