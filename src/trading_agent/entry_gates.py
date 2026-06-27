"""Phase 1 entry-quality safety gates.

After the multi-timeframe decision is assembled, a BUY can still be a poor entry
(bad reward/risk, no volume confirmation, an overbought chase). These gates ONLY
downgrade a BUY to WAIT / WAIT_FOR_PULLBACK -- they never create or upgrade a
signal -- and a missing score is treated as a FAILED check, never a silent pass.

The gates mutate ``final_decision`` in place (so existing consumers see the gated
value) and add:
  - pre_gate_decision : the decision before gating
  - gate_triggered    : which gate fired (or None)
  - trend_bias        : BULLISH | BEARISH | NEUTRAL
  - entry_decision    : the actionable field going forward
  - watch_levels      : pullback/breakout/invalidation refs (WAIT_FOR_PULLBACK only)
  - alert_summary     : human-readable summary (WAIT_FOR_PULLBACK only)
"""

from __future__ import annotations

from typing import Any

VOLUME_BREAKOUT_REQUIREMENT = 1.3


def apply_entry_quality_gates(payload: dict[str, Any]) -> dict[str, Any]:
    """Mutate ``payload`` in place with the entry-quality gates and new fields."""

    final_decision = payload.get("final_decision")
    # Preserve any existing reason; only overwrite it when a gate actually fires.
    final_decision_reason = payload.get("final_decision_reason", "") or ""

    rr_ratio = payload.get("rr_ratio")
    rr_score = payload.get("rr_score")
    volume_score = payload.get("volume_score")
    rsi = payload.get("rsi")
    volume_ratio = payload.get("volume_ratio")
    setup = payload.get("setup")
    regime_score = payload.get("regime_score")
    multi_timeframe = payload.get("multi_timeframe")
    support = payload.get("support")
    resistance = payload.get("resistance")
    ema20 = payload.get("ema20")
    stop_loss = payload.get("stop_loss")
    symbol = payload.get("symbol")

    pre_gate_decision = final_decision
    gate_triggered = None

    if pre_gate_decision == "BUY":
        if rr_ratio is not None and rr_ratio < 1.0:
            final_decision = "WAIT"
            final_decision_reason = "RR below minimum threshold (< 1.0R). Do not enter regardless of trend."
            gate_triggered = "RR_BELOW_1R"

        elif rr_score is None or volume_score is None:
            final_decision = "WAIT"
            final_decision_reason = (
                f"Entry scores unavailable — rr_score={rr_score}, volume_score={volume_score}. "
                "Cannot confirm quality."
            )
            gate_triggered = "MISSING_ENTRY_SCORE"

        elif rr_score == 0 and volume_score == 0:
            final_decision = "WAIT"
            final_decision_reason = "Poor entry quality: zero RR score and no volume confirmation."
            gate_triggered = "ZERO_RR_AND_VOLUME"

        elif rsi is not None and rsi > 75 and volume_ratio is not None and volume_ratio < 1.0:
            final_decision = "WAIT"
            final_decision_reason = "Overbought RSI with no volume expansion — likely chasing a move."
            gate_triggered = "OVERBOUGHT_LOW_VOLUME"

    # trend_bias: check alignment FIRST so a bearish multi-timeframe read is never
    # overridden by a high regime_score.
    alignment = multi_timeframe.get("alignment") if multi_timeframe else None
    rs = regime_score if regime_score is not None else None
    if alignment == "BULLISH_ALIGNMENT":
        trend_bias = "BULLISH"
    elif alignment == "BEARISH_ALIGNMENT":
        trend_bias = "BEARISH"
    elif rs is not None and rs >= 7:
        trend_bias = "BULLISH"
    elif rs is not None and rs <= 3:
        trend_bias = "BEARISH"
    else:
        trend_bias = "NEUTRAL"

    # entry_decision: the actionable field, derived AFTER the gates are applied.
    entry_decision = final_decision
    if pre_gate_decision == "BUY" and final_decision == "WAIT":
        if setup == "TREND_FOLLOWING":
            entry_decision = "WAIT_FOR_PULLBACK"
        else:
            entry_decision = "WAIT"

    payload["pre_gate_decision"] = pre_gate_decision
    payload["gate_triggered"] = gate_triggered
    payload["final_decision"] = final_decision
    payload["final_decision_reason"] = final_decision_reason
    payload["trend_bias"] = trend_bias
    payload["entry_decision"] = entry_decision

    if entry_decision == "WAIT_FOR_PULLBACK":
        # Normalize the pullback band once so low is always the lower bound, even
        # when support sits above ema20. Fix the value, not just the display.
        if support is not None and ema20 is not None:
            pb_low = min(support, ema20)
            pb_high = max(support, ema20)
        else:
            pb_low = None
            pb_high = None
        payload["watch_levels"] = {
            "pullback_entry": {"low": pb_low, "high": pb_high},
            "breakout_entry": {"level": resistance, "volume_requirement": VOLUME_BREAKOUT_REQUIREMENT},
            "invalidation": stop_loss,
        }
        payload["alert_summary"] = (
            f"{symbol} — {trend_bias} trend, entry {entry_decision}. "
            f"Watch pullback {_fmt(pb_low)}–{_fmt(pb_high)} or breakout above {_fmt(resistance)} "
            f"with volume >= {VOLUME_BREAKOUT_REQUIREMENT}x."
        )
    else:
        payload["watch_levels"] = None
        payload["alert_summary"] = None

    return payload


def _fmt(value: Any) -> str:
    return "n/a" if value is None else str(value)
