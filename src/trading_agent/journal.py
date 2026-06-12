"""Signal journal storage and deterministic outcome evaluation for Phase 1.4."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd


JOURNAL_FILENAME = "signal_journal.json"
EVALUATION_HORIZONS_DAYS = (1, 3, 7, 30)
LONG_BIASED_DECISIONS = {"BUY", "BUY WATCH", "WATCH FOR REVERSAL", "ADD", "HOLD"}
DEFENSIVE_DECISIONS = {"AVOID LONG", "REDUCE", "EXIT"}


class JournalError(RuntimeError):
    """Raised when the signal journal cannot be read, written, or evaluated."""


def _format_timestamp(timestamp: object) -> str:
    value = pd.Timestamp(timestamp)
    if value.tzinfo is None:
        value = value.tz_localize("UTC")
    return value.isoformat()


def _format_number(value: object, decimals: int = 2) -> int | float:
    rounded = round(float(value), decimals)
    if rounded.is_integer():
        return int(rounded)
    return rounded


def _journal_id(timestamp: str, symbol: str, setup: str, final_decision: str) -> str:
    return f"{timestamp}|{symbol}|{setup}|{final_decision}"


def build_journal_entry(payload: dict[str, Any]) -> dict[str, Any]:
    """Build the compact journal record from an output payload."""

    missing = [key for key in ("timestamp", "symbol", "setup", "final_decision", "price") if key not in payload]
    if missing:
        raise JournalError(f"Cannot journal payload missing fields: {', '.join(missing)}.")

    timestamp = _format_timestamp(payload["timestamp"])
    symbol = str(payload["symbol"])
    setup = str(payload["setup"])
    final_decision = str(payload["final_decision"])
    return {
        "id": _journal_id(timestamp, symbol, setup, final_decision),
        "timestamp": timestamp,
        "symbol": symbol,
        "setup": setup,
        "final_decision": final_decision,
        "price": _format_number(payload["price"]),
        "evaluations": {f"plus_{days}_day": None for days in EVALUATION_HORIZONS_DAYS},
    }


def load_journal(path: Path) -> list[dict[str, Any]]:
    """Load journal entries from disk."""

    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise JournalError(f"Signal journal contains invalid JSON: {path}.") from exc
    if not isinstance(payload, list):
        raise JournalError("Signal journal must contain a JSON array.")
    return payload


def write_journal(path: Path, entries: list[dict[str, Any]]) -> Path:
    """Persist journal entries as formatted JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    return path


def upsert_journal_entry(entries: list[dict[str, Any]], entry: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Insert a journal entry unless the same signal was already recorded."""

    for existing in entries:
        if existing.get("id") == entry["id"]:
            existing.setdefault("evaluations", entry["evaluations"])
            return entries, False
    entries.append(entry)
    return entries, True


def _evaluation_outcome(final_decision: str, price_change_pct: float) -> str:
    if final_decision in LONG_BIASED_DECISIONS:
        if price_change_pct > 0:
            return "FAVORABLE"
        if price_change_pct < 0:
            return "UNFAVORABLE"
        return "NEUTRAL"
    if final_decision in DEFENSIVE_DECISIONS:
        if price_change_pct < 0:
            return "FAVORABLE"
        if price_change_pct > 0:
            return "UNFAVORABLE"
        return "NEUTRAL"
    return "OBSERVED"


def _price_at_or_after(frame: pd.DataFrame, target_timestamp: pd.Timestamp) -> pd.Series | None:
    candidates = frame[frame["timestamp"] >= target_timestamp]
    if candidates.empty:
        return None
    return candidates.iloc[0]


def evaluate_journal_entries(entries: list[dict[str, Any]], price_frame: pd.DataFrame, symbol: str) -> int:
    """Evaluate due journal entries for the provided symbol using OHLCV data."""

    if price_frame.empty:
        raise JournalError("Cannot evaluate journal from an empty price frame.")
    if "timestamp" not in price_frame.columns or "close" not in price_frame.columns:
        raise JournalError("Price frame must include timestamp and close columns.")

    frame = price_frame[["timestamp", "close"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    evaluated_count = 0

    for entry in entries:
        if entry.get("symbol") != symbol:
            continue
        entry_timestamp = pd.Timestamp(entry["timestamp"])
        if entry_timestamp.tzinfo is None:
            entry_timestamp = entry_timestamp.tz_localize("UTC")
        entry_price = float(entry["price"])
        evaluations = entry.setdefault("evaluations", {})

        for days in EVALUATION_HORIZONS_DAYS:
            key = f"plus_{days}_day"
            if evaluations.get(key) is not None:
                continue
            target_timestamp = entry_timestamp + timedelta(days=days)
            target_row = _price_at_or_after(frame, target_timestamp)
            if target_row is None:
                evaluations[key] = None
                continue

            target_price = float(target_row["close"])
            price_change_pct = round(((target_price - entry_price) / entry_price) * 100, 2)
            evaluations[key] = {
                "target_timestamp": target_timestamp.isoformat(),
                "evaluated_timestamp": _format_timestamp(target_row["timestamp"]),
                "price": _format_number(target_price),
                "price_change_pct": price_change_pct,
                "outcome": _evaluation_outcome(str(entry["final_decision"]), price_change_pct),
            }
            evaluated_count += 1

    return evaluated_count


def update_signal_journal(
    payload: dict[str, Any],
    price_frame: pd.DataFrame,
    output_dir: Path,
    filename: str = JOURNAL_FILENAME,
) -> tuple[Path, dict[str, int | bool]]:
    """Record the latest signal and evaluate due journal horizons."""

    path = output_dir / filename
    entries = load_journal(path)
    entry = build_journal_entry(payload)
    entries, inserted = upsert_journal_entry(entries, entry)
    evaluated_count = evaluate_journal_entries(entries, price_frame, str(payload["symbol"]))
    write_journal(path, entries)
    return path, {"inserted": inserted, "evaluated_count": evaluated_count}
