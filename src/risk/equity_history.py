"""Append-only equity history. This is what makes the drawdown circuit breaker
real -- peak_value_usd should be derived from this accumulating history, not
seeded from a single same-day snapshot.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_EQUITY_HISTORY_PATH = Path("data/equity_history.jsonl")


def append_equity_history_point(point: dict[str, Any], path: Path = DEFAULT_EQUITY_HISTORY_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(point, sort_keys=True) + "\n")
    return path


def load_equity_history(path: Path = DEFAULT_EQUITY_HISTORY_PATH) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    points = []
    for line in path.read_text().splitlines():
        if line.strip():
            points.append(json.loads(line))
    return points


def compute_real_peak_value_usd(history: list[dict[str, Any]], current_total_value_usd: float) -> float:
    """Real all-time-high across history plus the current reading.

    Returns current_total_value_usd alone when history is empty -- the
    breaker reads a real (if short) history from the first datapoint onward,
    rather than waiting for some arbitrary minimum count.
    """

    values = [float(point["total_value_usd"]) for point in history]
    values.append(float(current_total_value_usd))
    return max(values)


def find_point_near(
    history: list[dict[str, Any]],
    target_timestamp: str,
    tolerance_seconds: float = 3600.0,
) -> dict[str, Any] | None:
    """Closest history point to target_timestamp, within tolerance.

    Used for day-over-day position-move comparisons (e.g. "find the snapshot
    from ~24h ago"). Returns None if history is empty or nothing is close
    enough to target_timestamp.
    """

    from datetime import datetime

    if not history:
        return None
    target = datetime.fromisoformat(target_timestamp)
    best: dict[str, Any] | None = None
    best_delta = None
    for point in history:
        try:
            point_time = datetime.fromisoformat(point["timestamp"])
        except (KeyError, ValueError):
            continue
        delta = abs((point_time - target).total_seconds())
        if delta <= tolerance_seconds and (best_delta is None or delta < best_delta):
            best, best_delta = point, delta
    return best
