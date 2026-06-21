"""Append-only, queryable log of RiskDecisions for later calibration review."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from decision.recommendation import RiskDecision


DEFAULT_RISK_DECISION_LOG_PATH = Path("outputs/risk_decision_log.jsonl")


def append_risk_decision(
    decision: RiskDecision,
    path: Path = DEFAULT_RISK_DECISION_LOG_PATH,
    generated_at: str | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"generated_at": generated_at or datetime.now(UTC).isoformat(), **decision.to_dict()}
    with path.open("a") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return path


def read_risk_decision_log(path: Path = DEFAULT_RISK_DECISION_LOG_PATH) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries
