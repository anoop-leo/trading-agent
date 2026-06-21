"""BTC structural thesis-risk provider for Investor Agent."""

from __future__ import annotations

import os
from typing import Any


SUPPORTED_THESIS_RISK_LEVELS = {"LOW", "MODERATE", "HIGH"}


class ThesisRiskProvider:
    """Read structural BTC thesis-risk flags from environment."""

    def fetch(self, offline: bool = False) -> dict[str, Any]:
        del offline
        level = (os.environ.get("BTC_THESIS_RISK_LEVEL") or "LOW").upper()
        if level not in SUPPORTED_THESIS_RISK_LEVELS:
            level = "MODERATE"
        flags = [
            flag.strip()
            for flag in (os.environ.get("BTC_THESIS_RISK_FLAGS") or "").split(",")
            if flag.strip()
        ]
        return {
            "level": level,
            "flags": flags,
            "source": "env" if flags or os.environ.get("BTC_THESIS_RISK_LEVEL") else "default",
            "missing": False,
            "fallback": False,
        }
