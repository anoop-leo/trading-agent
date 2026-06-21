"""Stablecoin/liquidity condition provider for BTC Investor Agent."""

from __future__ import annotations

import os
from typing import Any


SUPPORTED_LIQUIDITY_CONDITIONS = {"EXPANDING", "NEUTRAL", "CONTRACTING"}


class LiquidityProvider:
    """Return a deterministic liquidity condition with neutral fallback."""

    def fetch(self, offline: bool = False) -> dict[str, Any]:
        del offline
        value = (os.environ.get("BTC_LIQUIDITY_CONDITION") or "").upper()
        if value in SUPPORTED_LIQUIDITY_CONDITIONS:
            return {"value": value, "source": "env", "missing": False, "fallback": False}
        return {"value": "NEUTRAL", "source": "neutral_fallback", "missing": False, "fallback": True}
