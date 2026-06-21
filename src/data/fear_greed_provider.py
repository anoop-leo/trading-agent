"""Fear & Greed index provider for BTC Investor Agent."""

from __future__ import annotations

import json
import os
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


JsonOpener = Callable[..., Any]


class FearGreedProvider:
    """Fetch Crypto Fear & Greed from a public read-only endpoint."""

    def __init__(
        self,
        base_url: str = "https://api.alternative.me/fng/",
        timeout_seconds: float = 10.0,
        opener: JsonOpener = urlopen,
    ) -> None:
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self._opener = opener

    def fetch(self, offline: bool = False) -> dict[str, Any]:
        env_value = _env_float("BTC_FEAR_GREED_INDEX")
        if env_value is not None:
            return {"value": env_value, "source": "env", "missing": False, "fallback": False}
        if offline:
            return {"value": None, "source": "offline", "missing": True, "fallback": False}

        request = Request(f"{self.base_url}?limit=1", headers={"User-Agent": "trading-agent-investor/0.1"})
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError):
            return {"value": None, "source": "alternative.me", "missing": True, "fallback": False}

        try:
            value = float(payload["data"][0]["value"])
        except (KeyError, IndexError, TypeError, ValueError):
            return {"value": None, "source": "alternative.me", "missing": True, "fallback": False}
        return {"value": value, "source": "alternative.me", "missing": False, "fallback": False}


def _env_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None
