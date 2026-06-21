"""BTC MVRV provider for Investor Agent valuation scoring."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


COIN_METRICS_BASE_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
GLASSNODE_MVRV_URL = "https://api.glassnode.com/v1/metrics/market/mvrv"
DEFAULT_MVRV_CACHE_PATH = Path("data/cache/mvrv_btc.json")

JsonOpener = Callable[..., Any]


class MVRVProvider:
    """Fetch BTC MVRV from public/read-only valuation data sources."""

    def __init__(
        self,
        coinmetrics_base_url: str = COIN_METRICS_BASE_URL,
        glassnode_url: str = GLASSNODE_MVRV_URL,
        cache_path: Path = DEFAULT_MVRV_CACHE_PATH,
        timeout_seconds: float = 10.0,
        opener: JsonOpener = urlopen,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.coinmetrics_base_url = coinmetrics_base_url
        self.glassnode_url = glassnode_url
        self.cache_path = Path(cache_path)
        self.timeout_seconds = timeout_seconds
        self._opener = opener
        self._environ = environ if environ is not None else os.environ

    def fetch(self, offline: bool = False) -> dict[str, Any]:
        if not offline:
            coinmetrics = self._fetch_coinmetrics()
            if coinmetrics is not None:
                return coinmetrics

            glassnode_key = self._environ.get("GLASSNODE_API_KEY")
            if glassnode_key:
                glassnode = self._fetch_glassnode(glassnode_key)
                if glassnode is not None:
                    return glassnode

        cached = self._fetch_cache()
        if cached is not None:
            return cached
        return _missing_mvrv_payload("neutral_fallback")

    def _fetch_coinmetrics(self) -> dict[str, Any] | None:
        query = urlencode(
            {
                "assets": "btc",
                "metrics": "CapMVRVCur",
                "frequency": "1d",
                "page_size": "1",
                "paging_from": "end",
            }
        )
        request = Request(
            f"{self.coinmetrics_base_url}?{query}",
            headers={"User-Agent": "trading-agent-investor/0.1"},
        )
        try:
            payload = _read_json(self._opener, request, self.timeout_seconds)
            row = payload["data"][0]
            value = _to_float(row["CapMVRVCur"])
            as_of = _normalize_as_of(row.get("time"))
        except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
            return None
        return _mvrv_payload(value, "coinmetrics", as_of, is_fallback=False)

    def _fetch_glassnode(self, api_key: str) -> dict[str, Any] | None:
        query = urlencode({"a": "BTC", "api_key": api_key})
        request = Request(
            f"{self.glassnode_url}?{query}",
            headers={"User-Agent": "trading-agent-investor/0.1"},
        )
        try:
            payload = _read_json(self._opener, request, self.timeout_seconds)
            row = payload[-1]
            value = _to_float(row["v"])
            as_of = _timestamp_to_iso(row.get("t"))
        except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
            return None
        return _mvrv_payload(value, "glassnode", as_of, is_fallback=False)

    def _fetch_cache(self) -> dict[str, Any] | None:
        try:
            payload = json.loads(self.cache_path.read_text())
            value = _to_float(payload["mvrv"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

        source = str(payload.get("source") or "cache")
        normalized_source = "cache/manual" if source.lower() == "manual" else f"cache/{source}"
        return _mvrv_payload(
            value,
            normalized_source,
            _normalize_as_of(payload.get("as_of")),
            is_fallback=False,
        )


def fetch_btc_mvrv(
    offline: bool = False,
    provider: MVRVProvider | None = None,
) -> dict[str, Any]:
    return (provider or MVRVProvider()).fetch(offline=offline)


def _read_json(opener: JsonOpener, request: Request, timeout_seconds: float) -> Any:
    with opener(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _mvrv_payload(value: float, source: str, as_of: str | None, is_fallback: bool) -> dict[str, Any]:
    return {
        "symbol": "BTC",
        "mvrv": value,
        "value": value,
        "source": source,
        "as_of": as_of,
        "is_fallback": is_fallback,
        "missing": False,
        "fallback": False,
    }


def _missing_mvrv_payload(source: str) -> dict[str, Any]:
    return {
        "symbol": "BTC",
        "mvrv": None,
        "value": None,
        "source": source,
        "as_of": None,
        "is_fallback": True,
        "missing": True,
        "fallback": False,
    }


def _to_float(value: Any) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise ValueError("MVRV must be positive.")
    return parsed


def _normalize_as_of(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value)
    if text.endswith("Z"):
        return text
    if "T" in text:
        return text
    return f"{text}T00:00:00Z"


def _timestamp_to_iso(value: Any) -> str | None:
    if value in (None, ""):
        return None
    timestamp = int(value)
    if timestamp > 10_000_000_000:
        timestamp = int(timestamp / 1000)
    return datetime.fromtimestamp(timestamp, tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
