"""Equity news + earnings provider (Alpha Vantage).

Reuses the same ALPHA_VANTAGE_API_KEY already used by the fundamentals provider /
equity data adapter -- no new dependency or secret. Two endpoints:

- EARNINGS_CALENDAR (CSV): one call returns the whole upcoming calendar; we filter
  to the watchlist. This is the centerpiece signal.
- NEWS_SENTIMENT (JSON): one multi-ticker call returns a recent news feed with
  per-article sentiment and topics.

Both are read-only. On any failure / rate-limit / empty response the provider
returns an "available: False" envelope -- never a fabricated all-clear.
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
DEFAULT_REQUEST_INTERVAL_SECONDS = 15.0

Opener = Callable[..., Any]
SleepFn = Callable[[float], None]


class EquityNewsProvider:
    def __init__(
        self,
        base_url: str = ALPHA_VANTAGE_BASE_URL,
        timeout_seconds: float = 15.0,
        opener: Opener = urlopen,
        environ: Mapping[str, str] | None = None,
        request_interval_seconds: float = DEFAULT_REQUEST_INTERVAL_SECONDS,
        sleep_fn: SleepFn = time.sleep,
    ) -> None:
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self._opener = opener
        self._environ = environ if environ is not None else os.environ
        self.request_interval_seconds = request_interval_seconds
        self._sleep_fn = sleep_fn

    def _api_key(self) -> str | None:
        return self._environ.get("ALPHA_VANTAGE_API_KEY")

    def _request_text(self, params: dict[str, str]) -> str | None:
        url = f"{self.base_url}?{urlencode(params)}"
        try:
            request = Request(url, headers={"Accept": "application/json, text/csv"}, method="GET")
            with self._opener(request, timeout=self.timeout_seconds) as response:
                return response.read().decode("utf-8")
        except (HTTPError, URLError, OSError, TimeoutError) as exc:
            return f"__ERROR__{exc}"

    def fetch_earnings_calendar(self, symbols: Sequence[str], horizon: str = "3month", offline: bool = False) -> dict[str, Any]:
        envelope: dict[str, Any] = {"available": False, "source": "alpha_vantage:EARNINGS_CALENDAR", "data": [], "error": None}
        wanted = {s.upper() for s in symbols}
        if offline:
            envelope["error"] = "offline"
            return envelope
        api_key = self._api_key()
        if not api_key:
            envelope["error"] = "ALPHA_VANTAGE_API_KEY not set"
            return envelope

        text = self._request_text({"function": "EARNINGS_CALENDAR", "horizon": horizon, "apikey": api_key})
        if text is None or text.startswith("__ERROR__"):
            envelope["error"] = (text or "no response").replace("__ERROR__", "request failed: ")
            return envelope
        # Rate-limit / error responses come back as JSON, not CSV.
        stripped = text.lstrip()
        if stripped.startswith("{"):
            envelope["error"] = _alpha_vantage_message(stripped) or "non-CSV response"
            return envelope
        if not stripped.lower().startswith("symbol"):
            envelope["error"] = "unexpected earnings-calendar format"
            return envelope

        rows = []
        for row in csv.DictReader(io.StringIO(text)):
            symbol = (row.get("symbol") or "").upper()
            if symbol not in wanted:
                continue
            rows.append({
                "symbol": symbol,
                "name": row.get("name"),
                "report_date": row.get("reportDate"),
                "fiscal_date_ending": row.get("fiscalDateEnding"),
                "estimate": row.get("estimate"),
                "currency": row.get("currency"),
            })
        envelope["available"] = True
        envelope["data"] = rows
        return envelope

    def fetch_news_sentiment(self, symbols: Sequence[str], limit: int = 50, offline: bool = False) -> dict[str, Any]:
        envelope: dict[str, Any] = {"available": False, "source": "alpha_vantage:NEWS_SENTIMENT", "data": [], "error": None}
        if offline:
            envelope["error"] = "offline"
            return envelope
        api_key = self._api_key()
        if not api_key:
            envelope["error"] = "ALPHA_VANTAGE_API_KEY not set"
            return envelope

        text = self._request_text({
            "function": "NEWS_SENTIMENT",
            "tickers": ",".join(s.upper() for s in symbols),
            "limit": str(limit),
            "sort": "LATEST",
            "apikey": api_key,
        })
        if text is None or text.startswith("__ERROR__"):
            envelope["error"] = (text or "no response").replace("__ERROR__", "request failed: ")
            return envelope
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            envelope["error"] = "non-JSON news response"
            return envelope
        feed = payload.get("feed")
        if feed is None:
            envelope["error"] = _alpha_vantage_message(text) or "no news feed returned"
            return envelope
        envelope["available"] = True
        envelope["data"] = feed
        return envelope


def _alpha_vantage_message(text: str) -> str | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    for key in ("Information", "Note", "Error Message"):
        if key in payload:
            return str(payload[key])
    return None
