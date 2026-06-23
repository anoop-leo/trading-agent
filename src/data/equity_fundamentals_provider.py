"""Equity fundamentals provider for the Equity Investor Agent.

Uses Alpha Vantage OVERVIEW + CASH_FLOW (the same ALPHA_VANTAGE_API_KEY already
used by the equity data adapter). Falls back to a manual cache file, then to a
missing/fallback payload, so the agent can still run with degraded confidence.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
DEFAULT_FUNDAMENTALS_CACHE_DIR = Path("data/cache")
DEFAULT_REQUEST_INTERVAL_SECONDS = 15.0

JsonOpener = Callable[..., Any]
SleepFn = Callable[[float], None]


class EquityFundamentalsProvider:
    """Fetch trailing/forward valuation and quality metrics for one equity symbol.

    Alpha Vantage's free tier silently drops the second of two back-to-back
    calls (OVERVIEW then CASH_FLOW) far more often than its documented
    daily quota would suggest -- empirically, spacing them out fixes it.
    request_interval_seconds controls the pause between those two calls.
    """

    def __init__(
        self,
        base_url: str = ALPHA_VANTAGE_BASE_URL,
        cache_dir: Path = DEFAULT_FUNDAMENTALS_CACHE_DIR,
        timeout_seconds: float = 10.0,
        opener: JsonOpener = urlopen,
        environ: Mapping[str, str] | None = None,
        request_interval_seconds: float = DEFAULT_REQUEST_INTERVAL_SECONDS,
        sleep_fn: SleepFn = time.sleep,
    ) -> None:
        self.base_url = base_url
        self.cache_dir = Path(cache_dir)
        self.timeout_seconds = timeout_seconds
        self._opener = opener
        self._environ = environ if environ is not None else os.environ
        self.request_interval_seconds = request_interval_seconds
        self._sleep_fn = sleep_fn

    def fetch(self, symbol: str, offline: bool = False) -> dict[str, Any]:
        symbol = symbol.upper()
        if not offline:
            api_key = self._environ.get("ALPHA_VANTAGE_API_KEY")
            if api_key:
                fetched = self._fetch_alpha_vantage(symbol, api_key)
                if fetched is not None:
                    return fetched

        cached = self._fetch_cache(symbol)
        if cached is not None:
            return cached
        return _missing_fundamentals_payload(symbol, "neutral_fallback")

    def _fetch_alpha_vantage(self, symbol: str, api_key: str) -> dict[str, Any] | None:
        overview = self._alpha_vantage_request("OVERVIEW", symbol, api_key)
        if overview is None or not overview.get("Symbol"):
            return None
        if self.request_interval_seconds > 0:
            self._sleep_fn(self.request_interval_seconds)
        cash_flow = self._alpha_vantage_request("CASH_FLOW", symbol, api_key)

        try:
            market_cap = _to_float(overview.get("MarketCapitalization"))
            fcf_yield_pct = _fcf_yield_pct(cash_flow, market_cap)
            return {
                "symbol": symbol,
                "asset_type": overview.get("AssetType") or "Common Stock",
                "sector": overview.get("Sector"),
                "pe_trailing": _to_float_or_none(overview.get("TrailingPE")),
                "pe_forward": _to_float_or_none(overview.get("ForwardPE")),
                "price_to_book": _to_float_or_none(overview.get("PriceToBookRatio")),
                "peg_ratio": _to_float_or_none(overview.get("PEGRatio")),
                "return_on_equity_ttm": _to_float_or_none(overview.get("ReturnOnEquityTTM")),
                "quarterly_revenue_growth_yoy": _to_float_or_none(overview.get("QuarterlyRevenueGrowthYOY")),
                "quarterly_earnings_growth_yoy": _to_float_or_none(overview.get("QuarterlyEarningsGrowthYOY")),
                "dividend_yield": _to_float_or_none(overview.get("DividendYield")),
                "beta": _to_float_or_none(overview.get("Beta")),
                "market_cap_usd": market_cap,
                "fcf_yield_pct": fcf_yield_pct,
                "source": "alpha_vantage",
                "is_fallback": False,
                "missing": False,
            }
        except (KeyError, TypeError, ValueError):
            return None

    def _alpha_vantage_request(self, function: str, symbol: str, api_key: str) -> dict[str, Any] | None:
        query = urlencode({"function": function, "symbol": symbol, "apikey": api_key})
        request = Request(
            f"{self.base_url}?{query}",
            headers={"User-Agent": "trading-agent-investor/0.1"},
        )
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError):
            return None

    def _fetch_cache(self, symbol: str) -> dict[str, Any] | None:
        cache_path = self.cache_dir / f"fundamentals_{symbol.lower()}.json"
        try:
            payload = json.loads(cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        payload = dict(payload)
        payload.setdefault("symbol", symbol)
        payload["source"] = f"cache/{payload.get('source', 'manual')}"
        payload["is_fallback"] = False
        payload["missing"] = False
        return payload


def fetch_equity_fundamentals(
    symbol: str,
    offline: bool = False,
    provider: EquityFundamentalsProvider | None = None,
) -> dict[str, Any]:
    return (provider or EquityFundamentalsProvider()).fetch(symbol, offline=offline)


def _fcf_yield_pct(cash_flow: dict[str, Any] | None, market_cap: float | None) -> float | None:
    if not cash_flow or not market_cap or market_cap <= 0:
        return None
    try:
        latest = cash_flow["annualReports"][0]
        operating_cash_flow = _to_float(latest["operatingCashflow"])
        capital_expenditures = _to_float(latest["capitalExpenditures"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    free_cash_flow = operating_cash_flow - abs(capital_expenditures)
    return round((free_cash_flow / market_cap) * 100, 4)


def _missing_fundamentals_payload(symbol: str, source: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "asset_type": None,
        "sector": None,
        "pe_trailing": None,
        "pe_forward": None,
        "price_to_book": None,
        "peg_ratio": None,
        "return_on_equity_ttm": None,
        "quarterly_revenue_growth_yoy": None,
        "quarterly_earnings_growth_yoy": None,
        "dividend_yield": None,
        "beta": None,
        "market_cap_usd": None,
        "fcf_yield_pct": None,
        "source": source,
        "is_fallback": True,
        "missing": True,
    }


def _to_float(value: Any) -> float:
    if value in (None, "", "None", "-"):
        raise ValueError("Value is not numeric.")
    return float(value)


def _to_float_or_none(value: Any) -> float | None:
    try:
        return _to_float(value)
    except ValueError:
        return None
