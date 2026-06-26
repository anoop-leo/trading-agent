"""Monitoring thresholds. No secrets and no real holdings here -- this file
is safe to commit (unlike holdings.json / portfolio_state.json / alert state).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_MONITORING_CONFIG_PATH = Path("config/monitoring_config.json")

DEFAULT_WATCHLIST_SYMBOLS = ("MSFT", "MU", "NVDA", "GOOGL", "AVGO", "AMZN", "PLTR", "TSM", "VRT", "MRVL")

# BTC (core) is scored via investor_agent.py; the held alts and the not-held
# watchlist via crypto_investor_agent.py. Any holding no price provider supports
# is surfaced as a blind-spot note from data/holdings.json (the "excluded" list),
# not configured here.
DEFAULT_CRYPTO_CORE_SYMBOL = "BTC"
DEFAULT_CRYPTO_HELD_SYMBOLS = (
    "ETH", "LINK", "XRP", "SOL", "HBAR", "ONDO", "XLM", "SUI", "ADA", "UNI", "DOT", "AAVE", "GRT",
)
DEFAULT_CRYPTO_WATCHLIST_SYMBOLS = ("HYPE",)

_TUPLE_FIELDS = ("watchlist_symbols", "crypto_held_symbols", "crypto_watchlist_symbols")


@dataclass(frozen=True)
class MonitoringConfig:
    btc_core_target: float = 2.0
    position_daily_move_pct_threshold: float = 15.0
    bucket_near_cap_fraction: float = 0.90
    accumulation_zone_threshold: int = 70
    watchlist_symbols: tuple[str, ...] = field(default_factory=lambda: DEFAULT_WATCHLIST_SYMBOLS)
    crypto_core_symbol: str = DEFAULT_CRYPTO_CORE_SYMBOL
    crypto_held_symbols: tuple[str, ...] = field(default_factory=lambda: DEFAULT_CRYPTO_HELD_SYMBOLS)
    crypto_watchlist_symbols: tuple[str, ...] = field(default_factory=lambda: DEFAULT_CRYPTO_WATCHLIST_SYMBOLS)
    # Equity news + earnings (Phase 2). Earnings calendar is the centerpiece;
    # alert T-minus earnings_alert_lead_days ahead of a watchlist name's report.
    earnings_alert_lead_days: int = 3
    earnings_horizon: str = "3month"
    upcoming_earnings_window_days: int = 14
    earnings_caveat_days: int = 10
    equity_news_limit: int = 50

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in _TUPLE_FIELDS:
            payload[name] = list(getattr(self, name))
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MonitoringConfig":
        kwargs = dict(payload)
        for name in _TUPLE_FIELDS:
            if name in kwargs:
                kwargs[name] = tuple(kwargs[name])
        return cls(**kwargs)


def load_monitoring_config(path: Path = DEFAULT_MONITORING_CONFIG_PATH) -> MonitoringConfig:
    path = Path(path)
    if not path.exists():
        return MonitoringConfig()
    return MonitoringConfig.from_dict(json.loads(path.read_text()))


def write_default_monitoring_config(path: Path = DEFAULT_MONITORING_CONFIG_PATH) -> Path:
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(MonitoringConfig().to_dict(), indent=2, sort_keys=True) + "\n")
    return path
