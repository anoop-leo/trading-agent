"""Regenerate data/portfolio_state.json from data/holdings.json using live prices.

data/holdings.json is the editable source of truth (symbol, quantity, bucket).
This script fetches a live price per symbol through the same Binance provider
used elsewhere in the repo, aggregates by bucket, and writes
data/portfolio_state.json. Both files are gitignored -- this repo is public
and these contain real holdings.

Rerun any time holdings or prices change:
    PYTHONPATH=src venv_trading/bin/python -m risk.build_portfolio_state
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from risk.portfolio_state import PortfolioState, save_portfolio_state
from trading_agent.data import BinanceKlineProvider

DEFAULT_HOLDINGS_PATH = Path("data/holdings.json")
DEFAULT_PORTFOLIO_STATE_PATH = Path("data/portfolio_state.json")


def load_holdings(path: Path = DEFAULT_HOLDINGS_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def fetch_live_price(market_symbol: str, provider: BinanceKlineProvider) -> float:
    """Latest daily close, matching how the rest of the repo defines "price"."""

    candles = provider.fetch_ohlcv(market_symbol, "1d", 2)
    return float(candles.iloc[-1]["close"])


def build_portfolio_state_from_holdings(
    holdings: dict[str, Any],
    provider: BinanceKlineProvider | None = None,
) -> tuple[PortfolioState, dict[str, Any]]:
    """Compute a PortfolioState and a per-symbol diagnostics payload."""

    provider = provider or BinanceKlineProvider()
    bucket_totals = {"core": 0.0, "growth": 0.0, "speculative": 0.0}
    priced_holdings: list[dict[str, Any]] = []

    for holding in holdings["holdings"]:
        price = fetch_live_price(holding["market_symbol"], provider)
        value_usd = round(holding["quantity"] * price, 2)
        bucket_totals[holding["bucket"]] += value_usd
        priced_holdings.append(
            {
                "symbol": holding["symbol"],
                "quantity": holding["quantity"],
                "bucket": holding["bucket"],
                "price_usd": price,
                "value_usd": value_usd,
            }
        )

    cash_usd = float(holdings["cash_usd"])
    for cash_equivalent in holdings.get("cash_equivalents", []):
        cash_usd += float(cash_equivalent["quantity"]) * float(cash_equivalent["usd_value_per_unit"])

    total_value_usd = round(sum(bucket_totals.values()) + cash_usd, 2)
    state = PortfolioState(
        total_value_usd=total_value_usd,
        peak_value_usd=total_value_usd,
        cash_usd=round(cash_usd, 2),
        core_usd=round(bucket_totals["core"], 2),
        growth_usd=round(bucket_totals["growth"], 2),
        speculative_usd=round(bucket_totals["speculative"], 2),
    )
    diagnostics = {
        "priced_holdings": priced_holdings,
        "excluded": holdings.get("excluded", []),
        "total_value_usd": total_value_usd,
    }
    return state, diagnostics


def regenerate_portfolio_state(
    holdings_path: Path = DEFAULT_HOLDINGS_PATH,
    portfolio_state_path: Path = DEFAULT_PORTFOLIO_STATE_PATH,
    provider: BinanceKlineProvider | None = None,
) -> dict[str, Any]:
    holdings = load_holdings(holdings_path)
    state, diagnostics = build_portfolio_state_from_holdings(holdings, provider)
    save_portfolio_state(state, portfolio_state_path)
    diagnostics["portfolio_state_path"] = str(portfolio_state_path)
    diagnostics["state"] = state.to_dict()
    return diagnostics


if __name__ == "__main__":
    result = regenerate_portfolio_state()
    print(json.dumps(result, indent=2, sort_keys=True))
