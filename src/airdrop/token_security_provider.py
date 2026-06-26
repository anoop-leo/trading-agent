"""Read-only token-security data provider (GoPlus-style endpoint).

This issues a single public HTTP GET *about* a token contract and returns the raw
security attributes. It never authenticates, never touches a wallet, and writes
nothing. GoPlus's token_security endpoint is keyless (rate-limited), so this adds
no new secret/dependency.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


GOPLUS_BASE_URL = "https://api.gopluslabs.io"

# GoPlus chain ids. Names are normalized lowercase; aliases map to the same id.
CHAIN_IDS: dict[str, str] = {
    "ethereum": "1", "eth": "1", "mainnet": "1",
    "bsc": "56", "binance": "56", "bnb": "56",
    "polygon": "137", "matic": "137",
    "arbitrum": "42161", "arb": "42161",
    "optimism": "10", "op": "10",
    "avalanche": "43114", "avax": "43114",
    "base": "8453",
    "fantom": "250", "ftm": "250",
    "cronos": "25", "cro": "25",
    "gnosis": "100", "xdai": "100",
    "zksync": "324",
    "linea": "59144",
    "scroll": "534352",
    "mantle": "5000",
    "opbnb": "204",
}

JsonOpener = Callable[..., Any]


def resolve_chain_id(chain: str) -> str | None:
    return CHAIN_IDS.get(chain.strip().lower())


class TokenSecurityProvider:
    """Fetch read-only token-security attributes from a GoPlus-style endpoint."""

    def __init__(
        self,
        base_url: str = GOPLUS_BASE_URL,
        timeout_seconds: float = 15.0,
        opener: JsonOpener = urlopen,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._opener = opener

    def fetch(self, address: str, chain: str) -> dict[str, Any]:
        """Return a normalized envelope. Never raises on network/parse failure --
        an unavailable result is itself a (conservative) screening input."""

        address = address.strip()
        chain_id = resolve_chain_id(chain)
        envelope: dict[str, Any] = {
            "available": False,
            "source": "goplus",
            "chain": chain,
            "chain_id": chain_id,
            "address": address,
            "data": None,
            "error": None,
        }
        if chain_id is None:
            envelope["error"] = f"Unsupported chain {chain!r}. Known: {', '.join(sorted(set(CHAIN_IDS)))}."
            return envelope
        if not _looks_like_evm_address(address):
            envelope["error"] = f"{address!r} does not look like an EVM contract address (0x + 40 hex chars)."
            return envelope

        url = f"{self.base_url}/api/v1/token_security/{chain_id}?contract_addresses={address}"
        try:
            request = Request(url, headers={"Accept": "application/json"}, method="GET")
            with self._opener(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
            envelope["error"] = f"token-security fetch failed: {exc}"
            return envelope

        result = payload.get("result") or {}
        # GoPlus keys the result by the lowercased address.
        record = result.get(address.lower()) or (next(iter(result.values())) if result else None)
        if not record:
            envelope["error"] = "token not indexed by the security source (no data returned)."
            return envelope

        envelope["available"] = True
        envelope["data"] = record
        return envelope


def _looks_like_evm_address(address: str) -> bool:
    if not address.startswith("0x") or len(address) != 42:
        return False
    try:
        int(address, 16)
    except ValueError:
        return False
    return True
