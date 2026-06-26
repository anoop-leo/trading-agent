import json
import unittest
from urllib.error import URLError

from airdrop.token_security_provider import TokenSecurityProvider, resolve_chain_id

VALID_ADDRESS = "0x" + "a" * 40


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def read(self) -> bytes:
        return self._payload


def _opener_returning(payload: dict):
    def _opener(request, timeout=None):  # noqa: ANN001
        return _FakeResponse(payload)
    return _opener


class ResolveChainTests(unittest.TestCase):
    def test_known_aliases(self) -> None:
        self.assertEqual(resolve_chain_id("ethereum"), "1")
        self.assertEqual(resolve_chain_id("BSC"), "56")
        self.assertEqual(resolve_chain_id("avax"), "43114")

    def test_unknown_chain(self) -> None:
        self.assertIsNone(resolve_chain_id("dogechain-xyz"))


class FetchTests(unittest.TestCase):
    def test_valid_response_is_available(self) -> None:
        payload = {"code": 1, "result": {VALID_ADDRESS.lower(): {"token_symbol": "X", "is_honeypot": "0"}}}
        provider = TokenSecurityProvider(opener=_opener_returning(payload))
        envelope = provider.fetch(VALID_ADDRESS, "ethereum")
        self.assertTrue(envelope["available"])
        self.assertEqual(envelope["data"]["token_symbol"], "X")
        self.assertEqual(envelope["chain_id"], "1")

    def test_empty_result_is_unavailable(self) -> None:
        provider = TokenSecurityProvider(opener=_opener_returning({"code": 1, "result": {}}))
        envelope = provider.fetch(VALID_ADDRESS, "ethereum")
        self.assertFalse(envelope["available"])
        self.assertIn("not indexed", envelope["error"])

    def test_unsupported_chain_short_circuits(self) -> None:
        provider = TokenSecurityProvider(opener=_opener_returning({}))
        envelope = provider.fetch(VALID_ADDRESS, "nonsense-chain")
        self.assertFalse(envelope["available"])
        self.assertIn("Unsupported chain", envelope["error"])

    def test_malformed_address_short_circuits(self) -> None:
        provider = TokenSecurityProvider(opener=_opener_returning({}))
        envelope = provider.fetch("not-an-address", "ethereum")
        self.assertFalse(envelope["available"])
        self.assertIn("does not look like", envelope["error"])

    def test_network_error_is_swallowed_as_unavailable(self) -> None:
        def _raising_opener(request, timeout=None):  # noqa: ANN001
            raise URLError("boom")

        provider = TokenSecurityProvider(opener=_raising_opener)
        envelope = provider.fetch(VALID_ADDRESS, "ethereum")
        self.assertFalse(envelope["available"])
        self.assertIn("fetch failed", envelope["error"])


if __name__ == "__main__":
    unittest.main()
