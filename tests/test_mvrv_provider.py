import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from urllib.error import URLError

import numpy as np
import pandas as pd

from agents.investor_agent import InvestorAgentConfig, run_investor_agent
from data.mvrv_provider import MVRVProvider, fetch_btc_mvrv
from planning.goal_accumulation_plan import GoalAccumulationInput, build_goal_accumulation_plan
from scoring.accumulation_score import score_mvrv


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class FakeMarketDataProvider:
    def fetch_ohlcv(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        del symbol, limit
        if interval == "1d":
            close = np.concatenate([np.full(250, 160.0), np.full(50, 100.0)])
            high = close * 1.01
            high[10] = 400.0
            return _frame_from_close("2023-01-01", "D", close, high)
        if interval == "1w":
            return _frame("2020-01-01", 120, "W", 140.0, 70.0)
        if interval == "1M":
            return _frame("2017-01-01", 80, "ME", 120.0, 70.0)
        raise AssertionError(f"Unexpected interval {interval}")


class FakeProvider:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def fetch(self, offline: bool = False) -> dict[str, object]:
        del offline
        return self.payload


class MVRVProviderTests(unittest.TestCase):
    def test_coinmetrics_mvrv_success_returns_normalized_value(self) -> None:
        def opener(request: object, timeout: float = 0) -> FakeResponse:
            del timeout
            self.assertIn("CapMVRVCur", request.full_url)
            return FakeResponse({"data": [{"time": "2026-06-17T00:00:00.000000000Z", "CapMVRVCur": "1.39"}]})

        payload = fetch_btc_mvrv(provider=MVRVProvider(opener=opener))

        self.assertEqual(payload["symbol"], "BTC")
        self.assertEqual(payload["mvrv"], 1.39)
        self.assertEqual(payload["value"], 1.39)
        self.assertEqual(payload["source"], "coinmetrics")
        self.assertEqual(payload["as_of"], "2026-06-17T00:00:00.000000000Z")
        self.assertFalse(payload["is_fallback"])

    def test_glassnode_used_when_coinmetrics_fails_and_api_key_exists(self) -> None:
        def opener(request: object, timeout: float = 0) -> FakeResponse:
            del timeout
            if "coinmetrics" in request.full_url:
                raise URLError("coinmetrics down")
            self.assertIn("api.glassnode.com", request.full_url)
            return FakeResponse([{"t": 1781654400, "v": 1.42}])

        payload = fetch_btc_mvrv(provider=MVRVProvider(opener=opener, environ={"GLASSNODE_API_KEY": "test-key"}))

        self.assertEqual(payload["mvrv"], 1.42)
        self.assertEqual(payload["source"], "glassnode")
        self.assertEqual(payload["as_of"], "2026-06-17T00:00:00Z")

    def test_cache_used_when_online_providers_fail(self) -> None:
        with TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "mvrv_btc.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "symbol": "BTC",
                        "mvrv": 1.399,
                        "source": "manual",
                        "as_of": "2026-06-17",
                    }
                )
            )

            payload = fetch_btc_mvrv(provider=MVRVProvider(opener=_failing_opener, cache_path=cache_path))

        self.assertEqual(payload["mvrv"], 1.399)
        self.assertEqual(payload["source"], "cache/manual")
        self.assertEqual(payload["as_of"], "2026-06-17T00:00:00Z")
        self.assertFalse(payload["missing"])

    def test_neutral_fallback_when_all_providers_fail(self) -> None:
        with TemporaryDirectory() as temp_dir:
            payload = fetch_btc_mvrv(
                provider=MVRVProvider(opener=_failing_opener, cache_path=Path(temp_dir) / "missing.json")
            )

        self.assertIsNone(payload["mvrv"])
        self.assertEqual(payload["source"], "neutral_fallback")
        self.assertTrue(payload["missing"])
        self.assertTrue(payload["is_fallback"])

    def test_mvrv_below_1_5_scores_16(self) -> None:
        self.assertEqual(score_mvrv(1.399), 16)

    def test_mvrv_missing_locks_opportunistic_and_deep_value(self) -> None:
        plan = _goal_plan(mvrv_missing=True, mvrv_value=None, accumulation_score=90, margin_of_safety_score=80)

        self.assertFalse(_bucket(plan, "OPPORTUNISTIC")["allowed_now"])
        self.assertFalse(_bucket(plan, "DEEP_VALUE_RESERVE")["allowed_now"])

    def test_mvrv_present_allows_bucket_eligibility_check(self) -> None:
        plan = _goal_plan(
            mvrv_missing=False,
            mvrv_value=1.399,
            accumulation_score=85,
            margin_of_safety_score=75,
            distance_from_200d_ma_pct=-30.0,
            drawdown_from_cycle_high_pct=-55.0,
            final_investor_action="AGGRESSIVE_DCA_ALLOWED",
        )

        self.assertTrue(_bucket(plan, "OPPORTUNISTIC")["allowed_now"])
        self.assertTrue(_bucket(plan, "DEEP_VALUE_RESERVE")["allowed_now"])

    def test_mvrv_above_3_caps_accumulation(self) -> None:
        payload = _investor_payload({"value": 3.5, "mvrv": 3.5, "source": "coinmetrics", "as_of": "2026-06-17T00:00:00Z", "missing": False, "fallback": False})

        self.assertEqual(payload["accumulation_band"], "FAIR")
        self.assertEqual(payload["confidence_adjusted_dca_multiplier"], "0.5x to 1.0x normal DCA")
        self.assertFalse(_bucket(payload["goal_plan"], "OPPORTUNISTIC")["allowed_now"])
        self.assertFalse(_bucket(payload["goal_plan"], "DEEP_VALUE_RESERVE")["allowed_now"])

    def test_investor_report_includes_mvrv_source_and_as_of(self) -> None:
        payload = _investor_payload({"value": 1.399, "mvrv": 1.399, "source": "coinmetrics", "as_of": "2026-06-17T00:00:00Z", "missing": False, "fallback": False})

        self.assertEqual(payload["factor_scores"]["mvrv"]["value"], 1.399)
        self.assertEqual(payload["factor_scores"]["mvrv"]["score"], 16)
        self.assertEqual(payload["factor_scores"]["mvrv"]["source"], "coinmetrics")
        self.assertEqual(payload["factor_scores"]["mvrv"]["as_of"], "2026-06-17T00:00:00Z")


def _failing_opener(request: object, timeout: float = 0) -> FakeResponse:
    del request, timeout
    raise URLError("provider unavailable")


def _goal_plan(**overrides: object) -> dict[str, object]:
    defaults = {
        "current_btc": 1.13059494,
        "target_btc": 2.0,
        "current_price": 100000.0,
        "accumulation_score": 76,
        "margin_of_safety_score": 65,
        "mvrv_value": 1.399,
        "mvrv_missing": False,
        "fear_and_greed_value": 20.0,
        "thesis_risk_level": "LOW",
        "final_investor_action": "INCREASE_DCA_GRADUALLY",
        "rebalance_signal": "BELOW_TARGET",
        "current_btc_allocation_pct": 0.0,
        "target_btc_allocation_pct": 2.0,
        "max_btc_allocation_pct": 5.0,
        "institutional_score": 80,
        "distance_from_200d_ma_pct": -20.0,
        "drawdown_from_cycle_high_pct": -30.0,
    }
    defaults.update(overrides)
    return build_goal_accumulation_plan(GoalAccumulationInput(**defaults))


def _investor_payload(mvrv_payload: dict[str, object]) -> dict[str, object]:
    with TemporaryDirectory() as temp_dir:
        return run_investor_agent(
            InvestorAgentConfig(output_dir=Path(temp_dir), current_btc_allocation_pct=0.0),
            market_data_provider=FakeMarketDataProvider(),
            fear_greed_provider=FakeProvider({"value": 10.0, "missing": False, "fallback": False}),
            mvrv_provider=FakeProvider(mvrv_payload),
            liquidity_provider=FakeProvider({"value": "EXPANDING", "missing": False, "fallback": False}),
            thesis_risk_provider=FakeProvider({"level": "LOW", "flags": [], "missing": False, "fallback": False}),
        )


def _bucket(plan: dict[str, object], name: str) -> dict[str, object]:
    buckets = {item["bucket_name"]: item for item in plan["tranche_plan"]}
    return buckets[name]


def _frame(
    start: str,
    periods: int,
    freq: str,
    first_close: float,
    last_close: float,
) -> pd.DataFrame:
    close = np.linspace(first_close, last_close, periods)
    return _frame_from_close(start, freq, close)


def _frame_from_close(start: str, freq: str, close: np.ndarray, high: np.ndarray | None = None) -> pd.DataFrame:
    if high is None:
        high = close * 1.01
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=len(close), freq=freq, tz="UTC"),
            "open": close * 0.99,
            "high": high,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(len(close), 1000.0),
        }
    )


if __name__ == "__main__":
    unittest.main()
