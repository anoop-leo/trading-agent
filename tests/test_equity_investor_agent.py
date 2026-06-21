from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from agents.equity_investor_agent import (
    EquityInvestorConfig,
    is_core_etf_symbol,
    run_equity_investor_agent,
)
from data.equity_data_adapter import EquityDataResult


class FakeFundamentalsProvider:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def fetch(self, symbol: str, offline: bool = False) -> dict[str, object]:
        del symbol, offline
        return self.payload


def _daily_frame(periods: int = 1200, first_close: float = 60.0, last_close: float = 120.0) -> pd.DataFrame:
    close = np.linspace(first_close, last_close, periods)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=periods, freq="D", tz="UTC"),
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(periods, 1_000_000.0),
        }
    )


def _fake_loader(daily: pd.DataFrame):
    def loader(symbol: str, timeframes=("1d",), refresh_cache: bool = False, **kwargs):
        del symbol, timeframes, refresh_cache, kwargs
        return EquityDataResult(symbol="TEST", frames={"1d": daily}, providers={}, validations={}, attempts=[])

    return loader


GOOD_FUNDAMENTALS = {
    "symbol": "AAPL",
    "pe_trailing": 15.0,
    "pe_forward": 14.0,
    "price_to_book": 4.0,
    "peg_ratio": 0.9,
    "return_on_equity_ttm": 0.25,
    "quarterly_revenue_growth_yoy": 0.1,
    "quarterly_earnings_growth_yoy": 0.1,
    "fcf_yield_pct": 6.0,
    "missing": False,
    "is_fallback": False,
    "source": "alpha_vantage",
}

MISSING_FUNDAMENTALS = {
    "symbol": "AAPL",
    "pe_trailing": None,
    "pe_forward": None,
    "price_to_book": None,
    "peg_ratio": None,
    "return_on_equity_ttm": None,
    "quarterly_revenue_growth_yoy": None,
    "quarterly_earnings_growth_yoy": None,
    "fcf_yield_pct": None,
    "missing": True,
    "is_fallback": True,
    "source": "neutral_fallback",
}


class CoreEtfTests(unittest.TestCase):
    def test_known_etf_symbols_are_core(self) -> None:
        self.assertTrue(is_core_etf_symbol("SPY"))
        self.assertTrue(is_core_etf_symbol("qqq"))
        self.assertFalse(is_core_etf_symbol("AAPL"))

    def test_spy_passes_core_vehicle_check_and_proposes_buy(self) -> None:
        with TemporaryDirectory() as temp_dir:
            payload = run_equity_investor_agent(EquityInvestorConfig(symbol="SPY", output_dir=Path(temp_dir)))

        self.assertEqual(payload["bucket"], "core")
        self.assertTrue(payload["core_vehicle_check"]["reasonable_broad_market_vehicle"])
        self.assertEqual(payload["position_recommendation"]["action"], "buy")
        self.assertEqual(payload["position_recommendation"]["bucket"], "core")
        self.assertEqual(payload["position_recommendation"]["asset_class"], "equity")

    def test_unknown_etf_without_profile_is_treated_as_unverified(self) -> None:
        with TemporaryDirectory() as temp_dir:
            payload = run_equity_investor_agent(
                EquityInvestorConfig(symbol="ARBITRARY", bucket="core", output_dir=Path(temp_dir))
            )

        self.assertFalse(payload["core_vehicle_check"]["reasonable_broad_market_vehicle"])
        self.assertEqual(payload["position_recommendation"]["action"], "hold")


class GrowthEquityTests(unittest.TestCase):
    def test_strong_fundamentals_and_discount_produce_buy_recommendation(self) -> None:
        with TemporaryDirectory() as temp_dir:
            payload = run_equity_investor_agent(
                EquityInvestorConfig(symbol="AAPL", bucket="growth", output_dir=Path(temp_dir), default_position_usd=4_000.0),
                equity_data_loader=_fake_loader(_daily_frame(first_close=120.0, last_close=70.0)),
                fundamentals_provider=FakeFundamentalsProvider(GOOD_FUNDAMENTALS),
            )

        self.assertEqual(payload["agent"], "EQUITY_INVESTOR")
        self.assertEqual(payload["bucket"], "growth")
        self.assertGreater(payload["investor_score"], 55)
        self.assertEqual(payload["position_recommendation"]["action"], "buy")
        self.assertEqual(payload["position_recommendation"]["asset_class"], "equity")
        self.assertGreater(payload["position_recommendation"]["suggested_size_usd"], 0)

    def test_missing_fundamentals_degrades_confidence_and_caps_action(self) -> None:
        with TemporaryDirectory() as temp_dir:
            payload = run_equity_investor_agent(
                EquityInvestorConfig(symbol="AAPL", bucket="growth", output_dir=Path(temp_dir)),
                equity_data_loader=_fake_loader(_daily_frame()),
                fundamentals_provider=FakeFundamentalsProvider(MISSING_FUNDAMENTALS),
            )

        self.assertEqual(payload["data_quality"]["confidence"], "LOW")
        self.assertIn("pe", payload["data_quality"]["missing_fields"])

    def test_manual_price_overrides_are_used_without_network(self) -> None:
        with TemporaryDirectory() as temp_dir:
            payload = run_equity_investor_agent(
                EquityInvestorConfig(
                    symbol="AAPL", bucket="growth", output_dir=Path(temp_dir),
                    current_price=100.0, ma200=120.0, weekly_rsi=35.0, monthly_ema20=110.0, monthly_trend="BELOW_EMA20",
                ),
                fundamentals_provider=FakeFundamentalsProvider(GOOD_FUNDAMENTALS),
            )

        self.assertEqual(payload["market_snapshot"]["price"], 100.0)
        self.assertEqual(payload["market_snapshot"]["monthly_trend"], "BELOW_EMA20")

    def test_offline_without_manual_inputs_reports_all_missing(self) -> None:
        with TemporaryDirectory() as temp_dir:
            payload = run_equity_investor_agent(
                EquityInvestorConfig(symbol="AAPL", bucket="growth", output_dir=Path(temp_dir), offline=True),
                fundamentals_provider=FakeFundamentalsProvider(MISSING_FUNDAMENTALS),
            )

        self.assertIsNone(payload["market_snapshot"]["price"])
        self.assertEqual(payload["data_quality"]["confidence"], "LOW")

    def test_report_is_written_to_output_dir(self) -> None:
        with TemporaryDirectory() as temp_dir:
            run_equity_investor_agent(
                EquityInvestorConfig(symbol="AAPL", bucket="growth", output_dir=Path(temp_dir)),
                equity_data_loader=_fake_loader(_daily_frame()),
                fundamentals_provider=FakeFundamentalsProvider(GOOD_FUNDAMENTALS),
            )
            self.assertTrue((Path(temp_dir) / "equity_investor_aapl_report.json").exists())


if __name__ == "__main__":
    unittest.main()
