import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from agents.crypto_investor_agent import (
    CRYPTO_INVESTOR_REPORT_TEMPLATE,
    CryptoInvestorConfig,
    normalize_crypto_investor_symbol,
    run_crypto_investor_agent,
)


class FakeChainlinkMarketDataProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def fetch_ohlcv(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        self.calls.append((symbol, interval, limit))
        self.assert_symbol(symbol)
        if interval == "1d":
            return _frame("2023-01-01", 300, "D", 28.0, 16.0, cycle_high=54.0)
        if interval == "1w":
            return _frame("2020-01-01", 130, "W", 32.0, 18.0)
        if interval == "1M":
            return _frame("2018-01-01", 80, "ME", 30.0, 17.0)
        raise AssertionError(f"Unexpected interval {interval}")

    def assert_symbol(self, symbol: str) -> None:
        if symbol != "LINKUSDT":
            raise AssertionError(f"Unexpected symbol {symbol}")


class FakeSymbolMarketDataProvider:
    def __init__(self, expected_symbol: str, daily_only: bool = False) -> None:
        self.expected_symbol = expected_symbol
        self.daily_only = daily_only
        self.calls: list[tuple[str, str, int]] = []

    def fetch_ohlcv(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        self.calls.append((symbol, interval, limit))
        if symbol != self.expected_symbol:
            raise AssertionError(f"Unexpected symbol {symbol}")
        if self.daily_only:
            if interval != "1d":
                raise AssertionError(f"Unexpected interval {interval}")
            return _frame("2023-01-01", 1000, "D", 9.0, 22.0, cycle_high=35.0)
        if interval == "1d":
            return _frame("2023-01-01", 300, "D", 9.0, 22.0, cycle_high=35.0)
        if interval == "1w":
            return _frame("2020-01-01", 130, "W", 15.0, 24.0)
        if interval == "1M":
            return _frame("2018-01-01", 80, "ME", 8.0, 21.0)
        raise AssertionError(f"Unexpected interval {interval}")


class CryptoInvestorAgentTests(unittest.TestCase):
    def test_chainlink_aliases_normalize_to_link(self) -> None:
        self.assertEqual(normalize_crypto_investor_symbol("LINK"), "LINK")
        self.assertEqual(normalize_crypto_investor_symbol("LINKUSDT"), "LINK")
        self.assertEqual(normalize_crypto_investor_symbol("CHAINLINK"), "LINK")
        self.assertEqual(normalize_crypto_investor_symbol("CHAINLNK"), "LINK")

    def test_additional_asset_aliases_normalize(self) -> None:
        self.assertEqual(normalize_crypto_investor_symbol("ETHUSDT"), "ETH")
        self.assertEqual(normalize_crypto_investor_symbol("ETHEREUM"), "ETH")
        self.assertEqual(normalize_crypto_investor_symbol("SOLANA"), "SOL")
        self.assertEqual(normalize_crypto_investor_symbol("SOLUSDT"), "SOL")
        self.assertEqual(normalize_crypto_investor_symbol("RIPPLE"), "XRP")
        self.assertEqual(normalize_crypto_investor_symbol("XRPUSDT"), "XRP")
        self.assertEqual(normalize_crypto_investor_symbol("AVX"), "AVAX")
        self.assertEqual(normalize_crypto_investor_symbol("AVAXUSDT"), "AVAX")
        self.assertEqual(normalize_crypto_investor_symbol("ONDOUSDT"), "ONDO")
        self.assertEqual(normalize_crypto_investor_symbol("HYPER"), "HYPE")
        self.assertEqual(normalize_crypto_investor_symbol("HYPERLIQUID"), "HYPE")

    def test_crypto_investor_agent_outputs_chainlink_report_without_trading_labels(self) -> None:
        with TemporaryDirectory() as temp_dir:
            provider = FakeChainlinkMarketDataProvider()
            payload = run_crypto_investor_agent(
                CryptoInvestorConfig(symbol="CHAINLNK", output_dir=Path(temp_dir), thesis_risk_level="LOW"),
                market_data_provider=provider,
            )

            self.assertEqual(payload["agent"], "CRYPTO_INVESTOR")
            self.assertEqual(payload["symbol"], "LINK")
            self.assertEqual(payload["market_symbol"], "LINKUSDT")
            self.assertEqual(payload["asset_profile"]["name"], "Chainlink")
            self.assertIn("investor_score", payload)
            self.assertIn("factor_scores", payload)
            self.assertIn("distance_from_200d_ma", payload["factor_scores"])
            self.assertIn("volume_trend", payload["factor_scores"])
            self.assertFalse(payload["safety"]["live_trading_enabled"])
            self.assertFalse(payload["safety"]["broker_order_endpoints_enabled"])
            self.assertEqual([call[1] for call in provider.calls], ["1d", "1w", "1M"])
            self.assertTrue((Path(temp_dir) / CRYPTO_INVESTOR_REPORT_TEMPLATE.format(symbol="link")).exists())

            serialized = json.dumps(payload)
            self.assertNotIn('"BUY"', serialized)
            self.assertNotIn('"SELL"', serialized)
            self.assertNotIn("AVOID LONG", serialized)

    def test_crypto_investor_agent_offline_marks_missing_fields(self) -> None:
        with TemporaryDirectory() as temp_dir:
            payload = run_crypto_investor_agent(
                CryptoInvestorConfig(symbol="LINK", output_dir=Path(temp_dir), offline=True)
            )

            self.assertEqual(payload["data_quality"]["confidence"], "LOW")
            self.assertIn("price", payload["data_quality"]["missing_fields"])
            self.assertIn("weekly_rsi", payload["data_quality"]["missing_fields"])
            self.assertEqual(payload["final_investor_action"], "DO_NOT_ACCUMULATE")

    def test_high_thesis_risk_caps_chainlink_action(self) -> None:
        with TemporaryDirectory() as temp_dir:
            payload = run_crypto_investor_agent(
                CryptoInvestorConfig(symbol="LINKUSDT", output_dir=Path(temp_dir), thesis_risk_level="HIGH"),
                market_data_provider=FakeChainlinkMarketDataProvider(),
            )

            self.assertEqual(payload["thesis_risk"]["level"], "HIGH")
            self.assertNotEqual(payload["investor_band"], "STRONG_ACCUMULATION_ZONE")
            self.assertEqual(payload["risk_level"], "HIGH")

    def test_eth_investor_agent_uses_binance_market_symbol(self) -> None:
        with TemporaryDirectory() as temp_dir:
            provider = FakeSymbolMarketDataProvider("ETHUSDT")
            payload = run_crypto_investor_agent(
                CryptoInvestorConfig(symbol="ETH", output_dir=Path(temp_dir)),
                market_data_provider=provider,
            )

            self.assertEqual(payload["symbol"], "ETH")
            self.assertEqual(payload["market_symbol"], "ETHUSDT")
            self.assertEqual(payload["asset_profile"]["market_data_source"], "BINANCE")
            self.assertEqual([call[1] for call in provider.calls], ["1d", "1w", "1M"])

    def test_sol_and_xrp_investor_reports_are_supported(self) -> None:
        with TemporaryDirectory() as temp_dir:
            sol = run_crypto_investor_agent(
                CryptoInvestorConfig(symbol="SOLANA", output_dir=Path(temp_dir)),
                market_data_provider=FakeSymbolMarketDataProvider("SOLUSDT"),
            )
            xrp = run_crypto_investor_agent(
                CryptoInvestorConfig(symbol="RIPPLE", output_dir=Path(temp_dir)),
                market_data_provider=FakeSymbolMarketDataProvider("XRPUSDT"),
            )

            self.assertEqual(sol["symbol"], "SOL")
            self.assertEqual(sol["asset_profile"]["name"], "Solana")
            self.assertEqual(sol["asset_profile"]["sector"], "High-throughput smart contract platform")
            self.assertEqual(xrp["symbol"], "XRP")
            self.assertEqual(xrp["asset_profile"]["name"], "XRP")
            self.assertEqual(xrp["asset_profile"]["sector"], "Payments and settlement network")
            self.assertTrue((Path(temp_dir) / CRYPTO_INVESTOR_REPORT_TEMPLATE.format(symbol="sol")).exists())
            self.assertTrue((Path(temp_dir) / CRYPTO_INVESTOR_REPORT_TEMPLATE.format(symbol="xrp")).exists())

    def test_avax_and_ondo_investor_reports_are_supported(self) -> None:
        with TemporaryDirectory() as temp_dir:
            avax = run_crypto_investor_agent(
                CryptoInvestorConfig(symbol="AVX", output_dir=Path(temp_dir)),
                market_data_provider=FakeSymbolMarketDataProvider("AVAXUSDT"),
            )
            ondo = run_crypto_investor_agent(
                CryptoInvestorConfig(symbol="ONDO", output_dir=Path(temp_dir)),
                market_data_provider=FakeSymbolMarketDataProvider("ONDOUSDT"),
            )

            self.assertEqual(avax["symbol"], "AVAX")
            self.assertEqual(avax["asset_profile"]["name"], "Avalanche")
            self.assertEqual(ondo["symbol"], "ONDO")
            self.assertEqual(ondo["asset_profile"]["name"], "Ondo")
            self.assertTrue((Path(temp_dir) / CRYPTO_INVESTOR_REPORT_TEMPLATE.format(symbol="avax")).exists())
            self.assertTrue((Path(temp_dir) / CRYPTO_INVESTOR_REPORT_TEMPLATE.format(symbol="ondo")).exists())

    def test_hyper_alias_uses_hype_bybit_daily_data_and_derives_higher_timeframes(self) -> None:
        with TemporaryDirectory() as temp_dir:
            provider = FakeSymbolMarketDataProvider("HYPEUSDT", daily_only=True)
            payload = run_crypto_investor_agent(
                CryptoInvestorConfig(symbol="HYPER", output_dir=Path(temp_dir)),
                market_data_provider=provider,
            )

            self.assertEqual(payload["symbol"], "HYPE")
            self.assertEqual(payload["market_symbol"], "HYPEUSDT")
            self.assertEqual(payload["asset_profile"]["name"], "Hyperliquid")
            self.assertEqual(payload["asset_profile"]["market_data_source"], "BYBIT")
            self.assertEqual([call[1] for call in provider.calls], ["1d"])
            self.assertNotIn("weekly_rsi", payload["data_quality"]["missing_fields"])
            self.assertNotIn("monthly_trend", payload["data_quality"]["missing_fields"])

    def test_custom_crypto_asset_can_be_analyzed_from_manual_snapshot(self) -> None:
        with TemporaryDirectory() as temp_dir:
            payload = run_crypto_investor_agent(
                CryptoInvestorConfig(
                    symbol="SEI",
                    asset_name="Sei",
                    sector="Parallelized L1",
                    market_data_source="BINANCE",
                    output_dir=Path(temp_dir),
                    current_price=0.25,
                    ma200=0.40,
                    weekly_rsi=36,
                    monthly_ema20=0.31,
                    recent_cycle_high=1.25,
                    quote_volume_usd=125_000_000,
                    average_quote_volume_usd=80_000_000,
                    atr_pct=4.5,
                    thesis_risk_level="MODERATE",
                )
            )

            self.assertEqual(payload["symbol"], "SEI")
            self.assertEqual(payload["market_symbol"], "SEIUSDT")
            self.assertEqual(payload["asset_profile"]["name"], "Sei")
            self.assertEqual(payload["asset_profile"]["sector"], "Parallelized L1")
            self.assertEqual(payload["investor_band"], "STRONG_ACCUMULATION_ZONE")
            self.assertEqual(payload["final_investor_action"], "ACCUMULATE_OPPORTUNISTICALLY")
            self.assertEqual(payload["suggested_dca_multiplier"], "1.25x to 2.0x normal DCA")
            self.assertEqual(payload["market_snapshot"]["distance_from_200d_ma_pct"], -37.5)
            self.assertEqual(payload["market_snapshot"]["drawdown_from_cycle_high_pct"], -80.0)
            self.assertEqual(payload["market_snapshot"]["volume_ratio"], 1.5625)
            self.assertEqual(payload["data_quality"]["confidence"], "HIGH")
            self.assertTrue((Path(temp_dir) / CRYPTO_INVESTOR_REPORT_TEMPLATE.format(symbol="sei")).exists())


def _frame(
    start: str,
    periods: int,
    freq: str,
    first_close: float,
    last_close: float,
    cycle_high: float | None = None,
) -> pd.DataFrame:
    close = np.linspace(first_close, last_close, periods)
    high = close * 1.02
    if cycle_high is not None:
        high[10] = cycle_high
    volume = np.full(periods, 1_000_000.0)
    volume[-1] = 1_700_000.0
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=periods, freq=freq, tz="UTC"),
            "open": close * 0.99,
            "high": high,
            "low": close * 0.97,
            "close": close,
            "volume": volume,
        }
    )


if __name__ == "__main__":
    unittest.main()
