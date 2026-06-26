from pathlib import Path
import unittest

from trading_agent.config import AgentConfig, ConfigurationError


class AgentConfigTests(unittest.TestCase):
    def test_defaults_match_phase_one(self) -> None:
        config = AgentConfig()

        self.assertEqual(config.symbol, "BTCUSDT")
        self.assertEqual(config.interval, "1h")
        self.assertEqual(config.history_limit, 500)
        self.assertEqual(config.output_dir, Path("outputs"))
        self.assertEqual(config.position_mode, "NO_POSITION")
        self.assertEqual(config.timeframes, ("1h", "4h", "1d"))
        self.assertEqual(config.market_data_source, "AUTO")
        self.assertEqual(config.resolved_market_data_source, "BINANCE")

    def test_symbol_is_normalized_and_validated(self) -> None:
        config = AgentConfig(symbol="ethusdt")

        self.assertEqual(config.symbol, "ETHUSDT")

    def test_unsupported_symbol_raises(self) -> None:
        with self.assertRaises(ConfigurationError):
            AgentConfig(symbol="DOGEUSDT")

    def test_aaveusdt_supported_and_resolves_to_binance(self) -> None:
        config = AgentConfig(symbol="aaveusdt")
        self.assertEqual(config.symbol, "AAVEUSDT")
        self.assertEqual(config.resolved_market_data_source, "BINANCE")

    def test_invalid_history_limit_raises(self) -> None:
        with self.assertRaises(ConfigurationError):
            AgentConfig(history_limit=1001)
        with self.assertRaises(ConfigurationError):
            AgentConfig(history_limit=199)

    def test_position_mode_is_normalized_and_validated(self) -> None:
        self.assertEqual(AgentConfig(position_mode="holding").position_mode, "HOLDING")

        with self.assertRaises(ConfigurationError):
            AgentConfig(position_mode="SHORTING")

    def test_market_data_source_is_normalized_and_validated(self) -> None:
        self.assertEqual(AgentConfig(market_data_source="bybit").market_data_source, "BYBIT")
        self.assertEqual(AgentConfig(symbol="HYPEUSDT").resolved_market_data_source, "BYBIT")

        with self.assertRaises(ConfigurationError):
            AgentConfig(market_data_source="ROBINHOOD")
        with self.assertRaises(ConfigurationError):
            AgentConfig(symbol="HYPEUSDT", market_data_source="BINANCE")

    def test_timeframes_are_validated(self) -> None:
        self.assertEqual(AgentConfig(timeframes=["1h", "1d"]).timeframes, ("1h", "1d"))

        with self.assertRaises(ConfigurationError):
            AgentConfig(timeframes=["1h", "13h"])
        with self.assertRaises(ConfigurationError):
            AgentConfig(timeframes=["1h", "1h"])
        with self.assertRaises(ConfigurationError):
            AgentConfig(timeframes=[])


if __name__ == "__main__":
    unittest.main()
