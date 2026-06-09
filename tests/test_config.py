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

    def test_symbol_is_normalized_and_validated(self) -> None:
        config = AgentConfig(symbol="ethusdt")

        self.assertEqual(config.symbol, "ETHUSDT")

    def test_unsupported_symbol_raises(self) -> None:
        with self.assertRaises(ConfigurationError):
            AgentConfig(symbol="DOGEUSDT")

    def test_invalid_history_limit_raises(self) -> None:
        with self.assertRaises(ConfigurationError):
            AgentConfig(history_limit=1001)
        with self.assertRaises(ConfigurationError):
            AgentConfig(history_limit=199)

    def test_position_mode_is_normalized_and_validated(self) -> None:
        self.assertEqual(AgentConfig(position_mode="holding").position_mode, "HOLDING")

        with self.assertRaises(ConfigurationError):
            AgentConfig(position_mode="SHORTING")


if __name__ == "__main__":
    unittest.main()
