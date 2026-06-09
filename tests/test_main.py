from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from trading_agent.config import AgentConfig
from trading_agent.main import run


class FakeProvider:
    def fetch_ohlcv(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        timestamps = pd.date_range("2024-01-01", periods=limit, freq="h", tz="UTC")
        close = np.linspace(100.0, 200.0, limit)
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": close - 1,
                "high": close + 2,
                "low": close - 2,
                "close": close,
                "volume": np.full(limit, 100.0),
            }
        )


class MainTests(unittest.TestCase):
    def test_run_writes_json_and_invokes_chart_writer(self) -> None:
        chart_calls: list[Path] = []

        def chart_writer(_frame: pd.DataFrame, output_dir: Path, symbol: str, setup_label: str) -> Path:
            chart_calls.append(output_dir)
            self.assertEqual(symbol, "BTCUSDT")
            self.assertIsInstance(setup_label, str)
            path = output_dir / "chart.png"
            path.write_bytes(b"fake-png")
            return path

        with TemporaryDirectory() as temp_dir:
            config = AgentConfig(output_dir=Path(temp_dir), history_limit=220)
            payload = run(config, provider=FakeProvider(), chart_writer=chart_writer)

            self.assertEqual(payload["symbol"], "BTCUSDT")
            self.assertEqual(payload["position_mode"], "NO_POSITION")
            self.assertIn("decision", payload)
            self.assertIn("decision_meaning", payload)
            self.assertIn("rr_ratio", payload)
            self.assertIn("market_regime", payload)
            self.assertIn("setup", payload)
            self.assertTrue((Path(temp_dir) / "output.json").exists())
            self.assertEqual(chart_calls, [Path(temp_dir)])


if __name__ == "__main__":
    unittest.main()
