import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from risk.risk_config import RiskEngineConfig, load_risk_config, write_default_risk_config


class RiskConfigTests(unittest.TestCase):
    def test_missing_path_returns_defaults(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = load_risk_config(Path(temp_dir) / "missing.json")

        self.assertEqual(config, RiskEngineConfig())
        self.assertEqual(config.bucket_targets.core_pct, 45.0)

    def test_loads_overridden_values_from_disk(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "risk_config.json"
            payload = RiskEngineConfig().to_dict()
            payload["total_portfolio_value_usd"] = 500_000.0
            payload["bucket_targets"]["speculative_pct"] = 20.0
            path.write_text(json.dumps(payload))

            config = load_risk_config(path)

        self.assertEqual(config.total_portfolio_value_usd, 500_000.0)
        self.assertEqual(config.bucket_targets.speculative_pct, 20.0)

    def test_write_default_does_not_overwrite_existing_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "risk_config.json"
            path.write_text(json.dumps({"total_portfolio_value_usd": 1.0, "bucket_targets": {}}))

            write_default_risk_config(path)

            self.assertEqual(json.loads(path.read_text())["total_portfolio_value_usd"], 1.0)

    def test_write_default_creates_parent_directories(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "risk_config.json"

            written = write_default_risk_config(path)

            self.assertTrue(written.exists())
            self.assertEqual(load_risk_config(written), RiskEngineConfig())


if __name__ == "__main__":
    unittest.main()
