import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from decision.investor_recommendation import bucket_for_crypto_symbol
from monitoring.crypto_scan import (
    CryptoScoreInput,
    _resolve_bucket,
    compute_crypto_accumulation,
    run_crypto_accumulation_scan_job,
)
from monitoring.monitoring_config import MonitoringConfig
from risk.portfolio_state import PortfolioState
from risk.risk_config import RiskEngineConfig


def _state(speculative: float = 11_907.96, cash: float = 200_096.45) -> PortfolioState:
    return PortfolioState(
        total_value_usd=330_000.0,
        peak_value_usd=330_000.0,
        cash_usd=cash,
        core_usd=86_477.0,
        growth_usd=31_546.0,
        speculative_usd=speculative,
    )


def _input(symbol: str, score: int, role: str = "held", bucket: str = "speculative", held: bool = True, drivers=None) -> CryptoScoreInput:
    return CryptoScoreInput(
        symbol=symbol, role=role, bucket=bucket, held=held,
        score=score, band="GOOD" if score >= 70 else "NEUTRAL", confidence="HIGH", drivers=drivers,
    )


class ComputeCryptoAccumulationTests(unittest.TestCase):
    def test_in_zone_name_fires_alert_with_cap_room(self) -> None:
        entries, alerts, state = compute_crypto_accumulation(
            [_input("ADA", 72)], {}, RiskEngineConfig(), _state(), 70
        )
        self.assertTrue(entries["ADA"]["in_zone"])
        self.assertEqual(len(alerts), 1)
        self.assertIn("ADA", alerts[0])
        self.assertIn("room", alerts[0])
        self.assertTrue(state["crypto_ADA_in_zone"])

    def test_below_zone_no_alert_and_distance_reported(self) -> None:
        entries, alerts, _ = compute_crypto_accumulation(
            [_input("ADA", 55)], {}, RiskEngineConfig(), _state(), 70
        )
        self.assertEqual(alerts, [])
        self.assertFalse(entries["ADA"]["in_zone"])
        self.assertEqual(entries["ADA"]["distance_to_zone"], 15)

    def test_does_not_refire_while_remaining_in_zone(self) -> None:
        _, _, state = compute_crypto_accumulation([_input("ADA", 72)], {}, RiskEngineConfig(), _state(), 70)
        _, alerts, _ = compute_crypto_accumulation([_input("ADA", 75)], state, RiskEngineConfig(), _state(), 70)
        self.assertEqual(alerts, [])

    def test_cap_full_blocks_and_annotates_alert(self) -> None:
        # speculative bucket sitting exactly at its 12% cap -> no room to add
        entries, alerts, _ = compute_crypto_accumulation(
            [_input("ADA", 72)], {}, RiskEngineConfig(), _state(speculative=39_600.0), 70
        )
        self.assertEqual(entries["ADA"]["cap_room_usd"], 0.0)
        self.assertIn("cannot take a buy", alerts[0])

    def test_btc_core_alert_includes_driver_metrics(self) -> None:
        drivers = {"mvrv": 2.1, "fear_greed": 45, "cycle_phase": "ACCUMULATION"}
        _, alerts, _ = compute_crypto_accumulation(
            [_input("BTC", 80, role="core", bucket="core", drivers=drivers)], {}, RiskEngineConfig(), _state(), 70
        )
        self.assertIn("MVRV", alerts[0])
        self.assertIn("2.1", alerts[0])
        self.assertIn("cycle", alerts[0])

    def test_watchlist_entry_tagged_not_held(self) -> None:
        _, alerts, _ = compute_crypto_accumulation(
            [_input("HYPE", 75, role="watchlist", held=False)], {}, RiskEngineConfig(), _state(), 70
        )
        self.assertIn("NOT HELD", alerts[0])

    def test_drawdown_circuit_breaker_blocks_non_core_add(self) -> None:
        tripped = PortfolioState(
            total_value_usd=240_000.0, peak_value_usd=330_000.0,
            cash_usd=120_000.0, core_usd=80_000.0, growth_usd=30_000.0, speculative_usd=10_000.0,
        )
        entries, alerts, _ = compute_crypto_accumulation([_input("ADA", 72)], {}, RiskEngineConfig(), tripped, 70)
        self.assertEqual(entries["ADA"]["cap_room_usd"], 0.0)
        self.assertIn("circuit breaker", alerts[0].lower())


class ResolveBucketTests(unittest.TestCase):
    def test_held_symbol_uses_holdings_bucket(self) -> None:
        # LINK/XRP are growth in holdings.json, even though the generic map says speculative.
        self.assertEqual(_resolve_bucket("LINK", "held", {"LINK": "growth"}), "growth")
        self.assertEqual(bucket_for_crypto_symbol("LINK"), "speculative")  # guards the premise

    def test_core_is_always_core(self) -> None:
        self.assertEqual(_resolve_bucket("BTC", "core", {}), "core")

    def test_watchlist_falls_back_to_symbol_map(self) -> None:
        self.assertEqual(_resolve_bucket("HYPE", "watchlist", {}), "speculative")

    def test_unknown_holdings_bucket_falls_back_to_symbol_map(self) -> None:
        self.assertEqual(_resolve_bucket("LINK", "held", {"LINK": "nonsense"}), bucket_for_crypto_symbol("LINK"))


class RunCryptoAccumulationScanJobTests(unittest.TestCase):
    def test_job_scores_book_and_alerts_only_in_zone_names(self) -> None:
        def fake_btc(symbol: str, offline: bool) -> dict:
            return {"score": 80, "band": "GOOD_ACCUMULATION", "confidence": "HIGH",
                    "drivers": {"mvrv": 2.0, "fear_greed": 40, "cycle_phase": "ACCUMULATION"}}

        def fake_alt(symbol: str, offline: bool) -> dict:
            return {"score": 72 if symbol == "ADA" else 50, "band": "GOOD", "confidence": "HIGH", "drivers": None}

        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            config_path = base / "monitoring_config.json"
            config = MonitoringConfig(
                crypto_held_symbols=("ADA", "ETH"),
                crypto_watchlist_symbols=("HYPE",),
            )
            config_path.write_text(json.dumps(config.to_dict()) + "\n")
            state_path = base / "portfolio_state.json"
            state_path.write_text(json.dumps(_state().to_dict()) + "\n")
            scores_path = base / "crypto_scores.json"

            result = run_crypto_accumulation_scan_job(
                monitoring_config_path=config_path,
                alert_state_path=base / "alert_state.json",
                risk_config_path=base / "missing_risk_config.json",  # falls back to defaults
                portfolio_state_path=state_path,
                crypto_scores_path=scores_path,
                btc_scorer=fake_btc,
                alt_scorer=fake_alt,
                sleep_fn=lambda _seconds: None,
                send_alerts=False,
            )
            scores_written = scores_path.exists()

        self.assertTrue(scores_written)
        self.assertEqual(set(result["scores"]), {"BTC", "ADA", "ETH", "HYPE"})
        self.assertEqual(result["scores"]["BTC"]["bucket"], "core")
        self.assertEqual(result["scores"]["ETH"]["bucket"], "growth")
        # BTC (80) and ADA (72) cross; ETH (50) and HYPE (50) do not.
        alerted = " ".join(result["alerts"])
        self.assertIn("BTC", alerted)
        self.assertIn("ADA", alerted)
        self.assertNotIn("ETH", alerted)
        # No holdings file in this temp run -> excluded blind-spot list is empty.
        self.assertEqual(result["excluded"], [])

    def test_job_buckets_held_symbol_from_holdings_file(self) -> None:
        def fake_alt(symbol: str, offline: bool) -> dict:
            return {"score": 72, "band": "GOOD", "confidence": "HIGH", "drivers": None}

        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            config_path = base / "monitoring_config.json"
            config_path.write_text(
                json.dumps(MonitoringConfig(crypto_held_symbols=("LINK",), crypto_watchlist_symbols=()).to_dict()) + "\n"
            )
            state_path = base / "portfolio_state.json"
            state_path.write_text(json.dumps(_state().to_dict()) + "\n")
            holdings_path = base / "holdings.json"
            holdings_path.write_text(json.dumps({"holdings": [{"symbol": "LINK", "bucket": "growth"}]}) + "\n")

            result = run_crypto_accumulation_scan_job(
                monitoring_config_path=config_path,
                alert_state_path=base / "alert_state.json",
                risk_config_path=base / "missing.json",
                portfolio_state_path=state_path,
                holdings_path=holdings_path,
                crypto_scores_path=base / "crypto_scores.json",
                btc_scorer=lambda symbol, offline: {"score": 50, "band": "FAIR", "confidence": "HIGH", "drivers": None},
                alt_scorer=fake_alt,
                sleep_fn=lambda _seconds: None,
                send_alerts=False,
            )

        # holdings.json says LINK is growth -> cap check must use growth, not speculative.
        self.assertEqual(result["scores"]["LINK"]["bucket"], "growth")

    def test_one_bad_symbol_does_not_kill_the_scan(self) -> None:
        def fake_btc(symbol: str, offline: bool) -> dict:
            raise RuntimeError("btc source down")

        def fake_alt(symbol: str, offline: bool) -> dict:
            return {"score": 50, "band": "NEUTRAL", "confidence": "HIGH", "drivers": None}

        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            config_path = base / "monitoring_config.json"
            config_path.write_text(json.dumps(MonitoringConfig(crypto_held_symbols=("ADA",), crypto_watchlist_symbols=()).to_dict()) + "\n")
            state_path = base / "portfolio_state.json"
            state_path.write_text(json.dumps(_state().to_dict()) + "\n")

            result = run_crypto_accumulation_scan_job(
                monitoring_config_path=config_path,
                alert_state_path=base / "alert_state.json",
                risk_config_path=base / "missing.json",
                portfolio_state_path=state_path,
                crypto_scores_path=base / "crypto_scores.json",
                btc_scorer=fake_btc,
                alt_scorer=fake_alt,
                sleep_fn=lambda _seconds: None,
                send_alerts=False,
            )

        self.assertIn("BTC", result["errors"])
        self.assertIn("ADA", result["scores"])


if __name__ == "__main__":
    unittest.main()
