import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urlparse
import unittest

from data.coinbase_execution_cost_audit import (
    CoinbaseExecutionAuditConfig,
    CoinbaseExecutionAuditMergeConfig,
    build_execution_cost_report,
    build_final_execution_cost_report,
    build_orderbook_sample,
    fetch_public_product_book,
    merge_coinbase_execution_audit_samples,
    normalize_coinbase_product_id,
    run_coinbase_execution_cost_audit,
)


PRODUCT_BOOK_PAYLOAD = {
    "pricebook": {
        "product_id": "BTC-USD",
        "time": "2026-06-13T14:00:00Z",
        "bids": [
            {"price": "99990", "size": "0.05"},
            {"price": "99980", "size": "0.05"},
            {"price": "99970", "size": "0.05"},
            {"price": "99960", "size": "0.05"},
            {"price": "99950", "size": "0.05"},
        ],
        "asks": [
            {"price": "100010", "size": "0.05"},
            {"price": "100020", "size": "0.05"},
            {"price": "100030", "size": "0.05"},
            {"price": "100040", "size": "0.05"},
            {"price": "100050", "size": "0.05"},
        ],
    },
    "mid_market": "100000",
}


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class CoinbaseExecutionCostAuditTests(unittest.TestCase):
    def test_normalize_coinbase_product_id_accepts_common_btc_symbols(self) -> None:
        self.assertEqual(normalize_coinbase_product_id("BTC/USD"), "BTC-USD")
        self.assertEqual(normalize_coinbase_product_id("btcusdt"), "BTC-USDT")
        self.assertEqual(normalize_coinbase_product_id("BTC-USD"), "BTC-USD")

    def test_fetch_public_product_book_uses_read_only_market_endpoint(self) -> None:
        calls = []

        def opener(request, timeout):
            calls.append((request, timeout))
            return FakeResponse(PRODUCT_BOOK_PAYLOAD)

        config = CoinbaseExecutionAuditConfig(product_id="BTC/USD", timeout_seconds=4.0, order_book_limit=25)
        payload = fetch_public_product_book(config, opener=opener)

        request, timeout = calls[0]
        parsed = urlparse(request.full_url)
        query = parse_qs(parsed.query)
        self.assertEqual(parsed.path, "/api/v3/brokerage/market/product_book")
        self.assertEqual(query["product_id"], ["BTC-USD"])
        self.assertEqual(query["limit"], ["25"])
        self.assertEqual(timeout, 4.0)
        self.assertEqual(request.get_method(), "GET")
        self.assertIsNone(request.headers.get("Authorization"))
        self.assertEqual(payload["pricebook"]["product_id"], "BTC-USD")

    def test_fetch_public_product_book_adds_optional_bearer_token(self) -> None:
        calls = []

        def opener(request, timeout):
            calls.append(request)
            return FakeResponse(PRODUCT_BOOK_PAYLOAD)

        config = CoinbaseExecutionAuditConfig(bearer_token="read-only-token")
        fetch_public_product_book(config, opener=opener)

        self.assertEqual(calls[0].headers["Authorization"], "Bearer read-only-token")

    def test_build_orderbook_sample_calculates_spread_depth_and_all_in_cost(self) -> None:
        config = CoinbaseExecutionAuditConfig(fee_rate=0.001, intended_order_size_usd=2500.0)
        sample = build_orderbook_sample(PRODUCT_BOOK_PAYLOAD, config, "2026-06-13T14:00:00+00:00")

        self.assertEqual(sample["best_bid"], 99990.0)
        self.assertEqual(sample["best_ask"], 100010.0)
        self.assertAlmostEqual(sample["mid_price"], 100000.0)
        self.assertAlmostEqual(sample["spread_pct"], 0.0002)
        self.assertAlmostEqual(sample["bid_depth_1"], 4999.5)
        self.assertAlmostEqual(sample["ask_depth_5"], 25007.5)
        estimate = sample["market_order_estimates"]["order_2500_usd"]
        self.assertTrue(estimate["depth_supported"])
        self.assertAlmostEqual(estimate["estimated_slippage_pct"], 0.0)
        self.assertAlmostEqual(estimate["all_in_cost_per_side"], 0.0011)

    def test_expected_samples_uses_duration_and_interval(self) -> None:
        full_config = CoinbaseExecutionAuditConfig(duration_hours=24, interval_seconds=300)
        quick_config = CoinbaseExecutionAuditConfig(duration_hours=0.1, interval_seconds=60)

        self.assertEqual(full_config.expected_samples, 288)
        self.assertEqual(quick_config.expected_samples, 6)

    def test_report_requires_full_24h_collection_for_acceptable_verdict(self) -> None:
        config = CoinbaseExecutionAuditConfig(duration_hours=24, interval_seconds=300, intended_order_size_usd=2500.0)
        sample = build_orderbook_sample(PRODUCT_BOOK_PAYLOAD, config, "2026-06-13T14:00:00+00:00")
        report = build_execution_cost_report(
            config,
            [sample] * 250,
            [],
            "2026-06-13T14:00:00+00:00",
            "2026-06-14T14:00:00+00:00",
            "2026-06-14T14:00:00+00:00",
            True,
        )

        self.assertEqual(report["verdict"], "COINBASE_EXECUTION_ACCEPTABLE")
        self.assertEqual(report["verdict_basis"], "CUMULATIVE_SAMPLE_REQUIREMENT_MET")
        self.assertEqual(report["expected_samples"], 288)
        self.assertEqual(report["successful_samples"], 250)
        self.assertEqual(report["current_run_samples"], 250)
        self.assertEqual(report["cumulative_successful_samples"], 250)
        self.assertTrue(report["full_24h_complete"])
        self.assertTrue(report["criteria_met"])
        self.assertTrue(report["depth_support"]["supports_intended_trade_size"])

    def test_report_marks_insufficient_sample_size_borderline(self) -> None:
        config = CoinbaseExecutionAuditConfig(duration_hours=24, interval_seconds=300, intended_order_size_usd=2500.0)
        sample = build_orderbook_sample(PRODUCT_BOOK_PAYLOAD, config, "2026-06-13T14:00:00+00:00")
        report = build_execution_cost_report(
            config,
            [sample],
            [],
            "2026-06-13T14:00:00+00:00",
            "2026-06-14T14:00:00+00:00",
            "2026-06-14T14:00:00+00:00",
            True,
        )

        self.assertEqual(report["verdict"], "COINBASE_EXECUTION_BORDERLINE")
        self.assertEqual(report["verdict_basis"], "INSUFFICIENT_CUMULATIVE_SAMPLE_SIZE")
        self.assertFalse(report["full_24h_complete"])
        self.assertFalse(report["criteria_met"])

    def test_report_marks_high_costs_not_acceptable_after_full_collection(self) -> None:
        config = CoinbaseExecutionAuditConfig(
            duration_hours=24,
            interval_seconds=300,
            fee_rate=0.004,
            intended_order_size_usd=2500.0,
        )
        sample = build_orderbook_sample(PRODUCT_BOOK_PAYLOAD, config, "2026-06-13T14:00:00+00:00")
        report = build_execution_cost_report(
            config,
            [sample] * 250,
            [],
            "2026-06-13T14:00:00+00:00",
            "2026-06-14T14:00:00+00:00",
            "2026-06-14T14:00:00+00:00",
            True,
        )

        self.assertEqual(report["verdict"], "COINBASE_EXECUTION_BORDERLINE")
        self.assertEqual(report["verdict_basis"], "INSUFFICIENT_CUMULATIVE_SAMPLE_SIZE")
        self.assertFalse(report["criteria_met"])

    def test_run_writes_report_and_orderbook_csv(self) -> None:
        calls = []
        current_time = datetime(2026, 6, 13, 14, 0, tzinfo=UTC)

        def opener(request, timeout):
            calls.append(request)
            return FakeResponse(PRODUCT_BOOK_PAYLOAD)

        def now_fn():
            return current_time

        def sleeper(seconds):
            nonlocal current_time
            current_time = current_time + timedelta(seconds=seconds)

        with TemporaryDirectory() as temp_dir:
            config = CoinbaseExecutionAuditConfig(
                duration_hours=2 / 3600,
                interval_seconds=1,
                output_dir=Path(temp_dir),
            )
            payload = run_coinbase_execution_cost_audit(
                config,
                opener=opener,
                sleeper=sleeper,
                now_fn=now_fn,
            )

            report_path = Path(payload["artifacts"]["coinbase_execution_cost_audit"])
            csv_path = Path(payload["artifacts"]["coinbase_orderbook_samples"])
            master_path = Path(payload["artifacts"]["coinbase_orderbook_samples_master"])
            final_report_path = Path(payload["artifacts"]["coinbase_execution_cost_audit_final"])
            self.assertTrue(report_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertTrue(master_path.exists())
            self.assertTrue(final_report_path.exists())
            self.assertIn("coinbase_execution_cost_audit_20260613_140000.json", str(report_path))
            self.assertIn("coinbase_orderbook_samples_20260613_140000.csv", str(csv_path))
            report = json.loads(report_path.read_text())
            self.assertEqual(report["expected_samples"], 2)
            self.assertEqual(report["successful_samples"], 2)
            self.assertEqual(report["current_run_samples"], 2)
            self.assertEqual(report["cumulative_successful_samples"], 2)
            self.assertEqual(report["total_attempted_samples"], 2)
            self.assertEqual(report["audit_status"], "PARTIAL_COLLECTION")
            self.assertEqual(report["verdict_basis"], "INSUFFICIENT_CUMULATIVE_SAMPLE_SIZE")
            self.assertEqual(report["collection_mode"], "APPEND_ONLY_MASTER")
            self.assertTrue(report["overwrite_protection_enabled"])
            self.assertEqual(report["master_file_path"], str(master_path))
            self.assertEqual(report["current_run_file_path"], str(csv_path))
            self.assertTrue(report["collection_complete"])
            self.assertFalse(report["full_24h_complete"])
            self.assertFalse(report["strategy_trading_enabled"])
            with csv_path.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertIn("spread_pct", rows[0])
            self.assertEqual(calls[0].get_method(), "GET")
            with master_path.open() as handle:
                master_rows = list(csv.DictReader(handle))
            self.assertEqual(len(master_rows), 2)

    def test_run_refuses_to_overwrite_timestamped_outputs(self) -> None:
        current_time = datetime(2026, 6, 13, 14, 0, tzinfo=UTC)

        def opener(_request, _timeout):
            return FakeResponse(PRODUCT_BOOK_PAYLOAD)

        def now_fn():
            return current_time

        def sleeper(seconds):
            nonlocal current_time
            current_time = current_time + timedelta(seconds=seconds)

        with TemporaryDirectory() as temp_dir:
            config = CoinbaseExecutionAuditConfig(
                duration_hours=1 / 3600,
                interval_seconds=1,
                output_dir=Path(temp_dir),
            )
            run_coinbase_execution_cost_audit(config, opener=opener, sleeper=sleeper, now_fn=now_fn)
            current_time = datetime(2026, 6, 13, 14, 0, tzinfo=UTC)

            with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite existing audit file"):
                run_coinbase_execution_cost_audit(config, opener=opener, sleeper=sleeper, now_fn=now_fn)

    def test_final_merge_report_accepts_completed_sample_requirement(self) -> None:
        rows = [_sample_csv_row(index, all_in_cost=0.0012) for index in range(250)]
        config = CoinbaseExecutionAuditMergeConfig(input_csv_paths=(Path("unused.csv"),))

        report = build_final_execution_cost_report(config, rows)

        self.assertEqual(report["total_successful_samples"], 250)
        self.assertEqual(report["total_failed_samples"], 0)
        self.assertTrue(report["sample_requirement_met"])
        self.assertEqual(report["verdict"], "COINBASE_EXECUTION_ACCEPTABLE")
        self.assertEqual(report["verdict_basis"], "COMPLETED_SAMPLE_REQUIREMENT")
        self.assertTrue(report["shadow_trading_allowed"])
        self.assertFalse(report["live_trading_allowed"])

    def test_final_merge_report_marks_insufficient_samples_borderline(self) -> None:
        rows = [_sample_csv_row(index, all_in_cost=0.0012) for index in range(249)]
        config = CoinbaseExecutionAuditMergeConfig(input_csv_paths=(Path("unused.csv"),))

        report = build_final_execution_cost_report(config, rows)

        self.assertFalse(report["sample_requirement_met"])
        self.assertEqual(report["verdict"], "COINBASE_EXECUTION_BORDERLINE")
        self.assertEqual(report["verdict_basis"], "INSUFFICIENT_SAMPLE_SIZE")
        self.assertFalse(report["shadow_trading_allowed"])
        self.assertFalse(report["live_trading_allowed"])

    def test_merge_writes_deduplicated_final_outputs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            first_path = base / "previous.csv"
            second_path = base / "continuation.csv"
            _write_sample_csv(first_path, [_sample_csv_row(0), _sample_csv_row(1)])
            _write_sample_csv(second_path, [_sample_csv_row(1), _sample_csv_row(2)])
            config = CoinbaseExecutionAuditMergeConfig(
                input_csv_paths=(first_path, second_path),
                output_dir=base,
                minimum_successful_samples=3,
            )

            payload = merge_coinbase_execution_audit_samples(config)
            report = payload["coinbase_execution_cost_audit_final"]
            csv_path = Path(payload["artifacts"]["coinbase_orderbook_samples_final"])

            self.assertEqual(report["loaded_rows"], 4)
            self.assertEqual(report["duplicate_timestamps_removed"], 1)
            self.assertEqual(report["total_successful_samples"], 3)
            self.assertTrue(report["sample_requirement_met"])
            self.assertTrue(csv_path.exists())
            with csv_path.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)


def _sample_csv_row(index: int, all_in_cost: float = 0.0011) -> dict[str, str]:
    return {
        "timestamp": f"2026-06-14T00:{index:02d}:00+00:00",
        "exchange_time": f"2026-06-14T00:{index:02d}:00Z",
        "product_id": "BTC-USD",
        "best_bid": "100.00",
        "best_ask": "100.01",
        "mid_price": "100.005",
        "spread_pct": "0.0001",
        "bid_depth_1": "1000",
        "ask_depth_1": "1000",
        "bid_depth_5": "5000",
        "ask_depth_5": "5000",
        "order_2500_usd_all_in_cost_per_side": str(all_in_cost),
        "order_2500_usd_depth_supported": "True",
    }


def _write_sample_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
