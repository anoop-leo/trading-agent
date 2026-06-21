"""Read-only Coinbase Advanced execution-cost sampling for Phase 1.19."""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


COINBASE_EXECUTION_AUDIT_REPORT = "coinbase_execution_cost_audit_24h.json"
COINBASE_ORDERBOOK_SAMPLES = "coinbase_orderbook_samples_24h.csv"
COINBASE_FINAL_EXECUTION_AUDIT_REPORT = "coinbase_execution_cost_audit_final.json"
COINBASE_FINAL_ORDERBOOK_SAMPLES = "coinbase_orderbook_samples_final.csv"
COINBASE_MASTER_ORDERBOOK_SAMPLES = "coinbase_orderbook_samples_master.csv"
DEFAULT_ORDER_SIZES_USD = (100.0, 500.0, 1000.0, 2500.0, 5000.0)
DEFAULT_COINBASE_BASE_URL = "https://api.coinbase.com/api/v3/brokerage"
LIVE_READY_ALL_IN_COST_RATIO = 0.0015
P95_ALL_IN_COST_RATIO = 0.0025
FULL_COLLECTION_MIN_SUCCESSFUL_SAMPLES = 250

JsonOpener = Callable[..., Any]
SleepFn = Callable[[float], None]
NowFn = Callable[[], datetime]
ProgressCallback = Callable[[dict[str, Any]], None]


class CoinbaseExecutionAuditError(RuntimeError):
    """Raised when Coinbase execution-cost sampling cannot continue."""


@dataclass(frozen=True)
class CoinbaseExecutionAuditConfig:
    product_id: str = "BTC-USD"
    fee_rate: float = 0.001
    order_sizes_usd: tuple[float, ...] = DEFAULT_ORDER_SIZES_USD
    intended_order_size_usd: float = 2500.0
    duration_hours: float = 24.0
    interval_seconds: float = 300.0
    output_dir: Path = Path("outputs")
    base_url: str = DEFAULT_COINBASE_BASE_URL
    timeout_seconds: float = 10.0
    order_book_limit: int = 50
    bearer_token: str | None = None
    overwrite: bool = False

    @property
    def expected_samples(self) -> int:
        if self.interval_seconds <= 0:
            raise CoinbaseExecutionAuditError("interval_seconds must be greater than 0.")
        if self.duration_hours <= 0:
            raise CoinbaseExecutionAuditError("duration_hours must be greater than 0.")
        return max(1, int((self.duration_hours * 3600.0) / self.interval_seconds))


@dataclass(frozen=True)
class CoinbaseExecutionAuditMergeConfig:
    input_csv_paths: tuple[Path, ...]
    output_dir: Path = Path("outputs")
    intended_order_size_usd: float = 2500.0
    failed_samples: int = 0
    minimum_successful_samples: int = FULL_COLLECTION_MIN_SUCCESSFUL_SAMPLES
    max_error_rate: float = 0.01
    max_median_all_in_cost: float = LIVE_READY_ALL_IN_COST_RATIO
    max_p95_all_in_cost: float = P95_ALL_IN_COST_RATIO
    min_depth_support_ratio: float = 0.95


def normalize_coinbase_product_id(product_id: str) -> str:
    """Normalize common BTC symbols to Coinbase product ids."""

    raw = product_id.strip().upper().replace("/", "-").replace("_", "-")
    if "-" in raw:
        return raw
    if raw.endswith("USDT"):
        return f"{raw[:-4]}-USDT"
    if raw.endswith("USD"):
        return f"{raw[:-3]}-USD"
    return raw


def run_coinbase_execution_cost_audit(
    config: CoinbaseExecutionAuditConfig,
    opener: JsonOpener = urlopen,
    sleeper: SleepFn = time.sleep,
    now_fn: NowFn | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Collect Coinbase order-book samples and write Phase 1.19 audit artifacts."""

    now = now_fn or (lambda: datetime.now(UTC))
    expected_samples = config.expected_samples

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = _as_utc(now())
    end_time = start_time + timedelta(hours=config.duration_hours)
    started_at = _isoformat(start_time)
    run_stamp = _timestamp_for_filename(start_time)
    csv_path = output_dir / f"coinbase_orderbook_samples_{run_stamp}.csv"
    report_path = output_dir / f"coinbase_execution_cost_audit_{run_stamp}.json"
    master_path = output_dir / COINBASE_MASTER_ORDERBOOK_SAMPLES
    final_report_path = output_dir / COINBASE_FINAL_EXECUTION_AUDIT_REPORT
    _refuse_existing_file(csv_path, config.overwrite)
    _refuse_existing_file(report_path, config.overwrite)

    samples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    csv_header = _csv_header(config.order_sizes_usd)

    with csv_path.open("w", newline="") as csv_file, _open_master_csv(master_path, csv_header) as master_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_header)
        writer.writeheader()
        csv_file.flush()
        master_writer = csv.DictWriter(master_file, fieldnames=csv_header)

        loop_completed = False
        sample_attempt = 0
        current_time = start_time
        while current_time < end_time:
            sample_attempt += 1
            timestamp = _isoformat(current_time)
            try:
                payload = fetch_public_product_book(config, opener=opener)
                sample = build_orderbook_sample(payload, config, timestamp)
                samples.append(sample)
                row = _csv_row(sample, config.order_sizes_usd)
                writer.writerow(row)
                master_writer.writerow(row)
                csv_file.flush()
                master_file.flush()
                os.fsync(master_file.fileno())
                if progress_callback is not None:
                    progress_callback(
                        {
                            "phase": "coinbase_execution_sample",
                            "sample": sample_attempt,
                            "expected_samples": expected_samples,
                            "product_id": config.product_id,
                            "best_bid": sample["best_bid"],
                            "best_ask": sample["best_ask"],
                            "spread_pct": sample["spread_pct"],
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - keep the 24h sampler alive on transient API failures.
                error = {
                    "timestamp": timestamp,
                    "sample": sample_attempt,
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                }
                errors.append(error)
                if progress_callback is not None:
                    progress_callback({"phase": "coinbase_execution_error", **error})

            sleep_seconds = min(config.interval_seconds, max((end_time - _as_utc(now())).total_seconds(), 0.0))
            if sleep_seconds > 0:
                sleeper(sleep_seconds)
            current_time = _as_utc(now())
            loop_completed = current_time >= end_time

    finished_time = _as_utc(now())
    if finished_time >= end_time:
        loop_completed = True
    finished_at = _isoformat(finished_time)
    master_rows = _load_sample_csv(master_path)
    cumulative_rows, duplicate_samples_removed = _deduplicate_rows_by_timestamp(master_rows)
    cumulative_rows.sort(key=lambda row: row.get("timestamp", ""))
    report = build_execution_cost_report(
        config,
        samples,
        errors,
        started_at,
        finished_at,
        _isoformat(end_time),
        loop_completed,
        cumulative_rows,
        duplicate_samples_removed,
        master_path,
        csv_path,
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    final_report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {
        "coinbase_execution_cost_audit": report,
        "artifacts": {
            "coinbase_execution_cost_audit": str(report_path),
            "coinbase_orderbook_samples": str(csv_path),
            "coinbase_execution_cost_audit_final": str(final_report_path),
            "coinbase_orderbook_samples_master": str(master_path),
        },
    }


def merge_coinbase_execution_audit_samples(
    config: CoinbaseExecutionAuditMergeConfig,
) -> dict[str, Any]:
    """Merge Coinbase order-book sample CSV files and write the final audit verdict."""

    if not config.input_csv_paths:
        raise CoinbaseExecutionAuditError("At least one Coinbase order-book sample CSV is required.")

    missing_paths = [str(path) for path in config.input_csv_paths if not path.exists()]
    if missing_paths:
        raise CoinbaseExecutionAuditError(f"Coinbase sample CSV not found: {', '.join(missing_paths)}")

    loaded_rows: list[dict[str, str]] = []
    sources: list[dict[str, Any]] = []
    for path in config.input_csv_paths:
        rows = _load_sample_csv(path)
        sources.append({"path": str(path), "rows": len(rows)})
        loaded_rows.extend(rows)

    merged_rows, duplicate_timestamps_removed = _deduplicate_rows_by_timestamp(loaded_rows)
    merged_rows.sort(key=lambda row: row.get("timestamp", ""))

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / COINBASE_FINAL_ORDERBOOK_SAMPLES
    report_path = output_dir / COINBASE_FINAL_EXECUTION_AUDIT_REPORT

    _write_merged_sample_csv(csv_path, merged_rows)
    report = build_final_execution_cost_report(
        config,
        merged_rows,
        sources,
        duplicate_timestamps_removed,
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {
        "coinbase_execution_cost_audit_final": report,
        "artifacts": {
            "coinbase_execution_cost_audit_final": str(report_path),
            "coinbase_orderbook_samples_final": str(csv_path),
        },
    }


def build_final_execution_cost_report(
    config: CoinbaseExecutionAuditMergeConfig,
    merged_rows: list[dict[str, str]],
    sources: list[dict[str, Any]] | None = None,
    duplicate_timestamps_removed: int = 0,
) -> dict[str, Any]:
    """Build final Phase 1.19D verdict metrics from merged Coinbase CSV rows."""

    all_in_column = f"{_order_key(config.intended_order_size_usd)}_all_in_cost_per_side"
    depth_column = f"{_order_key(config.intended_order_size_usd)}_depth_supported"
    all_in_costs = [_parse_optional_float(row.get(all_in_column)) for row in merged_rows]
    all_in_costs = [value for value in all_in_costs if value is not None]
    spread_values = [_parse_optional_float(row.get("spread_pct")) for row in merged_rows]
    spread_values = [value for value in spread_values if value is not None]
    depth_values = [_parse_bool(row.get(depth_column)) for row in merged_rows if row.get(depth_column) is not None]

    total_successful_samples = len(merged_rows)
    total_failed_samples = config.failed_samples
    total_attempted_samples = total_successful_samples + total_failed_samples
    error_rate = total_failed_samples / total_attempted_samples if total_attempted_samples else 1.0
    depth_support_ratio = (
        sum(1 for value in depth_values if value) / len(depth_values)
        if depth_values
        else 0.0
    )

    average_all_in_cost = _mean(all_in_costs)
    median_all_in_cost = _median(all_in_costs)
    p95_all_in_cost = _percentile(all_in_costs, 0.95)
    worst_all_in_cost = max(all_in_costs) if all_in_costs else None
    sample_requirement_met = total_successful_samples >= config.minimum_successful_samples
    cost_criteria_pass = (
        sample_requirement_met
        and error_rate <= config.max_error_rate
        and median_all_in_cost is not None
        and median_all_in_cost <= config.max_median_all_in_cost
        and p95_all_in_cost is not None
        and p95_all_in_cost <= config.max_p95_all_in_cost
        and depth_support_ratio >= config.min_depth_support_ratio
    )
    verdict, verdict_basis = _final_execution_verdict(
        sample_requirement_met,
        cost_criteria_pass,
        error_rate,
        depth_support_ratio,
    )

    return {
        "phase": "1.19D",
        "mode": "READ_ONLY_ORDER_BOOK_AUDIT_MERGE",
        "strategy_trading_enabled": False,
        "automated_live_trading_enabled": False,
        "input_sources": sources or [],
        "intended_order_size_usd": config.intended_order_size_usd,
        "all_in_cost_column": all_in_column,
        "depth_support_column": depth_column,
        "loaded_rows": sum(source.get("rows", 0) for source in (sources or [])),
        "duplicate_timestamps_removed": duplicate_timestamps_removed,
        "total_successful_samples": total_successful_samples,
        "total_failed_samples": total_failed_samples,
        "total_attempted_samples": total_attempted_samples,
        "sample_requirement": config.minimum_successful_samples,
        "sample_requirement_met": sample_requirement_met,
        "error_rate": error_rate,
        "average_all_in_cost_per_side": average_all_in_cost,
        "median_all_in_cost_per_side": median_all_in_cost,
        "p95_all_in_cost_per_side": p95_all_in_cost,
        "worst_all_in_cost_per_side": worst_all_in_cost,
        "average_spread_pct": _mean(spread_values),
        "median_spread_pct": _median(spread_values),
        "p95_spread_pct": _percentile(spread_values, 0.95),
        "depth_support_ratio": depth_support_ratio,
        "criteria": {
            "minimum_successful_samples": config.minimum_successful_samples,
            "max_error_rate": config.max_error_rate,
            "max_median_all_in_cost_per_side": config.max_median_all_in_cost,
            "max_p95_all_in_cost_per_side": config.max_p95_all_in_cost,
            "min_depth_support_ratio": config.min_depth_support_ratio,
        },
        "criteria_results": {
            "sample_requirement_met": sample_requirement_met,
            "error_rate_pass": error_rate <= config.max_error_rate,
            "median_all_in_cost_pass": median_all_in_cost is not None
            and median_all_in_cost <= config.max_median_all_in_cost,
            "p95_all_in_cost_pass": p95_all_in_cost is not None and p95_all_in_cost <= config.max_p95_all_in_cost,
            "depth_support_pass": depth_support_ratio >= config.min_depth_support_ratio,
        },
        "verdict": verdict,
        "verdict_basis": verdict_basis,
        "shadow_trading_allowed": cost_criteria_pass,
        "live_trading_allowed": False,
        "notes": [
            "No strategy signals were traded.",
            "No Coinbase order endpoints were called.",
            "Final verdict is based only on merged read-only order-book sample rows.",
            "Live trading remains disabled even when shadow trading is allowed.",
        ],
    }


def fetch_public_product_book(
    config: CoinbaseExecutionAuditConfig,
    opener: JsonOpener = urlopen,
) -> dict[str, Any]:
    """Fetch the Coinbase Advanced public product book."""

    product_id = normalize_coinbase_product_id(config.product_id)
    query = urlencode({"product_id": product_id, "limit": config.order_book_limit})
    url = f"{config.base_url.rstrip('/')}/market/product_book?{query}"
    headers = {
        "Accept": "application/json",
        "Cache-Control": "no-cache",
        "User-Agent": "trading-agent-phase-1.19",
    }
    if config.bearer_token:
        headers["Authorization"] = f"Bearer {config.bearer_token}"
    request = Request(url, headers=headers, method="GET")
    try:
        with opener(request, timeout=config.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise CoinbaseExecutionAuditError(f"Coinbase HTTP {exc.code} while fetching product book.") from exc
    except URLError as exc:
        raise CoinbaseExecutionAuditError(f"Coinbase request failed: {exc.reason}.") from exc
    except json.JSONDecodeError as exc:
        raise CoinbaseExecutionAuditError("Coinbase returned invalid JSON.") from exc

    if not isinstance(payload, dict):
        raise CoinbaseExecutionAuditError("Coinbase returned an unexpected product-book payload.")
    return payload


def build_orderbook_sample(
    payload: dict[str, Any],
    config: CoinbaseExecutionAuditConfig,
    timestamp: str,
) -> dict[str, Any]:
    """Convert one Coinbase product-book payload into audit metrics."""

    pricebook = payload.get("pricebook")
    if not isinstance(pricebook, dict):
        raise CoinbaseExecutionAuditError("Coinbase product-book payload is missing pricebook.")

    bids = _parse_levels(pricebook.get("bids"), descending=True)
    asks = _parse_levels(pricebook.get("asks"), descending=False)
    if not bids or not asks:
        raise CoinbaseExecutionAuditError("Coinbase product-book payload has no bid/ask depth.")

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid_price = (best_bid + best_ask) / 2.0
    if mid_price <= 0:
        raise CoinbaseExecutionAuditError("Coinbase product-book payload has invalid prices.")

    spread_pct = (best_ask - best_bid) / mid_price
    estimates = {
        _order_key(size): estimate_market_order_cost(size, bids, asks, best_bid, best_ask, mid_price, config.fee_rate)
        for size in config.order_sizes_usd
    }
    intended_key = _order_key(config.intended_order_size_usd)
    intended_estimate = estimates.get(intended_key)

    return {
        "timestamp": timestamp,
        "exchange_time": pricebook.get("time"),
        "product_id": pricebook.get("product_id") or normalize_coinbase_product_id(config.product_id),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid_price": mid_price,
        "spread_pct": spread_pct,
        "half_spread_pct": spread_pct / 2.0,
        "bid_depth_1": depth_notional(bids, 1),
        "ask_depth_1": depth_notional(asks, 1),
        "bid_depth_5": depth_notional(bids, 5),
        "ask_depth_5": depth_notional(asks, 5),
        "market_order_estimates": estimates,
        "intended_order_size_usd": config.intended_order_size_usd,
        "intended_all_in_cost_per_side": intended_estimate["all_in_cost_per_side"]
        if intended_estimate is not None
        else None,
        "intended_depth_supported": bool(intended_estimate and intended_estimate["depth_supported"]),
    }


def estimate_market_order_cost(
    order_size_usd: float,
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    best_bid: float,
    best_ask: float,
    mid_price: float,
    fee_rate: float,
) -> dict[str, Any]:
    """Estimate one-side market-order cost from top-of-book and depth impact."""

    buy_fill = _estimate_buy_fill(order_size_usd, asks)
    sell_fill = _estimate_sell_fill(order_size_usd, bids, mid_price)
    buy_slippage = (
        max((buy_fill["weighted_price"] - best_ask) / mid_price, 0.0)
        if buy_fill["depth_supported"]
        else None
    )
    sell_slippage = (
        max((best_bid - sell_fill["weighted_price"]) / mid_price, 0.0)
        if sell_fill["depth_supported"]
        else None
    )

    depth_supported = bool(buy_fill["depth_supported"] and sell_fill["depth_supported"])
    if buy_slippage is None or sell_slippage is None:
        estimated_slippage = None
        all_in = None
    else:
        estimated_slippage = max(buy_slippage, sell_slippage)
        half_spread = max((best_ask - best_bid) / mid_price, 0.0) / 2.0
        all_in = fee_rate + half_spread + estimated_slippage

    return {
        "order_size_usd": order_size_usd,
        "buy_weighted_price": buy_fill["weighted_price"],
        "sell_weighted_price": sell_fill["weighted_price"],
        "buy_slippage_pct": buy_slippage,
        "sell_slippage_pct": sell_slippage,
        "estimated_slippage_pct": estimated_slippage,
        "all_in_cost_per_side": all_in,
        "depth_supported": depth_supported,
    }


def depth_notional(levels: list[tuple[float, float]], count: int) -> float:
    """Return quote notional depth for the first N price levels."""

    return sum(price * size for price, size in levels[:count])


def build_execution_cost_report(
    config: CoinbaseExecutionAuditConfig,
    samples: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    started_at: str,
    finished_at: str,
    scheduled_end_at: str | None = None,
    loop_completed: bool = True,
    cumulative_rows: list[dict[str, str]] | None = None,
    duplicate_samples_removed: int = 0,
    master_file_path: Path | None = None,
    current_run_file_path: Path | None = None,
) -> dict[str, Any]:
    expected_samples = config.expected_samples
    current_run_samples = len(samples)
    failed_samples = len(errors)
    cumulative_rows = cumulative_rows if cumulative_rows is not None else [
        {key: str(value) for key, value in _csv_row(sample, config.order_sizes_usd).items()}
        for sample in samples
    ]
    cumulative_successful_samples = len(cumulative_rows)
    cumulative_failed_samples = failed_samples
    total_attempted_samples = current_run_samples + failed_samples
    cumulative_attempted_samples = cumulative_successful_samples + cumulative_failed_samples
    collection_complete = loop_completed
    full_24h_complete = (
        config.duration_hours >= 24.0
        and loop_completed
        and current_run_samples >= FULL_COLLECTION_MIN_SUCCESSFUL_SAMPLES
    )

    all_in_column = f"{_order_key(config.intended_order_size_usd)}_all_in_cost_per_side"
    depth_column = f"{_order_key(config.intended_order_size_usd)}_depth_supported"
    spread_values = [_parse_optional_float(row.get("spread_pct")) for row in cumulative_rows]
    spread_values = [value for value in spread_values if value is not None]
    intended_costs = [_parse_optional_float(row.get(all_in_column)) for row in cumulative_rows]
    intended_costs = [value for value in intended_costs if value is not None]
    depth_values = [_parse_bool(row.get(depth_column)) for row in cumulative_rows if row.get(depth_column) is not None]
    intended_depth_supported_count = sum(1 for value in depth_values if value)
    reliability_error_rate = cumulative_failed_samples / cumulative_attempted_samples if cumulative_attempted_samples else 1.0
    no_api_reliability_issues = failed_samples == 0
    depth_support_ratio = intended_depth_supported_count / len(depth_values) if depth_values else 0.0
    order_size_summaries = {
        _order_key(size): _summarize_order_size_rows(cumulative_rows, size)
        for size in config.order_sizes_usd
    }

    success_criteria = {
        "median_all_in_cost_per_side_lte": LIVE_READY_ALL_IN_COST_RATIO,
        "p95_all_in_cost_per_side_lte": P95_ALL_IN_COST_RATIO,
        "requires_depth_support_for_intended_order_size": True,
        "requires_no_api_reliability_issues": True,
    }
    criteria_results = {
        "median_all_in_cost_per_side": _median(intended_costs),
        "p95_all_in_cost_per_side": _percentile(intended_costs, 0.95),
        "depth_support_ratio": depth_support_ratio,
        "no_api_reliability_issues": no_api_reliability_issues,
    }
    cost_criteria_met = (
        criteria_results["median_all_in_cost_per_side"] is not None
        and criteria_results["median_all_in_cost_per_side"] <= LIVE_READY_ALL_IN_COST_RATIO
        and criteria_results["p95_all_in_cost_per_side"] is not None
        and criteria_results["p95_all_in_cost_per_side"] <= P95_ALL_IN_COST_RATIO
        and depth_support_ratio >= 0.95
        and reliability_error_rate <= 0.01
    )
    verdict, verdict_basis = _execution_verdict(
        cost_criteria_met,
        cumulative_successful_samples,
    )

    return {
        "phase": "1.19",
        "mode": "READ_ONLY_ORDER_BOOK_AUDIT",
        "collection_mode": "APPEND_ONLY_MASTER",
        "overwrite_protection_enabled": True,
        "strategy_trading_enabled": False,
        "automated_live_trading_enabled": False,
        "product_id": normalize_coinbase_product_id(config.product_id),
        "base_url": config.base_url,
        "fee_rate": config.fee_rate,
        "order_sizes_usd": list(config.order_sizes_usd),
        "intended_order_size_usd": config.intended_order_size_usd,
        "started_at": started_at,
        "finished_at": finished_at,
        "scheduled_end_at": scheduled_end_at,
        "interval_seconds": config.interval_seconds,
        "duration_hours": config.duration_hours,
        "expected_samples": expected_samples,
        "minimum_successful_samples_for_full_verdict": FULL_COLLECTION_MIN_SUCCESSFUL_SAMPLES,
        "current_run_samples": current_run_samples,
        "successful_samples": current_run_samples,
        "failed_samples": failed_samples,
        "total_attempted_samples": total_attempted_samples,
        "cumulative_successful_samples": cumulative_successful_samples,
        "cumulative_failed_samples": cumulative_failed_samples,
        "cumulative_attempted_samples": cumulative_attempted_samples,
        "duplicate_samples_removed": duplicate_samples_removed,
        "master_file_path": str(master_file_path) if master_file_path is not None else None,
        "current_run_file_path": str(current_run_file_path) if current_run_file_path is not None else None,
        "collection_complete": collection_complete,
        "full_24h_complete": full_24h_complete,
        "audit_status": "CUMULATIVE_READY" if cumulative_successful_samples >= FULL_COLLECTION_MIN_SUCCESSFUL_SAMPLES else "PARTIAL_COLLECTION",
        "read_only_connection": current_run_samples > 0,
        "error_rate": reliability_error_rate,
        "api_reliability": {
            "failed_samples": cumulative_failed_samples,
            "error_rate": reliability_error_rate,
            "no_api_reliability_issues": reliability_error_rate <= 0.01,
            "errors": errors,
        },
        "average_spread_pct": _mean(spread_values),
        "median_spread_pct": _median(spread_values),
        "p95_spread_pct": _percentile(spread_values, 0.95),
        "average_estimated_all_in_cost_per_side": _mean(intended_costs),
        "median_estimated_all_in_cost_per_side": _median(intended_costs),
        "p95_estimated_all_in_cost_per_side": _percentile(intended_costs, 0.95),
        "order_size_costs": order_size_summaries,
        "depth_support": {
            "intended_order_size_usd": config.intended_order_size_usd,
            "supported_samples": intended_depth_supported_count,
            "support_ratio": depth_support_ratio,
            "supports_intended_trade_size": depth_support_ratio >= 0.95 and cumulative_successful_samples > 0,
        },
        "success_criteria": success_criteria,
        "criteria_results": criteria_results,
        "criteria_met": cost_criteria_met and cumulative_successful_samples >= FULL_COLLECTION_MIN_SUCCESSFUL_SAMPLES,
        "verdict": verdict,
        "verdict_basis": verdict_basis,
        "notes": [
            "No strategy signals were traded.",
            "No Coinbase order endpoints were called.",
            "all_in_cost_per_side = fee_rate + half_spread_pct + estimated_slippage_pct.",
            "Cost fields are stored as ratios; multiply by 100 for percent.",
        ],
    }


def _execution_verdict(
    cost_criteria_met: bool,
    cumulative_successful_samples: int,
) -> tuple[str, str]:
    if cumulative_successful_samples >= FULL_COLLECTION_MIN_SUCCESSFUL_SAMPLES and cost_criteria_met:
        return "COINBASE_EXECUTION_ACCEPTABLE", "CUMULATIVE_SAMPLE_REQUIREMENT_MET"
    return "COINBASE_EXECUTION_BORDERLINE", "INSUFFICIENT_CUMULATIVE_SAMPLE_SIZE"


def _summarize_order_size(samples: list[dict[str, Any]], order_size_usd: float) -> dict[str, Any]:
    key = _order_key(order_size_usd)
    estimates = [
        sample["market_order_estimates"][key]
        for sample in samples
        if key in sample.get("market_order_estimates", {})
    ]
    all_in = [
        estimate["all_in_cost_per_side"]
        for estimate in estimates
        if estimate.get("all_in_cost_per_side") is not None
    ]
    slippage = [
        estimate["estimated_slippage_pct"]
        for estimate in estimates
        if estimate.get("estimated_slippage_pct") is not None
    ]
    supported = sum(1 for estimate in estimates if estimate.get("depth_supported"))
    support_ratio = supported / len(estimates) if estimates else 0.0
    return {
        "order_size_usd": order_size_usd,
        "average_estimated_slippage_pct": _mean(slippage),
        "median_estimated_slippage_pct": _median(slippage),
        "p95_estimated_slippage_pct": _percentile(slippage, 0.95),
        "average_all_in_cost_per_side": _mean(all_in),
        "median_all_in_cost_per_side": _median(all_in),
        "p95_all_in_cost_per_side": _percentile(all_in, 0.95),
        "depth_supported_samples": supported,
        "support_ratio": support_ratio,
    }


def _summarize_order_size_rows(rows: list[dict[str, str]], order_size_usd: float) -> dict[str, Any]:
    key = _order_key(order_size_usd)
    all_in = [_parse_optional_float(row.get(f"{key}_all_in_cost_per_side")) for row in rows]
    all_in = [value for value in all_in if value is not None]
    slippage = [_parse_optional_float(row.get(f"{key}_estimated_slippage_pct")) for row in rows]
    slippage = [value for value in slippage if value is not None]
    depth_values = [_parse_bool(row.get(f"{key}_depth_supported")) for row in rows if row.get(f"{key}_depth_supported") is not None]
    supported = sum(1 for value in depth_values if value)
    support_ratio = supported / len(depth_values) if depth_values else 0.0
    return {
        "order_size_usd": order_size_usd,
        "average_estimated_slippage_pct": _mean(slippage),
        "median_estimated_slippage_pct": _median(slippage),
        "p95_estimated_slippage_pct": _percentile(slippage, 0.95),
        "average_all_in_cost_per_side": _mean(all_in),
        "median_all_in_cost_per_side": _median(all_in),
        "p95_all_in_cost_per_side": _percentile(all_in, 0.95),
        "depth_supported_samples": supported,
        "support_ratio": support_ratio,
    }


def _parse_levels(raw_levels: Any, descending: bool) -> list[tuple[float, float]]:
    if not isinstance(raw_levels, list):
        raise CoinbaseExecutionAuditError("Coinbase pricebook levels are malformed.")
    levels: list[tuple[float, float]] = []
    for raw in raw_levels:
        if not isinstance(raw, dict):
            raise CoinbaseExecutionAuditError("Coinbase pricebook level is malformed.")
        try:
            price = float(raw["price"])
            size = float(raw["size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CoinbaseExecutionAuditError("Coinbase pricebook level has invalid price or size.") from exc
        if price > 0 and size > 0:
            levels.append((price, size))
    return sorted(levels, key=lambda level: level[0], reverse=descending)


def _estimate_buy_fill(order_size_usd: float, asks: list[tuple[float, float]]) -> dict[str, Any]:
    remaining_quote = order_size_usd
    base_received = 0.0
    for price, size in asks:
        level_notional = price * size
        quote_at_level = min(remaining_quote, level_notional)
        base_received += quote_at_level / price
        remaining_quote -= quote_at_level
        if remaining_quote <= 1e-9:
            break
    if remaining_quote > 1e-9 or base_received <= 0:
        return {"weighted_price": None, "depth_supported": False}
    return {"weighted_price": order_size_usd / base_received, "depth_supported": True}


def _estimate_sell_fill(
    order_size_usd: float,
    bids: list[tuple[float, float]],
    mid_price: float,
) -> dict[str, Any]:
    base_to_sell = order_size_usd / mid_price
    remaining_base = base_to_sell
    quote_received = 0.0
    for price, size in bids:
        base_at_level = min(remaining_base, size)
        quote_received += base_at_level * price
        remaining_base -= base_at_level
        if remaining_base <= 1e-12:
            break
    if remaining_base > 1e-12 or base_to_sell <= 0:
        return {"weighted_price": None, "depth_supported": False}
    return {"weighted_price": quote_received / base_to_sell, "depth_supported": True}


def _csv_header(order_sizes_usd: tuple[float, ...]) -> list[str]:
    header = [
        "timestamp",
        "exchange_time",
        "product_id",
        "best_bid",
        "best_ask",
        "mid_price",
        "spread_pct",
        "bid_depth_1",
        "ask_depth_1",
        "bid_depth_5",
        "ask_depth_5",
    ]
    for size in order_sizes_usd:
        key = _order_key(size)
        header.extend(
            [
                f"{key}_estimated_slippage_pct",
                f"{key}_all_in_cost_per_side",
                f"{key}_depth_supported",
            ]
        )
    return header


def _csv_row(sample: dict[str, Any], order_sizes_usd: tuple[float, ...]) -> dict[str, Any]:
    row = {
        "timestamp": sample["timestamp"],
        "exchange_time": sample.get("exchange_time"),
        "product_id": sample["product_id"],
        "best_bid": sample["best_bid"],
        "best_ask": sample["best_ask"],
        "mid_price": sample["mid_price"],
        "spread_pct": sample["spread_pct"],
        "bid_depth_1": sample["bid_depth_1"],
        "ask_depth_1": sample["ask_depth_1"],
        "bid_depth_5": sample["bid_depth_5"],
        "ask_depth_5": sample["ask_depth_5"],
    }
    for size in order_sizes_usd:
        key = _order_key(size)
        estimate = sample["market_order_estimates"][key]
        row[f"{key}_estimated_slippage_pct"] = estimate["estimated_slippage_pct"]
        row[f"{key}_all_in_cost_per_side"] = estimate["all_in_cost_per_side"]
        row[f"{key}_depth_supported"] = estimate["depth_supported"]
    return row


def _load_sample_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise CoinbaseExecutionAuditError(f"Coinbase sample CSV has no header: {path}")
        if "timestamp" not in reader.fieldnames:
            raise CoinbaseExecutionAuditError(f"Coinbase sample CSV is missing timestamp column: {path}")
        return [dict(row) for row in reader]


def _deduplicate_rows_by_timestamp(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    by_timestamp: dict[str, dict[str, str]] = {}
    duplicates = 0
    for row in rows:
        timestamp = row.get("timestamp")
        if not timestamp:
            continue
        if timestamp in by_timestamp:
            duplicates += 1
            continue
        by_timestamp[timestamp] = row
    return list(by_timestamp.values()), duplicates


def _write_merged_sample_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = _merged_fieldnames(rows)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _merged_fieldnames(rows: list[dict[str, str]]) -> list[str]:
    preferred = [
        "timestamp",
        "exchange_time",
        "product_id",
        "best_bid",
        "best_ask",
        "mid_price",
        "spread_pct",
        "bid_depth_1",
        "ask_depth_1",
        "bid_depth_5",
        "ask_depth_5",
    ]
    seen = {key for row in rows for key in row}
    ordered = [key for key in preferred if key in seen]
    ordered.extend(sorted(key for key in seen if key not in set(ordered)))
    return ordered or ["timestamp"]


def _open_master_csv(path: Path, fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    csv_file = path.open("a", newline="")
    if not file_exists:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        csv_file.flush()
        os.fsync(csv_file.fileno())
    return csv_file


def _refuse_existing_file(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise CoinbaseExecutionAuditError("Refusing to overwrite existing audit file.")


def _timestamp_for_filename(value: datetime) -> str:
    return _as_utc(value).strftime("%Y%m%d_%H%M%S")


def _final_execution_verdict(
    sample_requirement_met: bool,
    cost_criteria_pass: bool,
    error_rate: float,
    depth_support_ratio: float,
) -> tuple[str, str]:
    if not sample_requirement_met:
        return "COINBASE_EXECUTION_BORDERLINE", "INSUFFICIENT_SAMPLE_SIZE"
    if cost_criteria_pass:
        return "COINBASE_EXECUTION_ACCEPTABLE", "COMPLETED_SAMPLE_REQUIREMENT"
    if error_rate <= 0.05 or depth_support_ratio >= 0.9:
        return "COINBASE_EXECUTION_BORDERLINE", "FAILED_COST_CRITERIA"
    return "COINBASE_EXECUTION_NOT_ACCEPTABLE", "FAILED_COST_CRITERIA"


def _order_key(order_size_usd: float) -> str:
    normalized = int(order_size_usd) if float(order_size_usd).is_integer() else order_size_usd
    return f"order_{normalized}_usd"


def _isoformat(value: datetime) -> str:
    return _as_utc(value).isoformat()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _median(values: list[float | None]) -> float | None:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    midpoint = len(clean) // 2
    if len(clean) % 2:
        return clean[midpoint]
    return (clean[midpoint - 1] + clean[midpoint]) / 2.0


def _percentile(values: list[float | None], percentile: float) -> float | None:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return clean[int(rank)]
    weight = rank - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def _parse_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
