"""Cross-asset validation for Agent Aggressive robustness research."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from backtesting.backtest_engine import BacktestConfig, load_or_download_timeframes, run_backtest
from backtesting.benchmarks.research import profit_capture_payload
from backtesting.benchmarks.strategies import BenchmarkResult
from data.equity_data_adapter import (
    EquityDataAdapterError,
    EquityDataResult,
    ProviderAttempt,
    load_equity_data,
    write_data_provider_diagnostics,
)


CRYPTO_ASSETS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
EQUITY_ASSETS = ("SPY", "QQQ")
OPTIONAL_ASSETS = ("IWM", "DIA", "TQQQ", "NVDA")
DEFAULT_CROSS_ASSETS = (*CRYPTO_ASSETS, *EQUITY_ASSETS)
COMMON_HISTORY_TARGET_START = "2018-01-01"
COMMON_HISTORY_MINIMUM_START = "2020-01-01"
ProgressCallback = Callable[[dict[str, Any]], None]


def run_cross_asset_validation(
    config: BacktestConfig,
    assets: tuple[str, ...] = DEFAULT_CROSS_ASSETS,
    include_optional: bool = False,
    cached_data_by_symbol: dict[str, dict[str, pd.DataFrame]] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run Agent Aggressive across assets without asset-specific tuning."""

    requested_assets = _normalize_assets(assets, include_optional)
    target_start = _target_start(config.start)
    timeframes = _ordered_timeframes(config.primary_timeframe, config.timeframes)
    loaded_frames: dict[str, dict[str, pd.DataFrame]] = {}
    asset_errors: dict[str, str] = {}
    availability: dict[str, dict[str, str]] = {}

    for asset in requested_assets:
        _emit(progress_callback, {"phase": "cross_asset_load", "symbol": asset})
        asset_config = _agent_aggressive_config(replace(config, symbol=asset, start=target_start))
        try:
            frames = dict((cached_data_by_symbol or {}).get(asset, {}))
            missing = tuple(timeframe for timeframe in timeframes if timeframe not in frames)
            if missing:
                frames.update(load_or_download_timeframes(asset_config, missing))
            loaded_frames[asset] = frames
            availability[asset] = _availability(frames, timeframes)
        except Exception as exc:  # noqa: BLE001 - report validation failures per asset.
            asset_errors[asset] = str(exc)

    successful_assets = tuple(asset for asset in requested_assets if asset in loaded_frames)
    if successful_assets:
        common_start = max(pd.Timestamp(target_start, tz="UTC"), *[
            pd.Timestamp(availability[asset]["start"]) for asset in successful_assets
        ])
        common_end = _common_end(config.end, successful_assets, availability)
    else:
        common_start = pd.Timestamp(target_start, tz="UTC")
        common_end = None

    benchmark_results: dict[str, BenchmarkResult] = {}
    rows: dict[str, Any] = {}
    for asset in requested_assets:
        if asset in asset_errors:
            rows[asset] = _failed_asset_row(asset, asset_errors[asset])
            continue
        if common_end is None or common_end <= common_start:
            rows[asset] = _failed_asset_row(asset, "No overlapping common history across loaded assets.")
            continue
        _emit(progress_callback, {"phase": "cross_asset_backtest", "symbol": asset})
        asset_config = _agent_aggressive_config(
            replace(
                config,
                symbol=asset,
                start=common_start.isoformat(),
                end=common_end.isoformat(),
            )
        )
        try:
            result = run_backtest(
                asset_config,
                cached_data=loaded_frames[asset],
                progress_callback=progress_callback,
            )
            benchmark = BenchmarkResult(
                name=asset,
                symbol=asset,
                start_date=result.start_date,
                end_date=result.end_date,
                initial_capital=result.initial_capital,
                final_equity=result.final_equity,
                metrics=result.metrics,
                trades=result.trades,
                equity_curve=result.equity_curve,
            )
            benchmark_results[asset] = benchmark
            capture = profit_capture_payload({asset: benchmark}, loaded_frames[asset][config.primary_timeframe])
            rows[asset] = _asset_row(asset, benchmark, capture["strategies"][asset])
        except Exception as exc:  # noqa: BLE001 - report validation failures per asset.
            rows[asset] = _failed_asset_row(asset, str(exc))

    for asset, row in rows.items():
        row["failure_analysis"] = failure_analysis(asset, row)

    payload = cross_asset_validation_payload(
        rows,
        requested_assets=requested_assets,
        common_start=common_start.isoformat(),
        common_end=common_end.isoformat() if common_end is not None else None,
        target_start=target_start,
    )
    path = write_cross_asset_validation(config.output_dir, payload)
    return {
        "cross_asset_validation": payload,
        "artifacts": {"cross_asset_validation": str(path)},
    }


def run_equity_validation(
    config: BacktestConfig,
    equity_assets: tuple[str, ...] = EQUITY_ASSETS,
    include_optional: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Validate equity data adapters and rerun Agent Aggressive cross-asset validation."""

    requested_equities = _normalize_equity_assets(equity_assets, include_optional)
    timeframes = _ordered_timeframes(config.primary_timeframe, config.timeframes)
    equity_results: list[EquityDataResult] = []
    equity_failures: dict[str, str] = {}
    failed_attempts: dict[str, list[ProviderAttempt]] = {}
    cached_data_by_symbol: dict[str, dict[str, pd.DataFrame]] = {}

    for symbol in requested_equities:
        _emit(progress_callback, {"phase": "equity_validation_load", "symbol": symbol})
        try:
            result = load_equity_data(
                symbol=symbol,
                start=config.start,
                end=config.end,
                timeframes=timeframes,
                cache_dir=config.cache_dir,
                refresh_cache=config.refresh_cache,
                timeout_seconds=config.request_timeout_seconds,
            )
            equity_results.append(result)
            cached_data_by_symbol[symbol] = result.frames
        except EquityDataAdapterError as exc:
            equity_failures[symbol] = str(exc)
            failed_attempts[symbol] = exc.attempts
        except Exception as exc:  # noqa: BLE001 - keep validating remaining symbols.
            equity_failures[symbol] = str(exc)
            failed_attempts[symbol] = []

    diagnostics_path = write_data_provider_diagnostics(config.output_dir, equity_results)
    _append_failed_diagnostics(diagnostics_path, equity_failures, failed_attempts)
    cross_assets = (*CRYPTO_ASSETS, *requested_equities)
    cross_payload = run_cross_asset_validation(
        config,
        assets=cross_assets,
        cached_data_by_symbol=cached_data_by_symbol,
        progress_callback=progress_callback,
    )["cross_asset_validation"]
    report = equity_validation_report_payload(
        equity_results=equity_results,
        equity_failures=equity_failures,
        cross_asset_report=cross_payload,
        requested_equities=requested_equities,
    )
    report_path = write_equity_validation_report(config.output_dir, report)
    return {
        "equity_validation_report": report,
        "cross_asset_validation": cross_payload,
        "artifacts": {
            "equity_validation_report": str(report_path),
            "data_provider_diagnostics": str(diagnostics_path),
            "cross_asset_validation": str(config.output_dir / "cross_asset_validation.json"),
        },
    }


def cross_asset_validation_payload(
    assets: dict[str, dict[str, Any]],
    *,
    requested_assets: tuple[str, ...] = DEFAULT_CROSS_ASSETS,
    common_start: str | None = None,
    common_end: str | None = None,
    target_start: str = COMMON_HISTORY_TARGET_START,
) -> dict[str, Any]:
    """Build the Phase 1.16 JSON report from per-asset rows."""

    scored = {asset: row for asset, row in assets.items() if row.get("status") == "OK"}
    rankings = {
        "by_return": _rank_assets(scored, "total_return_pct", reverse=True),
        "by_sharpe": _rank_assets(scored, "sharpe_ratio", reverse=True),
        "by_drawdown": _rank_assets(scored, "max_drawdown_pct", reverse=False),
        "by_robustness_score": _rank_assets(scored, "robustness_score", reverse=True),
    }
    class_analysis = {
        "crypto_average": asset_class_average(assets, CRYPTO_ASSETS),
        "equity_average": asset_class_average(assets, EQUITY_ASSETS),
    }
    success = success_criteria(assets)
    return {
        "goal": "Validate whether Agent Aggressive generalizes across markets without asset-specific tuning.",
        "strategy": "agent_aggressive",
        "strategy_changes": "none",
        "target_start": target_start,
        "minimum_start": COMMON_HISTORY_MINIMUM_START,
        "common_start": common_start,
        "common_end": common_end,
        "requested_assets": list(requested_assets),
        "failed_assets": [asset for asset, row in assets.items() if row.get("status") != "OK"],
        "metrics": [
            "total_return_pct",
            "cagr",
            "sharpe_ratio",
            "max_drawdown_pct",
            "profit_factor",
            "win_rate",
            "total_trades",
            "profit_capture_ratio",
        ],
        "robustness_score_definition": {
            "positive_return": 20,
            "sharpe_gt_0_8": 20,
            "drawdown_lt_25": 20,
            "profit_factor_gt_1": 20,
            "win_rate_gt_40": 20,
        },
        "asset_class_analysis": class_analysis,
        "success_criteria": success,
        "rankings": rankings,
        "recommended_production_assets": recommended_production_assets(assets),
        "trend_following_assessment": trend_following_assessment(assets, success),
        "assets": assets,
    }


def equity_validation_report_payload(
    *,
    equity_results: list[EquityDataResult],
    equity_failures: dict[str, str],
    cross_asset_report: dict[str, Any],
    requested_equities: tuple[str, ...] = EQUITY_ASSETS,
) -> dict[str, Any]:
    assets = cross_asset_report.get("assets", {})
    equity_rows = {
        symbol: assets.get(symbol, _failed_asset_row(symbol, equity_failures.get(symbol, "validation failed")))
        for symbol in requested_equities
    }
    equity_metrics = {
        symbol: {
            "return": row.get("total_return_pct"),
            "cagr": row.get("cagr"),
            "sharpe": row.get("sharpe_ratio"),
            "drawdown": row.get("max_drawdown_pct"),
            "profit_factor": row.get("profit_factor"),
            "win_rate": row.get("win_rate"),
            "trade_count": row.get("total_trades"),
            "profit_capture_ratio": row.get("profit_capture_ratio"),
            "robustness_score": row.get("robustness_score"),
            "status": row.get("status"),
            "error": row.get("error"),
        }
        for symbol, row in equity_rows.items()
    }
    data_validation = _equity_data_validation_payload(equity_results, equity_failures, requested_equities)
    success = equity_success_criteria(equity_rows, data_validation)
    assessment = universal_trend_following_assessment(cross_asset_report, success)
    return {
        "goal": "Fix equity data ingestion and validate Agent Aggressive across equities without strategy changes.",
        "strategy": "agent_aggressive",
        "strategy_changes": "none",
        "provider_fallback_chain": ["Yahoo Finance", "Stooq", "Alpha Vantage", "Twelve Data"],
        "data_validation": data_validation,
        "equity_metrics": equity_metrics,
        "equity_rankings": {
            "by_return": _rank_assets({k: v for k, v in equity_rows.items() if v.get("status") == "OK"}, "total_return_pct", True),
            "by_sharpe": _rank_assets({k: v for k, v in equity_rows.items() if v.get("status") == "OK"}, "sharpe_ratio", True),
            "by_drawdown": _rank_assets({k: v for k, v in equity_rows.items() if v.get("status") == "OK"}, "max_drawdown_pct", False),
            "by_robustness_score": _rank_assets({k: v for k, v in equity_rows.items() if v.get("status") == "OK"}, "robustness_score", True),
        },
        "updated_cross_asset_rankings": cross_asset_report.get("rankings", {}),
        "equity_vs_crypto_analysis": cross_asset_report.get("asset_class_analysis", {}),
        "success_criteria": success,
        "trend_following_assessment": assessment,
        "recommended_production_assets": cross_asset_report.get("recommended_production_assets", []),
        "assets": assets,
    }


def robustness_score(row: dict[str, Any]) -> int:
    score = 0
    if _float(row.get("total_return_pct")) is not None and _float(row.get("total_return_pct")) > 0:
        score += 20
    if _float(row.get("sharpe_ratio")) is not None and _float(row.get("sharpe_ratio")) > 0.8:
        score += 20
    if _float(row.get("max_drawdown_pct")) is not None and _float(row.get("max_drawdown_pct")) < 25:
        score += 20
    if _float(row.get("profit_factor")) is not None and _float(row.get("profit_factor")) > 1:
        score += 20
    if _float(row.get("win_rate")) is not None and _float(row.get("win_rate")) > 40:
        score += 20
    return score


def asset_class_average(assets: dict[str, dict[str, Any]], class_assets: tuple[str, ...]) -> dict[str, Any]:
    rows = [assets[asset] for asset in class_assets if assets.get(asset, {}).get("status") == "OK"]
    return {
        "assets": [asset for asset in class_assets if assets.get(asset, {}).get("status") == "OK"],
        "failed_assets": [asset for asset in class_assets if asset in assets and assets[asset].get("status") != "OK"],
        "average_return": _mean_metric(rows, "total_return_pct"),
        "average_sharpe": _mean_metric(rows, "sharpe_ratio"),
        "average_drawdown": _mean_metric(rows, "max_drawdown_pct"),
    }


def success_criteria(assets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    crypto_rows = [assets.get(asset) for asset in CRYPTO_ASSETS]
    equity_rows = [assets.get(asset) for asset in EQUITY_ASSETS]
    profitable_count = sum(
        1
        for asset in (*CRYPTO_ASSETS, *EQUITY_ASSETS)
        if assets.get(asset, {}).get("status") == "OK"
        and (_float(assets[asset].get("total_return_pct")) or 0.0) > 0
    )
    crypto_average = asset_class_average(assets, CRYPTO_ASSETS)
    equity_average = asset_class_average(assets, EQUITY_ASSETS)
    crypto_positive = all(row is not None and row.get("status") == "OK" and (_float(row.get("total_return_pct")) or 0.0) > 0 for row in crypto_rows)
    equity_positive = all(row is not None and row.get("status") == "OK" and (_float(row.get("total_return_pct")) or 0.0) > 0 for row in equity_rows)
    return {
        "crypto_sharpe_gt_0_8_average": (_float(crypto_average.get("average_sharpe")) or float("-inf")) > 0.8,
        "crypto_positive_return_all_assets": crypto_positive,
        "equity_sharpe_gt_0_8_average": (_float(equity_average.get("average_sharpe")) or float("-inf")) > 0.8,
        "equity_positive_return_all_assets": equity_positive,
        "profitable_assets_count": profitable_count,
        "at_least_4_of_5_profitable": profitable_count >= 4,
    }


def recommended_production_assets(assets: dict[str, dict[str, Any]]) -> list[str]:
    return [
        asset
        for asset, row in assets.items()
        if row.get("status") == "OK"
        and (_float(row.get("total_return_pct")) or 0.0) > 0
        and (_float(row.get("max_drawdown_pct")) or float("inf")) < 25
        and int(row.get("robustness_score", 0)) >= 60
    ]


def trend_following_assessment(assets: dict[str, dict[str, Any]], success: dict[str, Any]) -> dict[str, str]:
    if success.get("at_least_4_of_5_profitable") and success.get("crypto_sharpe_gt_0_8_average"):
        verdict = "validated"
        reason = "Agent Aggressive produced broad positive returns with acceptable crypto risk-adjusted performance."
    elif any(row.get("status") != "OK" for row in assets.values()):
        verdict = "inconclusive"
        reason = "One or more requested assets could not be validated with the available OHLCV history."
    else:
        verdict = "not_validated"
        reason = "Cross-asset results did not meet the profitability and Sharpe thresholds."
    return {"verdict": verdict, "reason": reason}


def equity_success_criteria(
    equity_rows: dict[str, dict[str, Any]],
    data_validation: dict[str, Any],
) -> dict[str, Any]:
    spy_loaded = int(data_validation.get("SPY", {}).get("rows") or 0) > 0
    qqq_loaded = int(data_validation.get("QQQ", {}).get("rows") or 0) > 0
    spy_validation_passed = data_validation.get("SPY", {}).get("validation") == "passed"
    qqq_validation_passed = data_validation.get("QQQ", {}).get("validation") == "passed"
    spy_backtest = equity_rows.get("SPY", {}).get("status") == "OK"
    qqq_backtest = equity_rows.get("QQQ", {}).get("status") == "OK"
    sharpe_values = [
        _float(row.get("sharpe_ratio"))
        for row in equity_rows.values()
        if row.get("status") == "OK"
    ]
    positive_required = all(
        equity_rows.get(symbol, {}).get("status") == "OK"
        and (_float(equity_rows[symbol].get("total_return_pct")) or 0.0) > 0
        for symbol in ("SPY", "QQQ")
    )
    return {
        "spy_data_loads": spy_loaded,
        "qqq_data_loads": qqq_loaded,
        "spy_data_validation_passed": spy_validation_passed,
        "qqq_data_validation_passed": qqq_validation_passed,
        "spy_backtest_complete": spy_backtest,
        "qqq_backtest_complete": qqq_backtest,
        "sharpe_gt_0_8_on_at_least_one_equity": any((value or float("-inf")) > 0.8 for value in sharpe_values),
        "positive_return_on_both_required_equities": positive_required,
        "validation_successful": (
            spy_loaded
            and qqq_loaded
            and spy_backtest
            and qqq_backtest
            and any((value or float("-inf")) > 0.8 for value in sharpe_values)
            and positive_required
        ),
    }


def universal_trend_following_assessment(
    cross_asset_report: dict[str, Any],
    equity_success: dict[str, Any],
) -> dict[str, Any]:
    class_analysis = cross_asset_report.get("asset_class_analysis", {})
    crypto_average = class_analysis.get("crypto_average", {})
    equity_average = class_analysis.get("equity_average", {})
    crypto_positive = cross_asset_report.get("success_criteria", {}).get("crypto_positive_return_all_assets", False)
    equity_positive = equity_success.get("positive_return_on_both_required_equities", False)
    crypto_sharpe = _float(crypto_average.get("average_sharpe")) or 0.0
    equity_sharpe = _float(equity_average.get("average_sharpe")) or 0.0
    assets = cross_asset_report.get("assets", {})
    scores = [
        int(row.get("robustness_score", 0))
        for row in assets.values()
        if row.get("status") == "OK"
    ]
    confidence = round(sum(scores) / len(scores)) if scores else 0
    if crypto_positive and equity_positive and crypto_sharpe > 0.8 and equity_sharpe > 0.8:
        assessment = "Universal Trend Following"
    elif crypto_positive and crypto_sharpe > 0.8:
        assessment = "Crypto-only Trend Following"
    elif equity_positive and equity_sharpe > 0.8:
        assessment = "Equity-only Trend Following"
    else:
        assessment = "Not Validated"
    return {
        "assessment": assessment,
        "confidence": int(max(0, min(100, confidence))),
        "crypto_average_sharpe": crypto_average.get("average_sharpe"),
        "equity_average_sharpe": equity_average.get("average_sharpe"),
    }


def failure_analysis(asset: str, row: dict[str, Any]) -> dict[str, Any]:
    if row.get("status") != "OK":
        return {
            "asset": asset,
            "failure_reason": row.get("error", "asset validation failed"),
            "rejected_entry_filters": {},
            "loss_exit_reasons": {},
            "trend_following_worked": False,
        }

    rejected = row.get("rejected_entry_reasons") or {}
    exit_losses = row.get("loss_exit_reasons") or {}
    return_pct = _float(row.get("total_return_pct")) or 0.0
    sharpe = _float(row.get("sharpe_ratio")) or 0.0
    score = int(row.get("robustness_score", 0))
    if return_pct > 0 and sharpe > 0.8:
        reason = "trend following worked with positive return and Sharpe above threshold"
        worked = True
    elif return_pct > 0:
        reason = "trend following produced gains, but risk-adjusted performance was weak"
        worked = True
    elif not row.get("total_trades"):
        reason = "no accepted trades"
        worked = False
    elif exit_losses:
        dominant_exit = max(exit_losses, key=lambda key: exit_losses[key])
        reason = f"losses concentrated in {dominant_exit} exits"
        worked = False
    elif rejected:
        dominant_filter = max(rejected, key=lambda key: rejected[key])
        reason = f"entry filters rejected most signals via {dominant_filter}"
        worked = False
    else:
        reason = "negative or weak return profile"
        worked = score >= 60
    return {
        "asset": asset,
        "failure_reason": reason if score < 80 else None,
        "rejected_entry_filters": rejected,
        "loss_exit_reasons": exit_losses,
        "trend_following_worked": worked,
    }


def write_equity_validation_report(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "equity_validation_report.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_cross_asset_validation(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "cross_asset_validation.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _asset_row(asset: str, result: BenchmarkResult, capture_metrics: dict[str, Any]) -> dict[str, Any]:
    metrics = result.metrics
    row = {
        "status": "OK",
        "asset_class": _asset_class(asset),
        "start_date": result.start_date,
        "end_date": result.end_date,
        "total_return_pct": metrics.get("total_return_pct"),
        "cagr": metrics.get("cagr"),
        "sharpe_ratio": metrics.get("sharpe_ratio"),
        "max_drawdown_pct": metrics.get("max_drawdown_pct"),
        "profit_factor": metrics.get("profit_factor"),
        "win_rate": metrics.get("win_rate"),
        "total_trades": metrics.get("total_trades"),
        "profit_capture_ratio": capture_metrics.get("profit_capture_ratio"),
        "rejected_entry_reasons": _nonzero_dict(metrics.get("rejected_entry_reasons", {})),
        "exit_reasons": _nonzero_dict(metrics.get("exit_reasons", {})),
        "loss_exit_reasons": _loss_exit_reasons(result.trades),
    }
    row["robustness_score"] = robustness_score(row)
    return row


def _failed_asset_row(asset: str, error: str) -> dict[str, Any]:
    return {
        "status": "FAILED",
        "asset_class": _asset_class(asset),
        "error": error,
        "robustness_score": 0,
    }


def _equity_data_validation_payload(
    equity_results: list[EquityDataResult],
    equity_failures: dict[str, str],
    requested_equities: tuple[str, ...],
) -> dict[str, Any]:
    by_symbol = {result.symbol: result for result in equity_results}
    rows: dict[str, Any] = {}
    for symbol in requested_equities:
        result = by_symbol.get(symbol)
        if result is None:
            rows[symbol] = {
                "symbol": symbol,
                "rows": 0,
                "provider": None,
                "validation": "failed",
                "error": equity_failures.get(symbol, "equity data failed to load"),
                "timeframes": {},
            }
            continue
        timeframe_rows = {
            timeframe: {
                "rows": validation.rows,
                "provider": validation.provider,
                "validation": validation.validation,
                "start": validation.start,
                "end": validation.end,
                "duplicate_timestamps": validation.duplicate_timestamps,
                "missing_timestamps": validation.missing_timestamps,
                "sorted_ascending": validation.sorted_ascending,
                "minimum_3_years_history": validation.minimum_3_years_history,
                "errors": validation.errors,
            }
            for timeframe, validation in result.validations.items()
        }
        primary = timeframe_rows.get("1h") or next(iter(timeframe_rows.values()), {})
        rows[symbol] = {
            "symbol": symbol,
            "rows": sum(frame["rows"] for frame in timeframe_rows.values()),
            "provider": primary.get("provider"),
            "validation": "passed" if all(frame["validation"] == "passed" for frame in timeframe_rows.values()) else "failed",
            "timeframes": timeframe_rows,
        }
    return rows


def _append_failed_diagnostics(
    diagnostics_path: Path,
    failures: dict[str, str],
    failed_attempts: dict[str, list[ProviderAttempt]],
) -> None:
    if not failures:
        return
    payload = json.loads(diagnostics_path.read_text(encoding="utf-8")) if diagnostics_path.exists() else {"assets": {}}
    assets = payload.setdefault("assets", {})
    for symbol, error in failures.items():
        assets[symbol] = {
            "symbol": symbol,
            "providers": {},
            "attempts": [attempt.__dict__.copy() for attempt in failed_attempts.get(symbol, [])],
            "validation": {},
            "error": error,
        }
    diagnostics_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _agent_aggressive_config(config: BacktestConfig) -> BacktestConfig:
    return replace(
        config,
        profile="aggressive",
        strategy_profile_override=None,
        use_trend_rider=False,
        use_hybrid_trend_rider=False,
        use_trend_holding=False,
        use_regime_gated_trend_holding=False,
        use_portfolio_governor=False,
        stop_type="fixed",
        collect_stop_candidates=False,
    )


def _availability(frames: dict[str, pd.DataFrame], timeframes: tuple[str, ...]) -> dict[str, str]:
    starts = []
    ends = []
    for timeframe in timeframes:
        frame = frames[timeframe]
        if frame.empty:
            raise ValueError(f"{timeframe} history is empty.")
        timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        starts.append(timestamps.min())
        ends.append(timestamps.max())
    return {
        "start": max(starts).isoformat(),
        "end": min(ends).isoformat(),
    }


def _common_end(config_end: str, assets: tuple[str, ...], availability: dict[str, dict[str, str]]) -> pd.Timestamp:
    available_end = min(pd.Timestamp(availability[asset]["end"]) for asset in assets)
    if config_end == "latest":
        return available_end
    return min(available_end, pd.Timestamp(config_end, tz="UTC"))


def _target_start(config_start: str) -> str:
    configured = pd.Timestamp(config_start)
    configured = configured.tz_localize("UTC") if configured.tzinfo is None else configured.tz_convert("UTC")
    preferred = pd.Timestamp(COMMON_HISTORY_TARGET_START, tz="UTC")
    return max(configured, preferred).date().isoformat()


def _normalize_assets(assets: tuple[str, ...], include_optional: bool) -> tuple[str, ...]:
    normalized = [asset.upper() for asset in assets]
    if include_optional:
        normalized.extend(asset for asset in OPTIONAL_ASSETS if asset not in normalized)
    deduped: list[str] = []
    for asset in normalized:
        if asset not in deduped:
            deduped.append(asset)
    return tuple(deduped)


def _normalize_equity_assets(assets: tuple[str, ...], include_optional: bool) -> tuple[str, ...]:
    normalized = [asset.upper() for asset in assets]
    if include_optional:
        normalized.extend(asset for asset in OPTIONAL_ASSETS if asset not in normalized)
    deduped: list[str] = []
    for asset in normalized:
        if asset not in deduped:
            deduped.append(asset)
    return tuple(deduped)


def _ordered_timeframes(primary_timeframe: str, timeframes: tuple[str, ...]) -> tuple[str, ...]:
    ordered = [primary_timeframe]
    for timeframe in timeframes:
        if timeframe not in ordered:
            ordered.append(timeframe)
    return tuple(ordered)


def _rank_assets(rows: dict[str, dict[str, Any]], metric: str, reverse: bool) -> list[str]:
    return sorted(
        rows,
        key=lambda asset: _float(rows[asset].get(metric)) if _float(rows[asset].get(metric)) is not None else float("-inf"),
        reverse=reverse,
    )


def _asset_class(asset: str) -> str:
    if asset in CRYPTO_ASSETS or asset.endswith("USDT"):
        return "crypto"
    return "equity"


def _loss_exit_reasons(trades: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trade in trades:
        if (_float(trade.get("pnl")) or 0.0) >= 0:
            continue
        reason = str(trade.get("exit_reason", "UNKNOWN"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _nonzero_dict(values: dict[str, Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in values.items() if int(value or 0) > 0}


def _mean_metric(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [_float(row.get(metric)) for row in rows]
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 4)


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _emit(progress_callback: ProgressCallback | None, event: dict[str, Any]) -> None:
    if progress_callback is not None:
        progress_callback(event)
