"""Fee and slippage transparency audit for Phase 1.17B backtests."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from backtesting.backtest_engine import BacktestConfig, BacktestResult, run_backtest
from backtesting.backtest_report import write_backtest_report


AUDIT_REPORT_NAME = "fee_slippage_audit_v2.json"
SUPPORTED_AUDIT_STRATEGIES = ("aggressive",)


def run_fee_slippage_audit(
    config: BacktestConfig,
    strategy: str = "aggressive",
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Run a backtest and write a deterministic fee/slippage reconciliation report."""

    if strategy not in SUPPORTED_AUDIT_STRATEGIES:
        supported = ", ".join(SUPPORTED_AUDIT_STRATEGIES)
        raise ValueError(f"Unsupported audit strategy {strategy!r}. Supported strategies: {supported}.")

    audit_config = replace(
        config,
        profile="aggressive",
        stop_type="fixed",
        close_open_position_on_end=True,
    )
    result = run_backtest(audit_config, progress_callback=progress_callback)
    artifacts = write_backtest_report(result, audit_config.output_dir)
    report = build_fee_slippage_audit_report(
        result,
        strategy=strategy,
        fee_rate=audit_config.fee_rate,
        slippage_rate=audit_config.slippage_rate,
        artifacts=artifacts,
    )
    audit_path = audit_config.output_dir / AUDIT_REPORT_NAME
    report["artifacts"]["fee_slippage_audit_v2"] = str(audit_path)
    audit_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "fee_slippage_audit": report,
        "artifacts": {
            **{name: str(path) for name, path in artifacts.items()},
            "fee_slippage_audit_v2": str(audit_path),
        },
    }


def build_fee_slippage_audit_report(
    result: BacktestResult,
    strategy: str = "aggressive",
    fee_rate: float | None = None,
    slippage_rate: float | None = None,
    artifacts: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Build a reconciliation payload from already-generated trade rows."""

    trades = result.trades
    total_gross_before = _sum_trade_field(trades, "gross_pnl_before_fees_and_slippage")
    total_gross_after_slippage = _sum_trade_field(trades, "gross_pnl_after_slippage_before_fees")
    total_fees = _sum_trade_field(trades, "total_fee")
    total_slippage = _sum_trade_field(trades, "total_slippage_cost")
    total_net_pnl = _sum_trade_field(trades, "net_pnl")
    exact_final_equity = _exact_final_equity(result)
    expected_final_equity = result.initial_capital + total_net_pnl
    reconciliation_delta = exact_final_equity - expected_final_equity
    net_pnl_mismatches = [
        {
            "trade_index": index,
            "pnl": float(trade.get("pnl", 0.0)),
            "net_pnl": float(trade.get("net_pnl", 0.0)),
            "delta": float(trade.get("net_pnl", 0.0)) - float(trade.get("pnl", 0.0)),
        }
        for index, trade in enumerate(trades)
        if not math.isclose(float(trade.get("net_pnl", 0.0)), float(trade.get("pnl", 0.0)), abs_tol=1e-6)
    ]
    missing_trade_fields = [
        field
        for field in (
            "signal_entry_price",
            "actual_entry_price",
            "signal_exit_price",
            "actual_exit_price",
            "entry_slippage_cost",
            "exit_slippage_cost",
            "total_slippage_cost",
            "gross_pnl_before_fees_and_slippage",
            "gross_pnl_after_slippage_before_fees",
            "entry_fee",
            "exit_fee",
            "total_fee",
            "net_pnl",
        )
        if any(field not in trade for trade in trades)
    ]
    return {
        "phase": "1.17B",
        "strategy": strategy,
        "symbol": result.symbol,
        "profile": result.profile,
        "start_date": result.start_date,
        "end_date": result.end_date,
        "initial_capital": result.initial_capital,
        "final_equity": round(exact_final_equity, 6),
        "fee_rate": fee_rate,
        "slippage_rate": slippage_rate,
        "fees_modeled": True,
        "slippage_modeled": True,
        "trade_count": len(trades),
        "total_gross_pnl_before_costs": round(total_gross_before, 6),
        "total_gross_pnl_after_slippage": round(total_gross_after_slippage, 6),
        "total_fees": round(total_fees, 6),
        "total_slippage_cost": round(total_slippage, 6),
        "total_net_pnl": round(total_net_pnl, 6),
        "final_equity_reconciliation_delta": round(reconciliation_delta, 8),
        "net_pnl_equals_pnl": len(net_pnl_mismatches) == 0,
        "final_equity_reconciles_to_net_pnl": math.isclose(reconciliation_delta, 0.0, abs_tol=1e-4),
        "net_pnl_mismatches": net_pnl_mismatches,
        "missing_trade_fields": missing_trade_fields,
        "artifacts": {name: str(path) for name, path in (artifacts or {}).items()},
    }


def _sum_trade_field(trades: list[dict[str, Any]], field: str) -> float:
    return sum(float(trade.get(field, 0.0) or 0.0) for trade in trades)


def _exact_final_equity(result: BacktestResult) -> float:
    if result.equity_curve.empty:
        return float(result.final_equity)
    return float(result.equity_curve.iloc[-1]["current_equity"])


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
