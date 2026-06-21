import unittest

import numpy as np
import pandas as pd

from research.exit_optimization_engine import (
    ExitModel,
    exit_models,
    prepare_exit_frame,
    rank_exit_models,
    simulate_exit_model,
)


def raw_frame(closes: list[float]) -> pd.DataFrame:
    close = np.array(closes, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=len(close), freq="h", tz="UTC"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(len(close), 100.0),
        }
    )


class ExitOptimizationEngineTests(unittest.TestCase):
    def test_exit_models_include_required_research_set(self) -> None:
        names = {model.name for model in exit_models()}

        self.assertIn("baseline", names)
        self.assertIn("ema20_trend_rider", names)
        self.assertIn("ema20_ema50_cross", names)
        self.assertIn("atr_trailing_2_0x", names)
        self.assertIn("atr_trailing_4_0x", names)
        self.assertIn("chandelier_exit", names)
        self.assertIn("partial_profit_trend_ride", names)
        self.assertIn("multi_target_25_25_50", names)
        self.assertIn("trend_strength_adx25", names)
        self.assertIn("volatility_adaptive_exit", names)
        self.assertGreaterEqual(
            len([model for model in exit_models() if model.is_hybrid]),
            3,
        )

    def test_rank_exit_models_prioritizes_profit_capture_first(self) -> None:
        rows = {
            "higher_sharpe": {
                "profit_capture_ratio": 0.2,
                "sharpe_ratio": 2.0,
                "total_return_pct": 20.0,
                "max_drawdown_pct": 2.0,
                "profit_factor": 2.0,
                "target_assessment": {"all_targets_met": False},
            },
            "higher_capture": {
                "profit_capture_ratio": 0.3,
                "sharpe_ratio": 0.5,
                "total_return_pct": 10.0,
                "max_drawdown_pct": 5.0,
                "profit_factor": 1.2,
                "target_assessment": {"all_targets_met": False},
            },
        }

        ranking = rank_exit_models(rows)

        self.assertEqual(ranking[0]["model"], "higher_capture")

    def test_partial_profit_model_records_partial_exit_and_breakeven_stop(self) -> None:
        closes = [100.0] * 320
        frame = prepare_exit_frame(raw_frame(closes), start="2020-01-01", end="latest")
        entry_row = frame.iloc[220]
        entry_price = float(entry_row["close"])
        frame.loc[221, "close"] = entry_price + 12.0
        frame.loc[221, "high"] = entry_price + 13.0
        frame.loc[221, "low"] = entry_price + 11.0
        frame.loc[222:, "close"] = entry_price - 1.0
        frame.loc[222:, "high"] = entry_price
        frame.loc[222:, "low"] = entry_price - 2.0
        baseline_trade = {
            "entry_timestamp": pd.Timestamp(entry_row["timestamp"]).isoformat(),
            "entry_price": entry_price,
            "entry_stop_loss": entry_price - 10.0,
            "entry_target_1": entry_price + 10.0,
            "initial_risk": 10.0,
            "entry_decision": "BUY WATCH",
            "entry_alignment": "BULLISH_ALIGNMENT",
        }
        model = ExitModel(
            name="partial_profit_trend_ride",
            model_type="partial_profit_trend_ride",
            description="test",
            parameters={"tp_fraction": 0.5},
        )

        result = simulate_exit_model(
            model,
            [baseline_trade],
            frame,
            initial_capital=10000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            allocation_per_trade=1.0,
            minimum_hold_hours=0.0,
        )

        self.assertEqual(len(result.trades), 1)
        trade = result.trades[0]
        self.assertEqual(trade["exit_reason"], "STOP_LOSS")
        self.assertEqual(len(trade["partial_exits"]), 1)
        self.assertEqual(trade["partial_exits"][0]["position_fraction"], 0.5)
        self.assertGreater(trade["pnl"], 0.0)
        self.assertGreater(result.final_equity, 10000.0)


if __name__ == "__main__":
    unittest.main()
