import unittest

import pandas as pd

from backtesting.backtest_engine import BacktestResult
from backtesting.broker_cost_validation import (
    BrokerCostProfile,
    broker_cost_profile_row,
    broker_cost_rankings_payload,
    broker_cost_recommendation,
    broker_cost_validation_payload,
)


class BrokerCostValidationTests(unittest.TestCase):
    def test_profile_row_calculates_cost_drag_and_live_readiness(self) -> None:
        result = fake_result(
            final_equity=11200.0,
            metrics={
                "total_return_pct": 12.0,
                "cagr": 10.0,
                "sharpe_ratio": 0.9,
                "max_drawdown_pct": 10.0,
                "profit_factor": 1.4,
                "win_rate": 45.0,
                "total_trades": 2,
            },
            trades=[
                trade(gross=100.0, fees=10.0, slippage=5.0, net=85.0),
                trade(gross=200.0, fees=20.0, slippage=10.0, net=170.0),
            ],
        )

        row = broker_cost_profile_row(result, BrokerCostProfile("test", 0.001, 0.0005))

        self.assertEqual(row["total_gross_pnl_before_costs"], 300.0)
        self.assertEqual(row["total_fees"], 30.0)
        self.assertEqual(row["total_slippage_cost"], 15.0)
        self.assertEqual(row["total_net_pnl"], 255.0)
        self.assertEqual(row["average_net_pnl_per_trade"], 127.5)
        self.assertEqual(row["cost_per_trade"], 22.5)
        self.assertEqual(row["cost_drag_pct"], 15.0)
        self.assertTrue(row["live_readiness"]["live_ready"])

    def test_rankings_use_sharpe_return_drawdown_profit_factor_and_cost_drag(self) -> None:
        rows = {
            "high_cost": {
                "sharpe_ratio": 0.8,
                "total_return_pct": 20.0,
                "max_drawdown_pct": 12.0,
                "profit_factor": 1.3,
                "cost_drag_pct": 55.0,
            },
            "best": {
                "sharpe_ratio": 1.0,
                "total_return_pct": 18.0,
                "max_drawdown_pct": 10.0,
                "profit_factor": 1.5,
                "cost_drag_pct": 20.0,
            },
            "low_drag": {
                "sharpe_ratio": 0.7,
                "total_return_pct": 12.0,
                "max_drawdown_pct": 8.0,
                "profit_factor": 1.2,
                "cost_drag_pct": 5.0,
            },
        }

        rankings = broker_cost_rankings_payload(rows)

        self.assertEqual(rankings["overall"][0], "best")
        self.assertEqual(rankings["by_cost_drag"][0], "low_drag")
        self.assertEqual(rankings["by_max_drawdown"][0], "low_drag")

    def test_recommendation_identifies_live_profiles_and_destructive_profile(self) -> None:
        rows = {
            "coinbase_high_volume": {
                "all_in_cost_per_side": 0.0013,
                "total_return_pct": 20.0,
                "sharpe_ratio": 0.9,
                "profit_factor": 1.4,
                "max_drawdown_pct": 10.0,
                "cost_drag_pct": 25.0,
                "live_readiness": {"live_ready": True},
            },
            "robinhood_harsh_spread": {
                "all_in_cost_per_side": 0.0095,
                "total_return_pct": -5.0,
                "sharpe_ratio": -0.1,
                "profit_factor": 0.8,
                "max_drawdown_pct": 20.0,
                "cost_drag_pct": 120.0,
                "live_readiness": {"live_ready": False},
            },
        }

        recommendation = broker_cost_recommendation(rows)

        self.assertEqual(recommendation["live_tradable_profiles"], ["coinbase_high_volume"])
        self.assertEqual(recommendation["profile_that_destroys_strategy"], "robinhood_harsh_spread")
        self.assertEqual(recommendation["maximum_tested_all_in_cost_per_side"], 0.0013)
        self.assertTrue(recommendation["do_not_proceed_to_live_trading"])

    def test_validation_payload_contains_profiles_and_rankings(self) -> None:
        profiles = (
            BrokerCostProfile("profile_a", 0.0, 0.0),
            BrokerCostProfile("profile_b", 0.001, 0.001),
        )
        results = {
            "profile_a": fake_result(final_equity=11000.0, trades=[trade(gross=100.0, fees=0.0, slippage=0.0, net=100.0)]),
            "profile_b": fake_result(final_equity=10500.0, trades=[trade(gross=100.0, fees=20.0, slippage=10.0, net=70.0)]),
        }

        payload = broker_cost_validation_payload(results, profiles)

        self.assertEqual(payload["phase"], "1.18")
        self.assertEqual(payload["rankings"]["by_net_return"][0], "profile_a")
        self.assertIn("recommendation", payload)


def fake_result(
    final_equity: float,
    trades: list[dict],
    metrics: dict | None = None,
) -> BacktestResult:
    metric_values = {
        "total_return_pct": round((final_equity / 10000.0 - 1) * 100, 2),
        "cagr": 5.0,
        "sharpe_ratio": 0.8,
        "max_drawdown_pct": 10.0,
        "profit_factor": 1.3,
        "win_rate": 50.0,
        "total_trades": len(trades),
    }
    metric_values.update(metrics or {})
    return BacktestResult(
        symbol="BTCUSDT",
        profile="aggressive",
        strategy_profile={},
        start_date="2024-01-01",
        end_date="2024-02-01",
        initial_capital=10000.0,
        final_equity=final_equity,
        metrics=metric_values,
        decisions=[],
        trades=trades,
        equity_curve=pd.DataFrame(
            {
                "timestamp": ["2024-01-01T00:00:00+00:00", "2024-02-01T00:00:00+00:00"],
                "current_equity": [10000.0, final_equity],
            }
        ),
    )


def trade(gross: float, fees: float, slippage: float, net: float) -> dict:
    return {
        "gross_pnl_before_fees_and_slippage": gross,
        "gross_pnl_after_slippage_before_fees": gross - slippage,
        "total_fee": fees,
        "total_slippage_cost": slippage,
        "net_pnl": net,
        "pnl": net,
    }


if __name__ == "__main__":
    unittest.main()
