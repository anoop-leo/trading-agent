import unittest
from datetime import date

from monitoring.equity_news import (
    classify_news,
    classify_news_category,
    compute_earnings_alerts,
    compute_upcoming_earnings,
)
from risk.alert_state import evaluate_earnings_proximity_alert


class UpcomingEarningsTests(unittest.TestCase):
    def test_attaches_days_until_and_drops_past(self) -> None:
        rows = [
            {"symbol": "NVDA", "report_date": "2026-07-05"},
            {"symbol": "MSFT", "report_date": "2026-06-20"},  # past
            {"symbol": "TSM", "report_date": "2026-07-01"},
        ]
        upcoming = compute_upcoming_earnings(rows, date(2026, 6, 26))
        self.assertEqual([e["symbol"] for e in upcoming], ["TSM", "NVDA"])  # soonest first, past dropped
        self.assertEqual(upcoming[0]["days_until"], 5)


class EarningsAlertTests(unittest.TestCase):
    def test_fires_within_lead_window_and_dedupes(self) -> None:
        upcoming = [{"symbol": "NVDA", "report_date": "2026-06-28", "days_until": 2}]
        alerts, state = compute_earnings_alerts(upcoming, {}, lead_days=3)
        self.assertEqual(len(alerts), 1)
        self.assertIn("NVDA reports earnings on 2026-06-28", alerts[0])
        self.assertIn("WHEN NOT TO ACT", alerts[0])
        # Re-run with persisted state: no re-fire.
        alerts2, _ = compute_earnings_alerts(upcoming, state, lead_days=3)
        self.assertEqual(alerts2, [])

    def test_no_alert_outside_lead_window(self) -> None:
        upcoming = [{"symbol": "NVDA", "report_date": "2026-07-20", "days_until": 24}]
        alerts, _ = compute_earnings_alerts(upcoming, {}, lead_days=3)
        self.assertEqual(alerts, [])

    def test_alert_helper_ignores_negative_days(self) -> None:
        message, _ = evaluate_earnings_proximity_alert({}, "NVDA", "2026-06-01", -5, 3)
        self.assertIsNone(message)


class ClassifyNewsTests(unittest.TestCase):
    def test_category_detection(self) -> None:
        self.assertEqual(classify_news_category("Analyst raises NVDA price target to $200", None), "analyst_rating")
        self.assertEqual(classify_news_category("Company files Form 4 insider sale", None), "sec_filing")
        self.assertEqual(classify_news_category("Big Co to acquire Small Co", None), "m_and_a")
        self.assertEqual(classify_news_category("Regulator opens antitrust probe", None), "regulatory")
        self.assertEqual(classify_news_category("Firm cuts full-year guidance", None), "guidance")
        self.assertIsNone(classify_news_category("Stock drifts higher in quiet trading", None))

    def test_tier_split_and_sentiment_label(self) -> None:
        feed = [
            {"title": "NVDA upgraded to Buy, price target raised", "source": "A",
             "ticker_sentiment": [{"ticker": "NVDA", "relevance_score": "0.6"}]},
            {"title": "MSFT shares wobble on light volume", "source": "B", "overall_sentiment_label": "Neutral",
             "ticker_sentiment": [{"ticker": "MSFT", "relevance_score": "0.4"}]},
            {"title": "Unrelated token rallies", "source": "C",
             "ticker_sentiment": [{"ticker": "DOGE", "relevance_score": "0.9"}]},  # not in watchlist
        ]
        tier1, tier2 = classify_news(feed, {"NVDA", "MSFT"})
        self.assertEqual(len(tier1["analyst_rating"]), 1)
        self.assertEqual(len(tier2), 1)
        self.assertIn("do not trade on this alone", tier2[0]["label"])

    def test_low_relevance_or_offlist_symbols_excluded(self) -> None:
        feed = [{"title": "Some news", "ticker_sentiment": [{"ticker": "NVDA", "relevance_score": "0.02"}]}]
        tier1, tier2 = classify_news(feed, {"NVDA"})
        self.assertEqual(tier2, [])
        self.assertTrue(all(not items for items in tier1.values()))


if __name__ == "__main__":
    unittest.main()
