import json
import unittest

from data.equity_news_provider import EquityNewsProvider

EARNINGS_CSV = (
    "symbol,name,reportDate,fiscalDateEnding,estimate,currency\n"
    "NVDA,NVIDIA Corp,2026-07-15,2026-06-30,1.20,USD\n"
    "MSFT,Microsoft Corp,2026-07-22,2026-06-30,3.10,USD\n"
    "ZZZZ,Other Co,2026-07-01,2026-06-30,0.5,USD\n"
)
NEWS_JSON = {"items": "2", "feed": [
    {"title": "NVDA price target raised", "source": "X", "topics": [],
     "ticker_sentiment": [{"ticker": "NVDA", "relevance_score": "0.5"}]},
]}


class _Resp:
    def __init__(self, text: str) -> None:
        self._b = text.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._b


def _opener(text: str):
    def _o(request, timeout=None):  # noqa: ANN001
        return _Resp(text)
    return _o


class EarningsCalendarTests(unittest.TestCase):
    def test_filters_to_watchlist(self) -> None:
        p = EquityNewsProvider(opener=_opener(EARNINGS_CSV), environ={"ALPHA_VANTAGE_API_KEY": "k"})
        env = p.fetch_earnings_calendar(["NVDA", "MSFT"])
        self.assertTrue(env["available"])
        symbols = {row["symbol"] for row in env["data"]}
        self.assertEqual(symbols, {"NVDA", "MSFT"})
        self.assertNotIn("ZZZZ", symbols)

    def test_missing_key_is_unavailable(self) -> None:
        p = EquityNewsProvider(opener=_opener(EARNINGS_CSV), environ={})
        env = p.fetch_earnings_calendar(["NVDA"])
        self.assertFalse(env["available"])
        self.assertIn("ALPHA_VANTAGE_API_KEY", env["error"])

    def test_rate_limit_json_is_unavailable(self) -> None:
        body = json.dumps({"Information": "rate limit: 25 requests per day"})
        p = EquityNewsProvider(opener=_opener(body), environ={"ALPHA_VANTAGE_API_KEY": "k"})
        env = p.fetch_earnings_calendar(["NVDA"])
        self.assertFalse(env["available"])
        self.assertIn("rate limit", env["error"])


class NewsSentimentTests(unittest.TestCase):
    def test_returns_feed(self) -> None:
        p = EquityNewsProvider(opener=_opener(json.dumps(NEWS_JSON)), environ={"ALPHA_VANTAGE_API_KEY": "k"})
        env = p.fetch_news_sentiment(["NVDA"])
        self.assertTrue(env["available"])
        self.assertEqual(len(env["data"]), 1)

    def test_missing_feed_is_unavailable(self) -> None:
        body = json.dumps({"Information": "rate limited"})
        p = EquityNewsProvider(opener=_opener(body), environ={"ALPHA_VANTAGE_API_KEY": "k"})
        env = p.fetch_news_sentiment(["NVDA"])
        self.assertFalse(env["available"])
        self.assertIn("rate limited", env["error"])


if __name__ == "__main__":
    unittest.main()
