import json
import unittest

from notify.telegram import send_telegram_message, telegram_credentials


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class TelegramCredentialsTests(unittest.TestCase):
    def test_reads_from_provided_environ_mapping(self) -> None:
        token, chat_id = telegram_credentials({"TELEGRAM_BOT_TOKEN": "abc", "TELEGRAM_CHAT_ID": "123"})
        self.assertEqual(token, "abc")
        self.assertEqual(chat_id, "123")

    def test_missing_credentials_return_none(self) -> None:
        token, chat_id = telegram_credentials({})
        self.assertIsNone(token)
        self.assertIsNone(chat_id)


class SendTelegramMessageTests(unittest.TestCase):
    def test_missing_bot_token_returns_false_without_network_call(self) -> None:
        calls = []

        def opener(request: object, timeout: float = 0) -> FakeResponse:
            calls.append(request)
            return FakeResponse({"ok": True})

        result = send_telegram_message(
            "hello", environ={"TELEGRAM_CHAT_ID": "123"}, opener=opener,
        )
        self.assertFalse(result)
        self.assertEqual(calls, [])

    def test_missing_chat_id_returns_false_without_network_call(self) -> None:
        calls = []

        def opener(request: object, timeout: float = 0) -> FakeResponse:
            calls.append(request)
            return FakeResponse({"ok": True})

        result = send_telegram_message(
            "hello", environ={"TELEGRAM_BOT_TOKEN": "abc"}, opener=opener,
        )
        self.assertFalse(result)
        self.assertEqual(calls, [])

    def test_successful_send_returns_true_and_hits_correct_url(self) -> None:
        captured = {}

        def opener(request: object, timeout: float = 0) -> FakeResponse:
            captured["url"] = request.full_url
            captured["data"] = request.data
            return FakeResponse({"ok": True, "result": {}})

        result = send_telegram_message(
            "hello world",
            environ={"TELEGRAM_BOT_TOKEN": "abc123", "TELEGRAM_CHAT_ID": "999"},
            opener=opener,
        )
        self.assertTrue(result)
        self.assertEqual(captured["url"], "https://api.telegram.org/bot abc123/sendMessage".replace(" ", ""))
        self.assertIn(b"chat_id=999", captured["data"])
        self.assertIn(b"hello", captured["data"])

    def test_explicit_args_override_environ(self) -> None:
        captured = {}

        def opener(request: object, timeout: float = 0) -> FakeResponse:
            captured["url"] = request.full_url
            return FakeResponse({"ok": True})

        send_telegram_message(
            "hi", bot_token="explicit-token", chat_id="explicit-chat",
            environ={"TELEGRAM_BOT_TOKEN": "env-token", "TELEGRAM_CHAT_ID": "env-chat"},
            opener=opener,
        )
        self.assertIn("explicit-token", captured["url"])

    def test_network_failure_returns_false(self) -> None:
        from urllib.error import URLError

        def opener(request: object, timeout: float = 0) -> FakeResponse:
            raise URLError("down")

        result = send_telegram_message(
            "hello", environ={"TELEGRAM_BOT_TOKEN": "abc", "TELEGRAM_CHAT_ID": "123"}, opener=opener,
        )
        self.assertFalse(result)

    def test_telegram_ok_false_returns_false(self) -> None:
        def opener(request: object, timeout: float = 0) -> FakeResponse:
            return FakeResponse({"ok": False, "description": "bad request"})

        result = send_telegram_message(
            "hello", environ={"TELEGRAM_BOT_TOKEN": "abc", "TELEGRAM_CHAT_ID": "123"}, opener=opener,
        )
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
