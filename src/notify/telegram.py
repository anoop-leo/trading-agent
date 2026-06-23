"""Telegram notifier. Secrets come from the environment only -- never hardcode
TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID, never write them to disk.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


TELEGRAM_API_BASE_URL = "https://api.telegram.org"

JsonOpener = Callable[..., Any]


def telegram_credentials(environ: Mapping[str, str] | None = None) -> tuple[str | None, str | None]:
    environ = environ if environ is not None else os.environ
    return environ.get("TELEGRAM_BOT_TOKEN"), environ.get("TELEGRAM_CHAT_ID")


def send_telegram_message(
    text: str,
    bot_token: str | None = None,
    chat_id: str | None = None,
    environ: Mapping[str, str] | None = None,
    opener: JsonOpener = urlopen,
    timeout_seconds: float = 10.0,
    base_url: str = TELEGRAM_API_BASE_URL,
) -> bool:
    """Send a message. Returns False (does not raise) if credentials are
    missing or the send fails -- callers should log and continue, never crash
    a scheduled job over a notification failure."""

    env_token, env_chat_id = telegram_credentials(environ)
    bot_token = bot_token or env_token
    chat_id = chat_id or env_chat_id
    if not bot_token or not chat_id:
        return False

    url = f"{base_url}/bot{bot_token}/sendMessage"
    data = urlencode({"chat_id": chat_id, "text": text, "disable_web_page_preview": "true"}).encode("utf-8")
    request = Request(url, data=data, method="POST")
    try:
        with opener(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError):
        return False
    return bool(payload.get("ok"))
