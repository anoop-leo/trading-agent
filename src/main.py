"""Compatibility entrypoint for `python src/main.py ...` commands."""

from trading_agent.main import main


if __name__ == "__main__":
    raise SystemExit(main())
