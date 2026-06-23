#!/bin/bash
# Daily equity scan (rate-limited Alpha Vantage calls) + digest send.
# Read-only.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi
mkdir -p logs
PYTHONPATH=src venv_trading/bin/python -m monitoring.daily_scan >> logs/daily_scan.log 2>&1
PYTHONPATH=src venv_trading/bin/python -m monitoring.daily_digest >> logs/daily_digest.log 2>&1
