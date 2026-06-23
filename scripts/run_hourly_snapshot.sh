#!/bin/bash
# Hourly crypto snapshot: regenerates portfolio value from live prices,
# appends equity history, checks event-alert conditions. Read-only.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi
mkdir -p logs
PYTHONPATH=src venv_trading/bin/python -m monitoring.hourly_snapshot >> logs/hourly_snapshot.log 2>&1
