#!/bin/bash
# E1b readout: LOGO/RvR on OUR re-extracted table (run with --after e1-refeat:complete).
set -euo pipefail
export PYTHONPATH="${PWD}:${PWD}/streamdet:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python streamdet/e1_readout.py \
  --features 'results/e1-refeat/sh_*.csv' \
  --out "$OUT"
