#!/bin/bash
# E1b readout: LOGO/RvR on OUR re-extracted table (run with --after e1-refeat:complete).
set -euo pipefail
export PYTHONPATH="${REPO_ROOT:-$PWD}:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python -m streamdet.scoring.e1_readout \
  --features 'results/e1-refeat/sh_*.csv' \
  --out "$OUT"
