#!/bin/bash
# Baseline: LOGO readout over ReStraV features (audited split path, seed 42).
set -euo pipefail
export PYTHONPATH="${REPO_ROOT:-$PWD}:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python -m streamdet.scoring.restrav_readout \
  --features 'results/restrav-feat/sh_*.csv' \
  --out "$OUT"
