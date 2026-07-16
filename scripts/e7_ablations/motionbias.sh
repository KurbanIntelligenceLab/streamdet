#!/bin/bash
# E7e: motion-bias control on stage-1 (matched + within-bin AUC).
set -euo pipefail
export PYTHONPATH="${REPO_ROOT:-$PWD}:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python -m streamdet.analysis.motionbias \
  --scores 'results/stream-scores/sh_0000.csv' \
  --gop-features 'results/gopfeat-27k/sh_*.csv' \
  --out "$OUT"
