#!/bin/bash
# E7e: motion-bias control on stage-1 (matched + within-bin AUC).
set -euo pipefail
export PYTHONPATH="${PWD}:${PWD}/streamdet:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python streamdet/analyze_motionbias.py \
  --scores 'results/stream-scores/sh_0000.csv' \
  --gop-features 'results/gopfeat-27k/sh_*.csv' \
  --out "$OUT"
