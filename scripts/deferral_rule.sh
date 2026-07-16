#!/bin/bash
# E7: learned deferral rule vs confidence gate (matched budgets, within-test CV).
set -euo pipefail
export PYTHONPATH="${PWD}:${PWD}/streamdet:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python streamdet/analyze_deferral_rule.py \
  --stage1 'results/stream-scores/sh_0000.csv' \
  --stage2 'results/pixel-scores/sh_0000.csv' \
  --gop-features 'results/gopfeat-27k/sh_*.csv' \
  --out "$OUT"
