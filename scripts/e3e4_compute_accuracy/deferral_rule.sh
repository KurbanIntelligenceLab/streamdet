#!/bin/bash
# E7: learned deferral rule vs confidence gate (matched budgets, within-test CV).
set -euo pipefail
export PYTHONPATH="${REPO_ROOT:-$PWD}:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python -m streamdet.analysis.deferral_rule \
  --stage1 'results/stream-scores/sh_0000.csv' \
  --stage2 'results/pixel-scores/sh_0000.csv' \
  --gop-features 'results/gopfeat-27k/sh_*.csv' \
  --out "$OUT"
