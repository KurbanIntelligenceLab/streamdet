#!/bin/bash
# CIs + paired tests for every decision-accuracy point (run after E2-E4 stages).
set -euo pipefail
export PYTHONPATH="${REPO_ROOT:-$PWD}:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python -m streamdet.analysis.significance \
  --stage1 results/stream-scores/sh_0000.csv \
  --stage2 results/pixel-scores/sh_0000.csv \
  --vlm 'results/vlm-scores/sh_*.csv' --knee-width 0.10
