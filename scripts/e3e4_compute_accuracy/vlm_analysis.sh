#!/bin/bash
# Fill VLM cells: baseline AUC + codec->VLM cascade on the subsample.
set -euo pipefail
export PYTHONPATH="${REPO_ROOT:-$PWD}:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python -m streamdet.analysis.vlm \
  --vlm 'results/vlm-scores/sh_*.csv' \
  --stage1 'results/stream-scores/sh_0000.csv' \
  --pixel 'results/pixel-scores/sh_0000.csv' \
  --out "$OUT"
