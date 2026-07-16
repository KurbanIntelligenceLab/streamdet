#!/bin/bash
# Fill VLM cells: baseline AUC + codec->VLM cascade on the subsample.
set -euo pipefail
export PYTHONPATH="${PWD}:${PWD}/streamdet:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python streamdet/vlm_analysis.py \
  --vlm 'results/vlm-scores/sh_*.csv' \
  --stage1 'results/stream-scores/sh_0000.csv' \
  --pixel 'results/pixel-scores/sh_0000.csv' \
  --out "$OUT"
