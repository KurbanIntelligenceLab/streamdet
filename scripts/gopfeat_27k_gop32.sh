#!/bin/bash
# E7b: per-chunk codec features, chunk size = 32 frames.
set -euo pipefail
export PYTHONPATH="${PWD}:${PWD}/streamdet:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python streamdet/gop_features.py \
  --manifest "${VIDAUDIT_RESULTS:?set VIDAUDIT_RESULTS to the VidAudit results dir}/baseline_clip_subset.csv" \
  --num-shards "${NUM_SHARDS:-54}" --shard-index "$TASK_ID" \
  --gop-len 32 --out "$OUT"
