#!/bin/bash
# E7b: per-chunk codec features, chunk size = 32 frames.
set -euo pipefail
export PYTHONPATH="${REPO_ROOT:-$PWD}:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python -m streamdet.features.gop_features \
  --manifest "${VIDAUDIT_RESULTS:?set VIDAUDIT_RESULTS to the VidAudit results dir}/baseline_clip_subset.csv" \
  --num-shards "${NUM_SHARDS:-54}" --shard-index "$TASK_ID" \
  --gop-len 32 --out "$OUT"
