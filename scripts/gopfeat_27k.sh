#!/bin/bash
# Stage-1 streaming features: per-16-frame-chunk 13-d MV features, 27k cell.
set -euo pipefail
export PYTHONPATH="${PWD}:${PWD}/streamdet:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python streamdet/gop_features.py \
  --manifest "${VIDAUDIT_RESULTS:?set VIDAUDIT_RESULTS to the VidAudit results dir}/baseline_clip_subset.csv" \
  --num-shards "${NUM_SHARDS:-54}" --shard-index "$TASK_ID" \
  --out "$OUT"
