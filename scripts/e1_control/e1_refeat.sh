#!/bin/bash
# E1b: re-extract clip-level 13-d features from the recorded MV arrays with OUR port.
set -euo pipefail
export PYTHONPATH="${REPO_ROOT:-$PWD}:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python -m streamdet.features.clip_features_from_npy \
  --manifest "${VIDAUDIT_RESULTS:?set VIDAUDIT_RESULTS to the VidAudit results dir}/baseline_clip_subset.csv" \
  --num-shards "${NUM_SHARDS:-54}" --shard-index "$TASK_ID" \
  --target-frames 12 \
  --out "$OUT"
