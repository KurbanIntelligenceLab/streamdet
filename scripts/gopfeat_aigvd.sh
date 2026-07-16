#!/bin/bash
# E6: canonical-re-encode + per-chunk codec features for AIGVDBench (mp4 mode).
set -euo pipefail
export PYTHONPATH="${PWD}:${PWD}/streamdet:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python streamdet/gop_features.py \
  --manifest 'results/manifest-aigvd/sh_0000.csv' \
  --num-shards "${NUM_SHARDS:-32}" --shard-index "$TASK_ID" \
  --max-gops 32 --max-frames 528 --out "$OUT"
