#!/bin/bash
# E5: per-chunk codec features over the long-form cell (npy mode; 6.7m_mv + reals).
set -euo pipefail
export PYTHONPATH="${REPO_ROOT:-$PWD}:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python -m streamdet.features.gop_features \
  --manifest 'results/manifest-longform/sh_0000.csv' \
  --num-shards "${NUM_SHARDS:-16}" --shard-index "$TASK_ID" \
  --max-gops 64 --out "$OUT"
