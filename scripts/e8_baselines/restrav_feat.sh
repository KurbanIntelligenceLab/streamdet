#!/bin/bash
# Baseline: ReStraV (Interno et al., NeurIPS 2025) 21-d features per clip.
set -euo pipefail
export PYTHONPATH="${REPO_ROOT:-$PWD}:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
python -m streamdet.features.restrav_features \
  --manifest "$VIDAUDIT_RESULTS/baseline_clip_subset.csv" \
  --num-shards "${NUM_SHARDS:-32}" --shard-index "$TASK_ID" \
  --out "$OUT"
