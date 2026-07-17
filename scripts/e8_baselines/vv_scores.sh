#!/bin/bash
# Baseline: VideoVeritas (ICML 2026, Qwen3-VL-8B) zero-shot verdicts, E4 subsample.
set -euo pipefail
export PYTHONPATH="${REPO_ROOT:-$PWD}:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$HOME/.cache/modelscope}"
python -m streamdet.escalation.vv_scores \
  --manifest 'results/manifest-vlm/sh_0000.csv' \
  --num-shards "${NUM_SHARDS:-36}" --shard-index "$TASK_ID" \
  --n-frames 16 --max-new-tokens 1024 --out "$OUT"
