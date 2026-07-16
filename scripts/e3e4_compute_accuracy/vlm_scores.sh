#!/bin/bash
# E4/Table 2 VLM arm: Ivy-xDetector (Qwen2.5-VL-3B) soft p(generated) per clip.
set -euo pipefail
export PYTHONPATH="${REPO_ROOT:-$PWD}:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
python -m streamdet.escalation.vlm_scores \
  --manifest 'results/manifest-vlm/sh_0000.csv' \
  --num-shards "${NUM_SHARDS:-36}" --shard-index "$TASK_ID" \
  --n-frames 6 --max-new-tokens 1024 --out "$OUT"
