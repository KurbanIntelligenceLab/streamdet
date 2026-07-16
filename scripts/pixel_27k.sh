#!/bin/bash
# E2 pixel baseline: per-chunk CLIP ViT-B/32 embeddings over the 27k cell.
set -euo pipefail
export PYTHONPATH="${PWD}:${PWD}/streamdet:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
python streamdet/pixel_chunk_features.py \
  --manifest "${VIDAUDIT_RESULTS:?set VIDAUDIT_RESULTS to the VidAudit results dir}/baseline_clip_subset.csv" \
  --num-shards "${NUM_SHARDS:-16}" --shard-index "$TASK_ID" \
  --out "$OUT"
