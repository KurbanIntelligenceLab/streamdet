#!/bin/bash
# Stage-1 streaming scores + anytime analysis (run with --after gopfeat-27k:complete).
set -euo pipefail
export PYTHONPATH="${PWD}:${PWD}/streamdet:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python streamdet/streaming_scores.py \
  --manifest "${VIDAUDIT_RESULTS:?set VIDAUDIT_RESULTS to the VidAudit results dir}/baseline_clip_subset.csv" \
  --gop-features 'results/gopfeat-27k/sh_*.csv' \
  --out "$OUT"
python streamdet/analyze_streaming.py \
  --scores "$OUT" \
  --out-prefix "$(dirname "$OUT")/stage1"
