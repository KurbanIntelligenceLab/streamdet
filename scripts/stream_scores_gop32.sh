#!/bin/bash
# E7b: streaming scores + analysis for chunk size = 32.
set -euo pipefail
export PYTHONPATH="${PWD}:${PWD}/streamdet:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python streamdet/streaming_scores.py \
  --manifest "${VIDAUDIT_RESULTS:?set VIDAUDIT_RESULTS to the VidAudit results dir}/baseline_clip_subset.csv" \
  --gop-features 'results/gopfeat-27k-gop32/sh_*.csv' \
  --out "$OUT"
python streamdet/analyze_streaming.py --scores "$OUT" \
  --out-prefix "$(dirname "$OUT")/gop32"
