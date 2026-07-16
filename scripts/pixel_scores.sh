#!/bin/bash
# E2 pixel arm: LOGO streaming scores (PCA-13 readout) + anytime analysis over CLIP chunks.
set -euo pipefail
export PYTHONPATH="${PWD}:${PWD}/streamdet:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python streamdet/streaming_scores.py \
  --manifest "${VIDAUDIT_RESULTS:?set VIDAUDIT_RESULTS to the VidAudit results dir}/baseline_clip_subset.csv" \
  --gop-features 'results/pixel-27k/sh_*.csv' \
  --reducer pca --n-components 13 \
  --out "$OUT"
python streamdet/analyze_streaming.py \
  --scores "$OUT" \
  --out-prefix "$(dirname "$OUT")/pixel"
