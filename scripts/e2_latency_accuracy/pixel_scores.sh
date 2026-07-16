#!/bin/bash
# E2 pixel arm: LOGO streaming scores (PCA-13 readout) + anytime analysis over CLIP chunks.
set -euo pipefail
export PYTHONPATH="${REPO_ROOT:-$PWD}:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python -m streamdet.scoring.streaming_scores \
  --manifest "${VIDAUDIT_RESULTS:?set VIDAUDIT_RESULTS to the VidAudit results dir}/baseline_clip_subset.csv" \
  --gop-features 'results/pixel-27k/sh_*.csv' \
  --reducer pca --n-components 13 \
  --out "$OUT"
python -m streamdet.analysis.streaming \
  --scores "$OUT" \
  --out-prefix "$(dirname "$OUT")/pixel"
