#!/bin/bash
# E7b: streaming scores + analysis for chunk size = 8.
set -euo pipefail
export PYTHONPATH="${REPO_ROOT:-$PWD}:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python -m streamdet.scoring.streaming_scores \
  --manifest "${VIDAUDIT_RESULTS:?set VIDAUDIT_RESULTS to the VidAudit results dir}/baseline_clip_subset.csv" \
  --gop-features 'results/gopfeat-27k-gop8/sh_*.csv' \
  --out "$OUT"
python -m streamdet.analysis.streaming --scores "$OUT" \
  --out-prefix "$(dirname "$OUT")/gop8"
