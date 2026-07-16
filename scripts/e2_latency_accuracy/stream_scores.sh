#!/bin/bash
# Stage-1 streaming scores + anytime analysis (run with --after gopfeat-27k:complete).
set -euo pipefail
export PYTHONPATH="${REPO_ROOT:-$PWD}:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python -m streamdet.scoring.streaming_scores \
  --manifest "${VIDAUDIT_RESULTS:?set VIDAUDIT_RESULTS to the VidAudit results dir}/baseline_clip_subset.csv" \
  --gop-features 'results/gopfeat-27k/sh_*.csv' \
  --out "$OUT"
python -m streamdet.analysis.streaming \
  --scores "$OUT" \
  --out-prefix "$(dirname "$OUT")/stage1"
