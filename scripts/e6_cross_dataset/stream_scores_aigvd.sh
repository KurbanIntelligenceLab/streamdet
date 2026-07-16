#!/bin/bash
# E6: streaming scores + anytime analysis on AIGVDBench (cross-dataset replication).
set -euo pipefail
export PYTHONPATH="${REPO_ROOT:-$PWD}:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python -m streamdet.scoring.streaming_scores \
  --manifest 'results/manifest-aigvd/sh_0000.csv' \
  --gop-features 'results/gopfeat-aigvd/sh_*.csv' \
  --out "$OUT"
python -m streamdet.analysis.streaming \
  --scores "$OUT" --max-prefix 64 \
  --out-prefix "$(dirname "$OUT")/aigvd"
