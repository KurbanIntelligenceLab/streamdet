#!/bin/bash
# E6: streaming scores + anytime analysis on AIGVDBench (cross-dataset replication).
set -euo pipefail
export PYTHONPATH="${PWD}:${PWD}/streamdet:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python streamdet/streaming_scores.py \
  --manifest 'results/manifest-aigvd/sh_0000.csv' \
  --gop-features 'results/gopfeat-aigvd/sh_*.csv' \
  --out "$OUT"
python streamdet/analyze_streaming.py \
  --scores "$OUT" --max-prefix 64 \
  --out-prefix "$(dirname "$OUT")/aigvd"
