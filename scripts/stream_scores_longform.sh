#!/bin/bash
# E5: streaming scores + anytime analysis on the long-form cell (rising curve).
set -euo pipefail
export PYTHONPATH="${PWD}:${PWD}/streamdet:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python streamdet/streaming_scores.py \
  --manifest 'results/manifest-longform/sh_0000.csv' \
  --gop-features 'results/gopfeat-longform/sh_*.csv' \
  --out "$OUT"
python streamdet/analyze_streaming.py \
  --scores "$OUT" --max-prefix 64 \
  --out-prefix "$(dirname "$OUT")/longform"
