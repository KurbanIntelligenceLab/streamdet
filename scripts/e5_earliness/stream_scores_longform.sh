#!/bin/bash
# E5: streaming scores + anytime analysis on the long-form cell (rising curve).
set -euo pipefail
export PYTHONPATH="${REPO_ROOT:-$PWD}:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python -m streamdet.scoring.streaming_scores \
  --manifest 'results/manifest-longform/sh_0000.csv' \
  --gop-features 'results/gopfeat-longform/sh_*.csv' \
  --out "$OUT"
python -m streamdet.analysis.streaming \
  --scores "$OUT" --max-prefix 64 \
  --out-prefix "$(dirname "$OUT")/longform"
