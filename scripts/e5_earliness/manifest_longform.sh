#!/bin/bash
# Build the long-form streaming cell manifest (>=128-frame clips, 5 sources).
set -euo pipefail
export PYTHONPATH="${REPO_ROOT:-$PWD}:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python -m streamdet.data.manifest_longform \
  --data-root "${DATA_DIR:?set DATA_DIR}" --min-frames 128 --per-source 400 \
  --out "$OUT"
