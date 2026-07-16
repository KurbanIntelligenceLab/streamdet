#!/bin/bash
# Build the long-form streaming cell manifest (>=128-frame clips, 5 sources).
set -euo pipefail
export PYTHONPATH="${PWD}:${PWD}/streamdet:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python streamdet/manifest_longform.py \
  --data-root "${DATA_DIR:?set DATA_DIR}" --min-frames 128 --per-source 400 \
  --out "$OUT"
