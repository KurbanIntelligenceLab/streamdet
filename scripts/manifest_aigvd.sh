#!/bin/bash
# E6: build AIGVDBench manifest (7 gen + Real) from the extracted video tree.
set -euo pipefail
export PYTHONPATH="${PWD}:${PWD}/streamdet:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python streamdet/manifest_videos.py \
  --root "${DATA_DIR:?set DATA_DIR}/aigvdbench/extracted" --real-name Real \
  --per-gen 400 --out "$OUT"
