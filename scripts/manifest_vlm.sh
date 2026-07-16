#!/bin/bash
# Build a balanced VLM subsample manifest from the matched-27k cell.
set -euo pipefail
export PYTHONPATH="${PWD}:${PWD}/streamdet:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python streamdet/subsample_manifest.py \
  --subset "${VIDAUDIT_RESULTS:?set VIDAUDIT_RESULTS to the VidAudit results dir}/baseline_clip_subset.csv" \
  --per-source 240 --out "$OUT"
