#!/bin/bash
# Survey streaming structure (frames/GOPs per clip) of the matched 27k cell.
set -euo pipefail
export PYTHONPATH="${PWD}:${PWD}/streamdet:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python streamdet/survey_cell.py \
  --subset "${VIDAUDIT_RESULTS:?set VIDAUDIT_RESULTS to the VidAudit results dir}/baseline_clip_subset_with_t2vz.csv" \
  --out "$OUT"
