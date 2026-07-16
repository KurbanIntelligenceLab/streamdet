#!/bin/bash
# Survey streaming structure (frames/GOPs per clip) of the matched 27k cell.
set -euo pipefail
export PYTHONPATH="${REPO_ROOT:-$PWD}:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python -m streamdet.data.survey_cell \
  --subset "${VIDAUDIT_RESULTS:?set VIDAUDIT_RESULTS to the VidAudit results dir}/baseline_clip_subset_with_t2vz.csv" \
  --out "$OUT"
