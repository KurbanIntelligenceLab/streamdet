#!/bin/bash
# E1a: reproduce the reference toolkit's published LOGO/RvR numbers from its feature table.
set -euo pipefail
export PYTHONPATH="${REPO_ROOT:-$PWD}:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python -m streamdet.scoring.e1_readout \
  --features "${VIDAUDIT_RESULTS:?set VIDAUDIT_RESULTS to the VidAudit results dir}/phase2a_combined_features.csv" \
  --subset "${VIDAUDIT_RESULTS:?set VIDAUDIT_RESULTS to the VidAudit results dir}/baseline_clip_subset.csv" \
  --out "$OUT"
