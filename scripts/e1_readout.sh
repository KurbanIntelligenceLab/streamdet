#!/bin/bash
# E1a: reproduce the WACV LOGO/RvR numbers from the recorded feature table.
set -euo pipefail
export PYTHONPATH="${PWD}:${PWD}/streamdet:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python streamdet/e1_readout.py \
  --features "${VIDAUDIT_RESULTS:?set VIDAUDIT_RESULTS to the VidAudit results dir}/phase2a_combined_features.csv" \
  --subset "${VIDAUDIT_RESULTS:?set VIDAUDIT_RESULTS to the VidAudit results dir}/baseline_clip_subset.csv" \
  --out "$OUT"
