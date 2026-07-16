#!/bin/bash
# E3/E4: codec->pixel cascade. Compute accounting + deferral-gain sweep (Prop 3).
set -euo pipefail
export PYTHONPATH="${PWD}:${PWD}/streamdet:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
python streamdet/analyze_cascade.py \
  --stage1 'results/stream-scores/sh_0000.csv' \
  --stage2 'results/pixel-scores/sh_0000.csv' \
  --c1 "${C1_MACS:-1}" --c2 "${C2_MACS:-4300000000}" \
  --out "$OUT"
