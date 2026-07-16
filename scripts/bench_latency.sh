#!/bin/bash
# E2 latency microbenchmark: per-chunk ms for codec stage-1 vs CLIP pixel path.
set -euo pipefail
export PYTHONPATH="${PWD}:${PWD}/streamdet:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
python streamdet/bench_latency.py \
  --manifest "${VIDAUDIT_RESULTS:?set VIDAUDIT_RESULTS to the VidAudit results dir}/baseline_clip_subset.csv" \
  --n-clips 200 --out "$OUT"
