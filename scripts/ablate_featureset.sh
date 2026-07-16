#!/bin/bash
# E7c: feature-set ablation on the SAME per-chunk features (no re-extraction).
# Tier-0 (4 global MV stats) vs Tier-1 (9 spectral) vs all-13.
set -euo pipefail
export PYTHONPATH="${PWD}:${PWD}/streamdet:${VIDAUDIT_PATH:-}:${PYTHONPATH:-}"
T0="mean_mv_mag,mv_sparsity,mv_variance,mv_temporal_diff"
T1="spectral_slope_median,spectral_slope_iqr,spectral_flatness_median,spectral_flatness_iqr,acf_decay_rate_median,acf_decay_rate_iqr,acf_first_zero_median,accel_kurtosis_median,accel_skewness_median"
D="$(dirname "$OUT")"
for name in tier0 tier1; do
  case $name in
    tier0) KEEP="$T0";; tier1) KEEP="$T1";;
  esac
  python streamdet/streaming_scores.py \
    --manifest "${VIDAUDIT_RESULTS:?set VIDAUDIT_RESULTS to the VidAudit results dir}/baseline_clip_subset.csv" \
    --gop-features 'results/gopfeat-27k/sh_*.csv' \
    --keep-features "$KEEP" --out "$D/scores_${name}.csv"
  python streamdet/analyze_streaming.py --scores "$D/scores_${name}.csv" \
    --out-prefix "$D/${name}"
done
# marker output so the scheduler sees a completed stage
echo "tier0,tier1 done" > "$OUT"
