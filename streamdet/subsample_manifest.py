"""Stratified subsample of the matched-27k cell for the VLM arm (feasible cost).

A per-chunk VLM sweep of the full cell is exactly what the cascade avoids, so we
score the VLM on a balanced subsample: --per-source clips from each generator and
each real source (deterministic, evenly spread over the sorted video_ids). The
subsample carries mp4_path so the VLM can decode frames; downstream analysis joins
it to the stage-1 codec scores by video_id (same clips), so the codec->VLM cascade
is evaluated on exactly this population.

Run (single task):
    python Code/streamdet/subsample_manifest.py \
        --subset "$DATA_DIR_VIDAUDIT/Paper/Results/baseline_clip_subset.csv" \
        --per-source 240 --out $OUT
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

import streamdet


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--per-source", type=int, default=240)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    with open(args.subset, newline="") as f:
        rows = list(csv.DictReader(f))
    by_src = {}
    for r in rows:
        by_src.setdefault(r["generator"], []).append(r)

    picked = []
    for src, rs in sorted(by_src.items()):
        rs = sorted(rs, key=lambda r: r["video_id"])
        if len(rs) > args.per_source:
            idx = np.linspace(0, len(rs) - 1, args.per_source).round().astype(int)
            rs = [rs[i] for i in sorted(set(idx))]
        picked.extend(rs)
        print(f"source={src} pool={len(by_src[src])} picked={len(rs)} "
              f"is_real={rs[0]['is_real']}", flush=True)

    cols = ["video_id", "generator", "label", "is_real", "mp4_path"]
    with streamdet.atomic_out(args.out) as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(picked)
    n_real = sum(int(r["is_real"]) for r in picked)
    print(f"result clips={len(picked)} real={n_real} gen={len(picked)-n_real} "
          f"out={args.out}", flush=True)


if __name__ == "__main__":
    main()
