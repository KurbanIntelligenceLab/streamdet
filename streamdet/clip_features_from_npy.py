"""E1b: rebuild the clip-level 13-d feature table from precomputed MV arrays.

Validates OUR feature path end-to-end against the WACV table: for every clip
in the subset CSV, load npy_path and run vidaudit's feature_vector on the FULL
array (clip-level, exactly what the WACV harness did). If our port is faithful,
the resulting table pushed through e1_readout reproduces the recorded per-fold
AUCs within the bootstrap CI.

Run as an array task (one process per shard):
    python Code/streamdet/clip_features_from_npy.py --manifest <subset.csv> \
        --num-shards N --shard-index $TASK_ID --out $OUT [--target-frames K]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time

import numpy as np

import streamdet  # noqa: F401
from vidaudit.features import FEATURE_NAMES
from vidaudit.features.mv import extract_features

META = ["video_id", "generator", "label", "is_real", "npy_path",
        "n_frames_analyzed", "n_frames_total"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--target-frames", type=int, default=None)
    args = ap.parse_args(argv)

    with open(args.manifest, newline="") as f:
        clips = list(csv.DictReader(f))
    mine = clips[args.shard_index::args.num_shards]
    print(f"clip_features_from_npy v1 target_frames={args.target_frames} "
          f"shard={args.shard_index}/{args.num_shards} clips={len(mine)}", flush=True)

    n_ok = n_fail = 0
    t0 = time.monotonic()
    with streamdet.atomic_out(args.out) as f:
        w = csv.DictWriter(f, fieldnames=META + list(FEATURE_NAMES),
                           extrasaction="ignore")
        w.writeheader()
        for i, c in enumerate(mine):
            try:
                mv = np.load(c["npy_path"])
                feats = extract_features(mv, target_frames=args.target_frames)
                if feats is None:
                    raise ValueError("too few motion frames")
                row = {k: c[k] for k in
                       ("video_id", "generator", "label", "is_real", "npy_path")}
                row.update({k: feats[k] for k in FEATURE_NAMES})
                row["n_frames_analyzed"] = feats["n_frames_analyzed"]
                row["n_frames_total"] = feats["n_frames_total"]
                w.writerow(row)
                n_ok += 1
            except Exception as e:
                n_fail += 1
                print(f"FAIL {c.get('video_id')}: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
            if (i + 1) % 100 == 0 or (i + 1) == len(mine):
                print(f"progress done={i+1}/{len(mine)} ok={n_ok} fail={n_fail} "
                      f"elapsed_s={time.monotonic()-t0:.0f}", flush=True)

    print(f"result shard={args.shard_index} ok={n_ok} fail={n_fail} "
          f"out={args.out}", flush=True)
    if n_ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
