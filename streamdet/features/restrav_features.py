"""RF1 baseline: ReStraV (Interno et al., NeurIPS 2025) 21-d perceptual-
straightening features per clip over the matched 27k cell.

A current published AIGV detector placed on the compute-accuracy plane as a
single full-prefix point (the paper's metric admits an offline detector as one
point). Features come from vidaudit's faithful port (frozen DINOv2 ViT-S/14,
torch.hub auto-download, no external weights); the LOGO readout is fit by
restrav_readout.py under the same audited split path as every other stage.

Run as a slurmboard GPU array task:
    python Code/streamdet/restrav_features.py --manifest <cell.csv> \
        --num-shards N --shard-index $TASK_ID --out $OUT
Output CSV: video_id, generator, label, is_real, f0..f20, latency_s.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time

import streamdet  # noqa: F401  (atomic_out + sys.path side effects)
from vidaudit.detectors.base import Clip
from vidaudit.detectors.restrav import ReStraV


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    args = ap.parse_args(argv)

    det = ReStraV()
    det._load()  # force the DINOv2 hub download up front (fail fast)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"restrav_features v1 model=ReStraV(DINOv2-S/14) device={dev} "
          f"n_frames={det.n_frames} shard={args.shard_index}/{args.num_shards}",
          flush=True)

    with open(args.manifest, newline="") as f:
        clips = list(csv.DictReader(f))
    mine = clips[args.shard_index::args.num_shards]

    n_ok = n_fail = 0
    t0 = time.monotonic()
    with streamdet.atomic_out(args.out) as f:
        w = csv.writer(f)
        w.writerow(["video_id", "generator", "label", "is_real"]
                   + [f"f{i}" for i in range(21)] + ["latency_s"])
        for i, c in enumerate(mine):
            try:
                clip = Clip(video_id=c["video_id"], path=c["mp4_path"],
                            source=c["generator"], is_real=int(c["is_real"]))
                t = time.monotonic()
                feats = det.features(clip)
                dt = time.monotonic() - t
                w.writerow([c["video_id"], c["generator"], c["label"],
                            c["is_real"]] + [f"{v:.6g}" for v in feats]
                           + [f"{dt:.2f}"])
                n_ok += 1
            except Exception as e:  # noqa: BLE001 — skip undecodable, count it
                print(f"FAIL {c['video_id']}: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                n_fail += 1
            if n_fail >= 25 and n_ok == 0:
                print("FATAL: first 25 clips all failed; aborting shard",
                      file=sys.stderr, flush=True)
                sys.exit(3)
            if (i + 1) % 100 == 0:
                el = time.monotonic() - t0
                print(f"progress {i+1}/{len(mine)} ok={n_ok} fail={n_fail} "
                      f"({el/(i+1):.2f}s/clip)", flush=True)
    print(f"result shard={args.shard_index} ok={n_ok} fail={n_fail} "
          f"elapsed_s={time.monotonic()-t0:.0f}", flush=True)
    if mine and n_ok == 0:
        # a task that produced NOTHING must fail loudly, not exit 0 with a
        # header-only shard: the canary health window cannot see inside files
        sys.exit(3)


if __name__ == "__main__":
    main()
