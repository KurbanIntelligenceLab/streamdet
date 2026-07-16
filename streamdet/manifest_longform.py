"""Build the LONG-FORM streaming cell manifest (E5 latency story).

Walks recorded MV trees (per-clip dirs with <clip>.npy + frame_types.txt) and
keeps clips with at least --min-frames frames, so every clip in the cell
streams for >= min_frames/16 chunks. Generated sources come from
GenVidBench/6.7m_mv (long Sora/Kling/OpenSora clips); real sources from
Pair1_mv/vript and Pair2_mv/hd_vg_130m. All clips are later evaluated on a
UNIFORM prefix horizon (truncation), so stream length carries no label signal.

Run (single task):
    python Code/streamdet/manifest_longform.py --data-root "$DATA_DIR" \
        --min-frames 128 --per-source 400 --out $OUT
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

import streamdet

SOURCES = [
    # (generator label, is_real, relative dir under GenVidBench)
    ("sora", 0, "6.7m_mv/OpenAI_Sora"),
    ("keling", 0, "6.7m_mv/keling"),
    ("opensora", 0, "6.7m_mv/OpenSora_13800"),
    ("vript", 1, "Pair1_mv/vript"),
    ("hd_vg_130m", 1, "Pair2_mv/hd_vg_130m"),
]


def clip_dirs(root):
    """Yield per-clip dirs (contain exactly one .npy) under root, recursively."""
    for dirpath, _, files in os.walk(root):
        npys = [f for f in files if f.endswith(".npy")]
        if len(npys) == 1 and "frame_types.txt" in files:
            yield dirpath, os.path.join(dirpath, npys[0])


def n_frames_of(clip_dir):
    with open(os.path.join(clip_dir, "frame_types.txt")) as f:
        return sum(1 for line in f if line.strip())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--min-frames", type=int, default=128)
    ap.add_argument("--per-source", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    print(f"manifest_longform v1 min_frames={args.min_frames} "
          f"per_source={args.per_source}", flush=True)
    rows = []
    for gen, is_real, rel in SOURCES:
        root = os.path.join(args.data_root, "GenVidBench", rel)
        found = kept = 0
        cands = []
        for cdir, npy in sorted(clip_dirs(root)):
            found += 1
            try:
                nf = n_frames_of(cdir)
            except OSError:
                continue
            if nf >= args.min_frames:
                vid = os.path.splitext(os.path.basename(npy))[0]
                cands.append({"video_id": f"{gen}__{vid}", "generator": gen,
                              "label": "real" if is_real else "generated",
                              "is_real": is_real, "mp4_path": "",
                              "npy_path": npy, "n_frames": nf})
        # deterministic subsample: spread evenly over the sorted candidates
        if len(cands) > args.per_source:
            idx = np.linspace(0, len(cands) - 1, args.per_source).round().astype(int)
            cands = [cands[i] for i in sorted(set(idx))]
        kept = len(cands)
        rows.extend(cands)
        print(f"source={gen} found={found} kept={kept}", flush=True)

    with streamdet.atomic_out(args.out) as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "generator", "label",
                                          "is_real", "mp4_path", "npy_path",
                                          "n_frames"])
        w.writeheader()
        w.writerows(rows)
    n_real = sum(r["is_real"] for r in rows)
    print(f"result clips={len(rows)} real={n_real} gen={len(rows)-n_real} "
          f"out={args.out}", flush=True)
    if not rows:
        sys.exit(1)


if __name__ == "__main__":
    main()
