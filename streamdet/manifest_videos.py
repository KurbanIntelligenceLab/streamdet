"""Build a manifest by walking a video-tree (E6 AIGVDBench, or any mp4 cell).

Walks <root>/<generator>/.../*.mp4 (and common video extensions), assigns each
clip a generator = the immediate top-level subdir under root, and is_real=1
iff that subdir matches --real-name (default 'Real'). Optionally caps clips
per generator for a balanced, affordable cell.

Run (single task):
    python Code/streamdet/manifest_videos.py --root "$DATA_DIR/aigvdbench/extracted" \
        --real-name Real --per-gen 400 --out $OUT
Output columns: video_id, generator, label, is_real, mp4_path, npy_path(empty).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

import streamdet

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def find_videos(gen_root):
    out = []
    for dirpath, _, files in os.walk(gen_root):
        for f in files:
            if os.path.splitext(f)[1].lower() in VIDEO_EXTS:
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--real-name", default="Real")
    ap.add_argument("--per-gen", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    gens = sorted(d for d in os.listdir(args.root)
                  if os.path.isdir(os.path.join(args.root, d)))
    print(f"manifest_videos v1 root={args.root} generators={gens} "
          f"per_gen={args.per_gen}", flush=True)
    rows = []
    for gen in gens:
        vids = find_videos(os.path.join(args.root, gen))
        if len(vids) > args.per_gen:
            idx = np.linspace(0, len(vids) - 1, args.per_gen).round().astype(int)
            vids = [vids[i] for i in sorted(set(idx))]
        is_real = int(gen == args.real_name)
        for p in vids:
            stem = os.path.splitext(os.path.basename(p))[0]
            rows.append({"video_id": f"{gen}__{stem}", "generator": gen,
                         "label": "real" if is_real else "generated",
                         "is_real": is_real, "mp4_path": p, "npy_path": ""})
        print(f"generator={gen} is_real={is_real} kept={len(vids)}", flush=True)

    with streamdet.atomic_out(args.out) as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "generator", "label",
                                          "is_real", "mp4_path", "npy_path"])
        w.writeheader()
        w.writerows(rows)
    n_real = sum(r["is_real"] for r in rows)
    print(f"result clips={len(rows)} real={n_real} gen={len(rows)-n_real} "
          f"out={args.out}", flush=True)
    if not rows:
        sys.exit(1)


if __name__ == "__main__":
    main()
