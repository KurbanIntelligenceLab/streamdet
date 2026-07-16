"""Survey the streaming structure of a clip cell (cheap, read-only).

For every clip in a subset CSV (with npy_path columns pointing at extracted MV
arrays, each stored as <clip>/<clip>.npy beside frame_types.txt), read the npy
HEADER (shape only, no data) and the frame-type sequence, and emit one row per
clip: frame count, I-frame count, GOP length stats. The aggregate answers "how
many decision points does streaming get per clip, per generator?" — which
determines whether a cell supports the streaming protocol as-is or needs a
re-encode with a shorter GOP.

Run (single task):
    python Code/streamdet/survey_cell.py --subset <subset.csv> --out $OUT
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import zlib  # noqa: F401

import numpy as np

import streamdet


def npy_shape(path):
    """Read an .npy file's shape from its header without loading data."""
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(f)
        else:
            shape, _, _ = np.lib.format.read_array_header_2_0(f)
    return shape


def gop_stats(frame_types: str):
    """I/P/B string (newline-separated) -> (n_frames, n_I, gop_lens)."""
    types = [t.strip() for t in frame_types.strip().splitlines() if t.strip()]
    i_pos = [k for k, t in enumerate(types) if t == "I"]
    gop_lens = []
    for j, start in enumerate(i_pos):
        end = i_pos[j + 1] if j + 1 < len(i_pos) else len(types)
        gop_lens.append(end - start)
    return len(types), len(i_pos), gop_lens


def _try(fn, arg):
    try:
        return fn(arg)
    except (FileNotFoundError, OSError, ValueError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)

    with open(args.subset, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"survey_cell v2 subset={args.subset} clips={len(rows)} "
          f"workers={args.workers}", flush=True)

    def survey_one(r):
        npy = r["npy_path"]
        ft = os.path.join(os.path.dirname(npy), "frame_types.txt")
        shape = npy_shape(npy)
        with open(ft) as g:
            n_frames, n_gops, lens = gop_stats(g.read())
        return [r["video_id"], r["generator"], r["is_real"],
                shape[0], n_frames, n_gops,
                min(lens) if lens else 0, max(lens) if lens else 0,
                int(np.median(lens)) if lens else 0]

    from concurrent.futures import ThreadPoolExecutor
    n_ok = n_miss = 0
    with streamdet.atomic_out(args.out) as f:
        w = csv.writer(f)
        w.writerow(["video_id", "generator", "is_real", "T_npy", "n_frames",
                    "n_gops", "gop_len_min", "gop_len_max", "gop_len_median"])
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, res in enumerate(ex.map(
                    lambda r: _try(survey_one, r), rows, chunksize=64)):
                if res is None:
                    n_miss += 1
                else:
                    w.writerow(res)
                    n_ok += 1
                if (i + 1) % 5000 == 0:
                    print(f"progress done={i+1}/{len(rows)} ok={n_ok} "
                          f"miss={n_miss}", flush=True)

    # concise per-generator aggregate to stdout
    import pandas as pd
    d = pd.read_csv(args.out)
    agg = d.groupby("generator").agg(
        clips=("video_id", "count"), fr_med=("n_frames", "median"),
        gops_med=("n_gops", "median"), gops_p10=("n_gops", lambda x: x.quantile(.1)),
        gops_p90=("n_gops", lambda x: x.quantile(.9)),
        goplen_med=("gop_len_median", "median"))
    print(agg.to_csv(float_format="%.1f"), flush=True)
    print(f"result clips_ok={n_ok} clips_miss={n_miss} out={args.out}", flush=True)
    if n_ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
