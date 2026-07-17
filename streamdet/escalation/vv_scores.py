"""RF1 baseline: VideoVeritas (ICML 2026, Qwen3-VL-8B, Apache-2.0) per-clip
verdicts on the E4 subsample.

A second CURRENT published reasoning detector (beyond Ivy-xDetector) placed as
a full-prefix point on the compute-accuracy plane. Zero-shot; same subsample,
same output schema as vlm_scores.py, so the E4 analysis applies unchanged.

Run as a slurmboard GPU array task:
    python Code/streamdet/vv_scores.py --manifest <subsample.csv> \
        --num-shards N --shard-index $TASK_ID --out $OUT [--n-frames 16]
Output CSV: video_id, generator, label, is_real, score, verdict, latency_s.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time

import streamdet  # noqa: F401
from vidaudit.detectors.base import Clip
from vidaudit.detectors.registry import get as get_detector
import vidaudit.detectors  # noqa: F401  (registers all wrappers)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--n-frames", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args(argv)

    det = get_detector("videoveritas")
    det.n_frames = args.n_frames
    det.max_new_tokens = args.max_new_tokens
    det._load()  # force the ModelScope download/load up front (fail fast)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"vv_scores v1 model=VideoVeritas(Qwen3-VL-8B) device={dev} "
          f"n_frames={args.n_frames} max_new_tokens={args.max_new_tokens} "
          f"shard={args.shard_index}/{args.num_shards}", flush=True)

    with open(args.manifest, newline="") as f:
        clips = list(csv.DictReader(f))
    mine = clips[args.shard_index::args.num_shards]

    n_ok = n_fail = 0
    t0 = time.monotonic()
    with streamdet.atomic_out(args.out) as f:
        w = csv.writer(f)
        w.writerow(["video_id", "generator", "label", "is_real", "score",
                    "verdict", "latency_s"])
        for i, c in enumerate(mine):
            try:
                clip = Clip(video_id=c["video_id"], path=c["mp4_path"],
                            source=c["generator"], is_real=int(c["is_real"]))
                t = time.monotonic()
                s = float(det.score(clip))
                dt = time.monotonic() - t
                w.writerow([c["video_id"], c["generator"], c["label"],
                            c["is_real"], f"{s:.5f}",
                            "fake" if s >= 0.5 else "real", f"{dt:.2f}"])
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                print(f"FAIL {c['video_id']}: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                n_fail += 1
            if (i + 1) % 10 == 0:
                el = time.monotonic() - t0
                print(f"progress {i+1}/{len(mine)} ok={n_ok} fail={n_fail} "
                      f"({el/(i+1):.1f}s/clip)", flush=True)
    print(f"result shard={args.shard_index} ok={n_ok} fail={n_fail} "
          f"elapsed_s={time.monotonic()-t0:.0f}", flush=True)


if __name__ == "__main__":
    main()
