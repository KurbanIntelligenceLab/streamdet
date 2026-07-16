"""E4 / Table 2 VLM arm: score clips with the small VLM reasoning detector.

Runs Ivy-xDetector (Qwen2.5-VL-3B, AI-Safeguard/Ivy-Fake, the runbook's small
VLM upgrade that fits a consumer GPU) per clip and emits a soft p(generated)
from the verdict-token logits (vidaudit's MLLMDetector.score). Zero-shot: the
VLM is not trained on our split, so its score needs no LOGO readout.

Because a per-chunk VLM sweep of the whole 27k cell is what the cascade exists
to AVOID, we score a stratified subsample per clip (full observed prefix): this
gives the VLM's standalone accuracy (the expensive upper-bound baseline) and the
verdicts for the deferred set (defer-to-VLM cascade). We also record per-clip
wall-clock so Table 2's cost cell is measured, not assumed.

Run as a GPU array task (one process per shard):
    python Code/streamdet/vlm_scores.py --manifest <subsample.csv> \
        --num-shards N --shard-index $TASK_ID --out $OUT [--n-frames 6] [--max-new-tokens 1024]
Output CSV: video_id, generator, label, is_real, score, verdict, latency_s.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time

import streamdet  # noqa: F401
from vidaudit.detectors.base import Clip
from vidaudit.detectors.ivy import IvyXDetector


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--n-frames", type=int, default=6)      # Ivy paper: 1 fps, <=6
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args(argv)

    det = IvyXDetector(n_frames=args.n_frames, max_new_tokens=args.max_new_tokens)
    det._load()   # force the checkpoint download/load up front (fail fast)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"vlm_scores v1 model=Ivy-xDetector(Qwen2.5-VL-3B) device={dev} "
          f"n_frames={args.n_frames} max_new_tokens={args.max_new_tokens} "
          f"shard={args.shard_index}/{args.num_shards}", flush=True)

    with open(args.manifest, newline="") as f:
        clips = list(csv.DictReader(f))
    mine = clips[args.shard_index::args.num_shards]

    n_ok = n_fail = 0
    t_start = time.monotonic()
    with streamdet.atomic_out(args.out) as f:
        w = csv.writer(f)
        w.writerow(["video_id", "generator", "label", "is_real", "score",
                    "verdict", "latency_s"])
        for i, c in enumerate(mine):
            try:
                t0 = time.monotonic()
                clip = Clip(video_id=c["video_id"], path=c["mp4_path"],
                            source=c["generator"], is_real=int(c["is_real"]))
                s = det.score(clip)
                lat = time.monotonic() - t0
                verdict = "fake" if s >= 0.5 else "real"
                w.writerow([c["video_id"], c["generator"], c["label"],
                            c["is_real"], f"{s:.5f}", verdict, round(lat, 2)])
                n_ok += 1
            except Exception as e:
                n_fail += 1
                print(f"FAIL {c.get('video_id')}: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
            if (i + 1) % 20 == 0 or (i + 1) == len(mine):
                el = time.monotonic() - t_start
                print(f"progress done={i+1}/{len(mine)} ok={n_ok} fail={n_fail} "
                      f"elapsed_s={el:.0f} s/clip={el/max(i+1,1):.1f}", flush=True)

    print(f"result shard={args.shard_index} clips_ok={n_ok} clips_fail={n_fail} "
          f"out={args.out}", flush=True)
    if n_ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
