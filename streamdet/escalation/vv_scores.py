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
    ap.add_argument("--max-side", type=int, default=896,
                    help="cap each frame's long side (px); native-res 720p+ "
                         "sources otherwise OOM the vision tower")
    args = ap.parse_args(argv)

    det = get_detector("videoveritas")
    det.n_frames = args.n_frames
    det.max_new_tokens = args.max_new_tokens
    # Cap frame resolution BEFORE the processor: GenVidBench mixes 256px
    # generators with 720p+ sources (pika/svd/mora/musev and the real
    # vript/hd_vg clips); 16 native-res 720p frames ask ~24 GiB in one alloc.
    # 896 = 32 vision patches of 28 px, aspect preserved.
    orig_frames = det._frames
    def capped_frames(clip):
        ims = orig_frames(clip)
        out = []
        for k, im in enumerate(ims):
            w, h = im.size
            if max(w, h) > args.max_side:
                sc = args.max_side / max(w, h)
                im = im.resize((max(28, round(w * sc)), max(28, round(h * sc))))
                if k == 0:
                    print(f"cap {clip.video_id[:48]}: {w}x{h} -> {im.size}",
                          flush=True)
            out.append(im)
        return out
    det._frames = capped_frames
    det._load()  # force the ModelScope download/load up front (fail fast)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"vv_scores v2 model=VideoVeritas(Qwen3-VL-8B) device={dev} "
          f"n_frames={args.n_frames} max_new_tokens={args.max_new_tokens} "
          f"max_side={args.max_side} shard={args.shard_index}/{args.num_shards}",
          flush=True)

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
                # A huge-resolution clip can demand tens of GiB in one forward
                # pass; the failed alloc also leaves the cache fragmented so
                # every LATER clip OOMs too. Free the cache, retry this clip
                # once at half the frames, and always restore the setting.
                if dev == "cuda" and "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    try:
                        det.n_frames = max(4, args.n_frames // 2)
                        t = time.monotonic()
                        s = float(det.score(clip))
                        dt = time.monotonic() - t
                        w.writerow([c["video_id"], c["generator"], c["label"],
                                    c["is_real"], f"{s:.5f}",
                                    "fake" if s >= 0.5 else "real",
                                    f"{dt:.2f}"])
                        n_ok += 1
                        n_fail -= 1
                        print(f"RETRY-OK {c['video_id']} at n_frames="
                              f"{det.n_frames}", flush=True)
                    except Exception as e2:  # noqa: BLE001
                        print(f"FAIL-RETRY {c['video_id']}: "
                              f"{type(e2).__name__}: {e2}",
                              file=sys.stderr, flush=True)
                        torch.cuda.empty_cache()
                    finally:
                        det.n_frames = args.n_frames
            if (i + 1) % 10 == 0:
                el = time.monotonic() - t0
                print(f"progress {i+1}/{len(mine)} ok={n_ok} fail={n_fail} "
                      f"({el/(i+1):.1f}s/clip)", flush=True)
    print(f"result shard={args.shard_index} ok={n_ok} fail={n_fail} "
          f"elapsed_s={time.monotonic()-t0:.0f}", flush=True)
    if n_fail > n_ok:
        # A majority-failing shard is a systemic problem (bad env, OOM class,
        # unreadable source). Refuse to look "done" with a hollow shard.
        print(f"FATAL: majority of clips failed ({n_fail}/{n_ok+n_fail}); "
              "failing the task so slurmboard re-runs it", file=sys.stderr,
              flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
