"""E2 latency microbenchmark: per-chunk wall-clock for stage-1 vs pixel CLIP.

Measures, on a sample of real mp4 clips and on THIS node's hardware:
  * stage-1 codec path: PyAV decode with export_mvs + MV binning (per chunk),
    13-d feature extraction (target_frames=12), and L2-LR scoring — CPU.
    NOTE the decode is included honestly: our extractor obtains MVs via the
    decoder's side-data export, so the measured cost is an UPPER bound on a
    parse-only implementation.
  * pixel path: per-chunk frame decode + CLIP ViT-B/32 preprocess + forward
    (4 frames/chunk) on GPU if available, else CPU.
Reports per-chunk milliseconds (p50/p90/mean) per system and writes one CSV
row per (clip, system, chunk).

Run (single GPU task):
    python Code/streamdet/bench_latency.py --manifest <subset.csv> \
        --n-clips 200 --out $OUT
"""
from __future__ import annotations

import argparse
import csv
import sys
import time

import numpy as np

import streamdet
from vidaudit.features import FEATURE_NAMES
from vidaudit.features.mv import extract_features

CHUNK = 16


def stage1_chunk_times(path, scorer, mb=16, max_chunks=64):
    """Yield (chunk_idx, decode_mv_s, feat_s, score_s) streaming through the clip."""
    import av
    from av.codec.context import Flags2
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        stream.codec_context.flags2 |= Flags2.export_mvs
        dim_x = dim_y = None
        grids = []
        ci = 0
        t0 = time.monotonic()
        for frame in container.decode(stream):
            if dim_x is None:
                dim_x = (frame.width - 1) // mb + 1
                dim_y = (frame.height - 1) // mb + 1
            grid = np.zeros((dim_y, dim_x, 5), dtype=np.int64)
            try:
                mvs = frame.side_data.get("MOTION_VECTORS")
            except Exception:
                mvs = None
            if mvs is not None:
                for r in mvs.to_ndarray():
                    if r["w"] == mb and r["h"] == mb:
                        x = int(r["src_x"]) // mb
                        y = int(r["src_y"]) // mb
                        if 0 <= x < dim_x and 0 <= y < dim_y and not grid[y, x].any():
                            grid[y, x] = (r["source"], r["dst_x"], r["dst_y"],
                                          r["src_x"], r["src_y"])
            grids.append(grid)
            if len(grids) == CHUNK:
                decode_mv_s = time.monotonic() - t0
                window = np.stack(grids, axis=0)
                t1 = time.monotonic()
                feats = extract_features(window, target_frames=12)
                feat_s = time.monotonic() - t1
                score_s = 0.0
                if feats is not None:
                    v = np.array([feats[k] for k in FEATURE_NAMES])[None, :]
                    t2 = time.monotonic()
                    scorer.predict_proba(np.nan_to_num(v))
                    score_s = time.monotonic() - t2
                yield ci, decode_mv_s, feat_s, score_s
                ci += 1
                grids = []
                t0 = time.monotonic()
                if ci >= max_chunks:
                    break
    finally:
        container.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n-clips", type=int, default=200)
    ap.add_argument("--frames-per-chunk", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import torch
    from transformers import CLIPModel, CLIPProcessor
    from sklearn.linear_model import LogisticRegression
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "none"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    rng = np.random.default_rng(0)
    scorer = LogisticRegression().fit(rng.normal(size=(64, 13)),
                                      rng.integers(0, 2, 64))
    print(f"bench_latency v1 device={device} gpu={gpu} chunk={CHUNK} "
          f"fpc={args.frames_per_chunk}", flush=True)

    with open(args.manifest, newline="") as f:
        clips = [c for c in csv.DictReader(f) if c.get("mp4_path")]
    idx = np.linspace(0, len(clips) - 1, min(args.n_clips, len(clips))).round()
    sample = [clips[int(i)] for i in sorted(set(idx))]

    from pixel_chunk_features import decode_chunk_frames
    rows = []
    n_fail = 0
    for i, c in enumerate(sample):
        try:
            for ci, dmv, fs, ss in stage1_chunk_times(c["mp4_path"], scorer):
                rows.append([c["video_id"], "codec", ci,
                             round(1e3 * dmv, 3), round(1e3 * fs, 3),
                             round(1e3 * ss, 3), round(1e3 * (dmv + fs + ss), 3)])
            tpd = time.monotonic()
            chunks = decode_chunk_frames(c["mp4_path"], CHUNK,
                                         args.frames_per_chunk, 64)
            pix_decode_s = time.monotonic() - tpd
            pd_per_chunk = pix_decode_s / max(len(chunks), 1)
            for ci, imgs, _ in chunks:
                t0 = time.monotonic()
                with torch.no_grad():
                    inputs = processor(images=imgs, return_tensors="pt").to(device)
                    pooled = model.vision_model(
                        pixel_values=inputs["pixel_values"]).pooler_output
                    model.visual_projection(pooled)
                    if device == "cuda":
                        torch.cuda.synchronize()
                el = time.monotonic() - t0
                # decode_mv_ms column reused for the pixel path's frame-decode
                # share, so both systems' total_ms now include decode (fair).
                rows.append([c["video_id"], "clip_pixel", ci,
                             round(1e3 * pd_per_chunk, 3), round(1e3 * el, 3),
                             0.0, round(1e3 * (pd_per_chunk + el), 3)])
        except Exception as e:
            n_fail += 1
            print(f"FAIL {c['video_id']}: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
        if (i + 1) % 25 == 0:
            print(f"progress done={i+1}/{len(sample)} rows={len(rows)} "
                  f"fail={n_fail}", flush=True)

    with streamdet.atomic_out(args.out) as f:
        w = csv.writer(f)
        w.writerow(["video_id", "system", "chunk_idx", "decode_mv_ms",
                    "feat_ms", "score_ms", "total_ms"])
        w.writerows(rows)

    import pandas as pd
    d = pd.DataFrame(rows, columns=["video_id", "system", "chunk_idx",
                                    "decode_mv_ms", "feat_ms", "score_ms",
                                    "total_ms"])
    # MACs per chunk (the hardware-independent compute axis, the honest headline).
    # CLIP ViT-B/32 image encoder ~= 4.4 GMACs/224px image (Radford et al.);
    # codec stage-1 parse+score is O(GOP*grid) spectral ops on a 12x16x16 field,
    # ~1e5 MACs -- 4-5 orders of magnitude less, and NO GPU.
    clip_macs = 4.4e9 * args.frames_per_chunk
    codec_macs = 1e5
    macs = {"clip_pixel": clip_macs, "codec": codec_macs}
    for sysname, g in d.groupby("system"):
        t = g["total_ms"]
        dec = g["decode_mv_ms"]; feat = g["feat_ms"]
        print(f"result system={sysname} device={device if sysname!='codec' else 'cpu'} "
              f"chunks={len(g)} total_p50_ms={t.median():.2f} total_mean_ms={t.mean():.2f} "
              f"decode_p50_ms={dec.median():.2f} compute_p50_ms={feat.median():.2f} "
              f"macs_per_chunk={macs.get(sysname,0):.3g}", flush=True)
    print(f"result gpu={gpu} out={args.out}", flush=True)
    if not rows:
        sys.exit(1)


if __name__ == "__main__":
    main()
