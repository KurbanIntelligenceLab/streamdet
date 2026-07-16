"""E2 pixel baseline: per-chunk CLIP ViT-B/32 embeddings (the pixel-CNN-per-chunk arm).

Decodes each clip's frames in order, splits them into the same fixed 16-frame
chunks as stage 1, samples --frames-per-chunk frames per chunk, and mean-pools
their CLIP image embeddings into one 512-d vector per (clip, chunk). This is
vidaudit's CLIP appearance baseline (PCA-13 + L2-LR readout downstream) made
streaming: it must DECODE PIXELS and run a GPU forward pass per chunk — the
cost contrast with the codec stage is the point of E2.

Run as a GPU array task (one process per shard):
    python Code/streamdet/pixel_chunk_features.py --manifest <subset.csv> \
        --num-shards N --shard-index $TASK_ID --out $OUT
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

import streamdet  # noqa: F401

_MODEL = "openai/clip-vit-base-patch32"


def decode_chunk_frames(path, chunk_len=16, per_chunk=4, max_chunks=None):
    """Decode a video; return (chunk_idx, [PIL frames]) with per_chunk frames
    sampled evenly inside each fixed window. The final partial window is kept
    if it has >= per_chunk frames."""
    import av
    from PIL import Image
    container = av.open(str(path))
    chunks, buf, idx = [], [], 0
    try:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            buf.append(frame)
            if len(buf) == chunk_len:
                chunks.append((idx, buf)); buf, idx = [], idx + 1
                if max_chunks is not None and idx >= max_chunks:
                    break
        if buf and len(buf) >= per_chunk and (max_chunks is None or idx < max_chunks):
            chunks.append((idx, buf))
        out = []
        for ci, frames in chunks:
            pick = np.linspace(0, len(frames) - 1, per_chunk).round().astype(int)
            imgs = [Image.fromarray(frames[i].to_ndarray(format="rgb24"))
                    for i in sorted(set(pick))]
            out.append((ci, imgs, len(frames)))
    finally:
        container.close()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--chunk-len", type=int, default=16)
    ap.add_argument("--frames-per-chunk", type=int, default=4)
    ap.add_argument("--max-chunks", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args(argv)

    import torch
    from transformers import CLIPModel, CLIPProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(_MODEL).to(device).eval()
    processor = CLIPProcessor.from_pretrained(_MODEL)
    print(f"pixel_chunk_features v1 model={_MODEL} device={device} "
          f"chunk_len={args.chunk_len} fpc={args.frames_per_chunk} "
          f"shard={args.shard_index}/{args.num_shards}", flush=True)

    with open(args.manifest, newline="") as f:
        clips = list(csv.DictReader(f))
    mine = clips[args.shard_index::args.num_shards]

    dim = 512
    meta_cols = ["video_id", "generator", "label", "is_real", "gop_idx",
                 "n_frames_chunk", "decode_s", "feat_s"]
    n_ok = n_fail = n_rows = 0
    t0 = time.monotonic()
    with streamdet.atomic_out(args.out) as f:
        w = csv.writer(f)
        w.writerow(meta_cols + [f"clip_{j}" for j in range(dim)])
        for i, c in enumerate(mine):
            try:
                td = time.monotonic()
                chunks = decode_chunk_frames(c["mp4_path"], args.chunk_len,
                                             args.frames_per_chunk, args.max_chunks)
                decode_s = time.monotonic() - td
                if not chunks:
                    raise ValueError("no complete chunks decoded")
                imgs = [im for _, ims, _ in chunks for im in ims]
                counts = [len(ims) for _, ims, _ in chunks]
                tf = time.monotonic()
                embs = []
                with torch.no_grad():
                    for b in range(0, len(imgs), args.batch_size):
                        inputs = processor(images=imgs[b:b + args.batch_size],
                                           return_tensors="pt").to(device)
                        pooled = model.vision_model(
                            pixel_values=inputs["pixel_values"]).pooler_output
                        embs.append(model.visual_projection(pooled).float().cpu())
                E = torch.cat(embs).numpy()
                feat_s = time.monotonic() - tf
                off = 0
                for (ci, _, nfr), k in zip(chunks, counts):
                    v = E[off:off + k].mean(axis=0); off += k
                    w.writerow([c["video_id"], c["generator"], c["label"],
                                c["is_real"], ci, nfr, round(decode_s, 4),
                                round(feat_s, 4)] + [f"{x:.5g}" for x in v])
                    n_rows += 1
                n_ok += 1
            except Exception as e:
                n_fail += 1
                print(f"FAIL {c.get('video_id')}: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
            if (i + 1) % 50 == 0 or (i + 1) == len(mine):
                print(f"progress done={i+1}/{len(mine)} ok={n_ok} fail={n_fail} "
                      f"rows={n_rows} elapsed_s={time.monotonic()-t0:.0f}", flush=True)

    print(f"result shard={args.shard_index} clips_ok={n_ok} clips_fail={n_fail} "
          f"chunk_rows={n_rows} out={args.out}", flush=True)
    if n_ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
