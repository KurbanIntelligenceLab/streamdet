"""Extract REAL figure-1 assets for the chosen exemplar clips: GOP-aligned
frames (JPEG) and the codec motion-vector field per GOP (npz), straight from
the bitstream via PyAV's side-data export. Everything in the final figure is
real data; nothing is drawn by hand.

Reads figures/fig1_clips.json: {"clips": [{"video_id":..., "role":...}]}
(mp4 paths come from the cell manifest). Writes, per clip, into the shard dir:
  <role>_f<k>.jpg          frame at the start of GOP k (k = 0,1,2,3)
  <role>_mv.npz            arrays g<k>_x, g<k>_y: mean MV per 16x16 macroblock
and an index CSV to $OUT.

Run as a single slurmboard task:
    python Code/streamdet/fig1_data.py --manifest <cell.csv> --out $OUT
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

import streamdet  # noqa: F401


def extract(path, out_dir, role, n_gops=4, gop_len=16, jpg_h=270):
    import av
    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.codec_context.options = {"flags2": "+export_mvs"}
    W = stream.codec_context.width
    H = stream.codec_context.height
    gw, gh = W // 16, H // 16
    sums = {}
    counts = {}
    frames_saved = 0
    fidx = 0
    mv_out = {}
    for frame in container.decode(stream):
        g = fidx // gop_len
        if g >= n_gops:
            break
        if fidx % gop_len == 0:
            img = frame.to_image()
            scale = jpg_h / img.height
            img = img.resize((int(img.width * scale), jpg_h))
            img.save(out_dir / f"{role}_f{g}.jpg", quality=88)
            frames_saved += 1
        sd = frame.side_data.get("MOTION_VECTORS")
        if sd is not None:
            arr = sd.to_ndarray()
            gx = np.zeros((gh, gw))
            gy = np.zeros((gh, gw))
            gc = np.zeros((gh, gw))
            for mv in arr:
                bx = min(int(mv["dst_x"]) // 16, gw - 1)
                by = min(int(mv["dst_y"]) // 16, gh - 1)
                if bx < 0 or by < 0:
                    continue
                gx[by, bx] += (mv["motion_x"] / max(1, mv["motion_scale"]))
                gy[by, bx] += (mv["motion_y"] / max(1, mv["motion_scale"]))
                gc[by, bx] += 1
            key = f"g{g}"
            if key not in sums:
                sums[key] = [np.zeros((gh, gw)), np.zeros((gh, gw))]
                counts[key] = np.zeros((gh, gw))
            sums[key][0] += gx
            sums[key][1] += gy
            counts[key] += gc
        fidx += 1
    for key, (sx, sy) in sums.items():
        c = np.maximum(counts[key], 1)
        mv_out[f"{key}_x"] = (sx / c).astype(np.float32)
        mv_out[f"{key}_y"] = (sy / c).astype(np.float32)
    np.savez(out_dir / f"{role}_mv.npz", **mv_out)
    return frames_saved, len(sums), (gw, gh)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    spec = json.loads(Path("figures/fig1_clips.json").read_text())
    with open(args.manifest, newline="") as f:
        paths = {r["video_id"]: r["mp4_path"] for r in csv.DictReader(f)}
    out_dir = Path(os.path.dirname(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for c in spec["clips"]:
        vid, role = c["video_id"], c["role"]
        p = paths.get(vid)
        if p is None:
            print(f"MISSING in manifest: {vid}", flush=True)
            continue
        nf, ng, grid = extract(p, out_dir, role)
        print(f"extracted role={role} frames={nf} gops_with_mv={ng} "
              f"grid={grid} vid={vid[:50]}", flush=True)
        rows.append({"role": role, "video_id": vid, "frames": nf,
                     "gops_with_mv": ng})
    with streamdet.atomic_out(args.out) as f:
        w = csv.DictWriter(f, fieldnames=["role", "video_id", "frames",
                                          "gops_with_mv"])
        w.writeheader()
        w.writerows(rows)
    print(f"result clips={len(rows)} out={args.out}", flush=True)


if __name__ == "__main__":
    main()
