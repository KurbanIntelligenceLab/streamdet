"""Local end-to-end smoke test for the stage-1 streaming pipeline (no cluster).

Synthesizes tiny H.264 clips with the canonical GOP structure (keyint=16,
scenecut off), runs gop_features shard extraction, then streaming_scores LOGO
scoring, and asserts the outputs are sane. "Generated" clips get temporally
smooth drifting noise; "real" clips get jittery noise — so scores should
separate at least weakly, but the assertion is only structural (the point is
the plumbing, not accuracy).

Run: python Code/streamdet/smoke_local.py  (~30 s, CPU)
"""
from __future__ import annotations

import csv
import os
import sys
import tempfile
import zlib

import numpy as np

import streamdet  # noqa: F401
from streamdet.features import gop_features
from streamdet.scoring import streaming_scores


def write_clip(path, kind, seed, frames=48, size=64, fps=8):
    import av
    rng = np.random.default_rng(seed)
    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = stream.height = size
    stream.pix_fmt = "yuv420p"
    stream.options = {"g": "16", "keyint_min": "16", "sc_threshold": "0",
                      "crf": "23", "profile": "main"}
    base = rng.integers(0, 255, (size, size, 3)).astype(np.uint8)
    for t in range(frames):
        if kind == "gen":  # smooth global drift
            shift = int(2 * t)
            img = np.roll(base, shift % size, axis=1)
        else:              # jittery random shifts
            img = np.roll(base, int(rng.integers(-6, 7)), axis=1)
            img = np.roll(img, int(rng.integers(-6, 7)), axis=0)
        frame = __import__("av").VideoFrame.from_ndarray(img, format="rgb24")
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


def main():
    tmp = tempfile.mkdtemp(prefix="streamdet_smoke_")
    manifest = os.path.join(tmp, "manifest.csv")
    rows = []
    n = 0
    for gen, kind, count in (("vript", "real", 10), ("genA", "gen", 6),
                             ("genB", "gen", 6)):
        for i in range(count):
            p = os.path.join(tmp, f"{gen}_{i}.mp4")
            seed = zlib.crc32(f"{gen}_{i}".encode()) % 2**31  # deterministic
            write_clip(p, kind, seed=seed)
            rows.append({"video_id": f"{gen}_{i}", "generator": gen,
                         "label": "real" if kind == "real" else "generated",
                         "is_real": 1 if kind == "real" else 0, "mp4_path": p})
            n += 1
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[smoke] wrote {n} clips -> {manifest}")

    feats = os.path.join(tmp, "gopfeat.csv")
    gop_features.main(["--manifest", manifest, "--out", feats,
                       "--num-shards", "1", "--shard-index", "0"])
    import pandas as pd
    g = pd.read_csv(feats)
    # A clip can legitimately produce 0 scorable chunks (all-degenerate/low-motion
    # windows under target_frames=12); the real pipeline handles those via the
    # abstain protocol (streaming_scores --manifest). So allow a small fraction to
    # abstain rather than demanding every clip appear.
    present = set(g["video_id"])
    frac_present = len(present) / len(rows)
    assert frac_present >= 0.8, f"too many clips abstained: {frac_present:.2f}"
    per_clip = g.groupby("video_id")["gop_idx"].nunique()
    assert (per_clip >= 1).all(), f"a listed clip has 0 chunks:\n{per_clip}"
    assert (per_clip >= 2).mean() > 0.5, f"most clips should score >=2 chunks:\n{per_clip}"
    assert g["iframe_ok"].mean() > 0.9, "GOP-16 I-frame structure not detected"
    assert g[list(gop_features.FEATURE_NAMES)].notna().all().all(), "NaN features"
    print(f"[smoke] gop_features OK: {len(g)} rows, "
          f"{per_clip.min()}-{per_clip.max()} GOPs/clip, "
          f"iframe_ok={g['iframe_ok'].mean():.2f}")

    scores = os.path.join(tmp, "scores.csv")
    streaming_scores.main(["--gop-features", feats, "--out", scores])
    s = pd.read_csv(scores)
    assert set(s["held_generator"]) == {"genA", "genB"}, "folds wrong"
    assert set(s["split"]) == {"test", "calib"}, "splits wrong"
    assert s["score"].between(0, 1).all(), "scores not probabilities"
    te = s[s["split"] == "test"]
    assert (te.groupby(["held_generator", "video_id"])["gop_idx"].nunique() >= 1).all()
    print(f"[smoke] streaming_scores OK: {len(s)} rows, "
          f"{te['video_id'].nunique()} test clips scored per fold")
    print("[smoke] ALL LOCAL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
