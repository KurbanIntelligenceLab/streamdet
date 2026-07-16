"""Per-GOP codec-MV features for the streaming harness.

Segments a clip's motion-vector array [T, H_mb, W_mb, 5] into fixed-length GOP
windows (the vidaudit P1 canonical re-encode pins keyint=16, scenecut=0, so
frame k belongs to GOP k//16 exactly) and runs vidaudit's shared 13-d spectral
extractor on each window independently. One output row per (clip, GOP).

Run as an array task (one process per shard):
    python Code/streamdet/gop_features.py \
        --manifest <manifest.csv> --num-shards N --shard-index $TASK_ID \
        --out $OUT [--gop-len 16] [--max-gops 32]

Manifest columns (vidaudit convention): video_id, generator, label, is_real, mp4_path.
Output CSV columns: video_id, generator, label, is_real, gop_idx, n_gops_clip,
n_frames_gop, iframe_ok, decode_s, feat_s, <13 feature columns>.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

import streamdet  # noqa: F401  (path bootstrap)
from vidaudit.detectors.temporalspec import TemporalSpec
from vidaudit.features import FEATURE_NAMES
from vidaudit.features.mv import extract_features

META_COLS = ["video_id", "generator", "label", "is_real", "gop_idx",
             "n_gops_clip", "n_frames_gop", "iframe_ok", "decode_s", "feat_s"]


def read_frame_types(npy_path: str):
    """Read the I/P/B sequence stored beside a precomputed MV .npy, or None."""
    ft = os.path.join(os.path.dirname(npy_path), "frame_types.txt")
    if not os.path.exists(ft):
        return None
    with open(ft) as f:
        return [t.strip() for t in f.read().strip().splitlines() if t.strip()]


def segment_by_frame_types(mv: np.ndarray, types):
    """Split [T,H,W,5] at the encoder's true I-frames; yield (gop_idx, window, True).

    The authoritative segmentation when frame_types.txt exists: GOP k spans
    [I_k, I_{k+1}). Trailing frames beyond len(types) (or vice versa) are
    tolerated up to a 1-frame slack; larger mismatches raise.
    """
    T = mv.shape[0]
    if abs(T - len(types)) > 1:
        raise ValueError(f"mv frames ({T}) != frame_types ({len(types)})")
    i_pos = [k for k, t in enumerate(types[:T]) if t == "I"] or [0]
    bounds = i_pos + [T]
    for g in range(len(i_pos)):
        yield g, mv[bounds[g]:bounds[g + 1]], True


def segment_gops(mv: np.ndarray, gop_len: int = 16):
    """Split [T,H,W,5] into fixed windows; yield (gop_idx, window, iframe_ok).

    iframe_ok: whether the window's first frame carries zero motion (an I-frame),
    which the canonical keyint=16/scenecut=0 recipe guarantees. A False value
    flags a clip whose encode does not match the canonical GOP structure.
    """
    T = mv.shape[0]
    dx = mv[..., 1] - mv[..., 3]
    dy = mv[..., 2] - mv[..., 4]
    frame_motion = (np.abs(dx) + np.abs(dy)).reshape(T, -1).sum(axis=1)
    for k in range(0, T, gop_len):
        window = mv[k:k + gop_len]
        yield k // gop_len, window, bool(frame_motion[k] == 0)


def extract_mv_capped(path: str, mb: int = 16, max_frames: int | None = None):
    """Decode H.264 motion vectors into [T,H,W,5], STOPPING after max_frames.

    Mirrors TemporalSpec._extract_mv but bounds the decode loop, so a very long
    clip never loads its whole self into memory. Used for direct (non-canonical)
    extraction of arbitrary-codec cells like AIGVDBench, where re-encoding every
    clip is too fragile/heavy. H.264 sources give real MVs; HEVC yields zeros
    (flagged downstream as degenerate)."""
    import av
    from av.codec.context import Flags2
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        stream.codec_context.flags2 |= Flags2.export_mvs
        dim_x = dim_y = None
        grids = []
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
            if max_frames and len(grids) >= max_frames:
                break
    finally:
        container.close()
    if not grids:
        raise RuntimeError(f"no frames decoded: {path}")
    return np.stack(grids, axis=0)


def _canonicalize_tmp(path: str, max_frames: int | None = None) -> str:
    """Re-encode `path` to a temp H.264 clip under the canonical recipe
    (keyint=16, scenecut=0) so codec MVs match the training cell. With
    max_frames, only the first N frames are re-encoded (`-frames:v`), which
    bounds BOTH the re-encode and the later full decode — long clips (some
    AIGVDBench reals are thousands of frames) otherwise blow memory/time.
    Returns the temp path (caller deletes it)."""
    import tempfile
    from vidaudit.data.canonical import ffmpeg_cmd
    import subprocess
    fd, dst = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    cmd = ffmpeg_cmd(path, dst)
    if max_frames:                       # insert -frames:v N just before dst
        cmd = cmd[:-1] + ["-frames:v", str(max_frames), cmd[-1]]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, timeout=600)
    if proc.returncode != 0:
        try:
            os.remove(dst)
        except OSError:
            pass
        raise RuntimeError("canonical re-encode failed: " +
                           "\n".join(proc.stderr.splitlines()[-3:]))
    return dst


def clip_gop_rows(path: str, gop_len: int = 16, max_gops: int | None = None,
                  mb: int = 16, npy_path: str | None = None,
                  target_frames: int | None = 12, canonicalize: bool = False,
                  max_frames: int | None = None):
    """Extract per-GOP 13-d feature rows for one clip.

    Two input modes: decode an mp4 via PyAV export_mvs (path), or load a
    precomputed MV array (npy_path) with its sibling frame_types.txt giving
    the encoder's true GOP boundaries. max_frames caps how many frames are
    processed per clip (bounds memory/time on very long clips; only the
    streaming prefix matters). Returns (rows, n_gops_total, decode_s); GOPs
    whose window is degenerate (extract_features -> None) are skipped.
    """
    t0 = time.monotonic()
    tmp = None
    if npy_path:
        mv = np.load(npy_path)
        if max_frames:
            mv = mv[:max_frames]
        types = read_frame_types(npy_path)
        if types and max_frames:
            types = types[:max_frames]
    elif canonicalize:
        tmp = _canonicalize_tmp(path, max_frames=max_frames)
        try:
            mv = TemporalSpec(mb=mb)._extract_mv(tmp)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
        if max_frames:
            mv = mv[:max_frames]
        types = None
    else:
        # direct, decode-bounded extraction (no re-encode subprocess)
        mv = extract_mv_capped(path, mb=mb, max_frames=max_frames)
        types = None
    decode_s = time.monotonic() - t0

    # Fixed-length windows are the arrival unit: encoder GOPs in the wild are
    # too long to stream on (the recorded GenVidBench arrays carry ONE I-frame
    # per clip), so the detector chunks the arriving frames itself; gop_len is
    # the E7 chunk-size knob. When frame types are known, iframe_ok records
    # whether the window START coincides with a true I-frame (it does under
    # the canonical keyint=gop_len re-encode).
    segments = segment_gops(mv, gop_len)
    rows = []
    n_gops = 0
    for gop_idx, window, iframe_ok in segments:
        if types is not None:
            k = gop_idx * gop_len
            iframe_ok = k < len(types) and types[k] == "I"
        n_gops += 1
        if max_gops is not None and gop_idx >= max_gops:
            continue
        t1 = time.monotonic()
        # target_frames pins every scored chunk to the SAME number of motion
        # frames (WACV protocol uses 12 at clip level): without it, feature
        # values depend on window length and the varying last partial chunk
        # leaks length into the readout. Short windows (<target) are skipped.
        feats = extract_features(window, target_frames=target_frames)
        feat_s = time.monotonic() - t1
        if feats is None:
            continue
        row = {"gop_idx": gop_idx, "n_frames_gop": int(window.shape[0]),
               "iframe_ok": int(iframe_ok), "feat_s": round(feat_s, 6)}
        row.update({k: feats[k] for k in FEATURE_NAMES})
        rows.append(row)
    return rows, n_gops, decode_s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--gop-len", type=int, default=16)
    ap.add_argument("--max-gops", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap frames processed per clip (bounds memory/time on "
                         "long clips); pair with --max-gops for streaming")
    ap.add_argument("--mb", type=int, default=16)
    ap.add_argument("--target-frames", type=int, default=12,
                    help="pin every chunk's feature window to this many motion "
                         "frames (0 disables); chunks with fewer are skipped")
    ap.add_argument("--canonicalize", action="store_true",
                    help="re-encode each mp4 to the canonical H.264 recipe "
                         "(keyint=16) before MV extraction; for non-normalized "
                         "cells like AIGVDBench (ignored in npy mode)")
    args = ap.parse_args(argv)

    with open(args.manifest, newline="") as f:
        clips = list(csv.DictReader(f))
    mine = clips[args.shard_index::args.num_shards]
    print(f"gop_features v1 gop_len={args.gop_len} max_gops={args.max_gops} "
          f"shard={args.shard_index}/{args.num_shards} clips={len(mine)} "
          f"manifest={args.manifest}", flush=True)

    fieldnames = META_COLS + list(FEATURE_NAMES)
    n_ok = n_fail = n_rows = 0
    t_start = time.monotonic()
    with streamdet.atomic_out(args.out) as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for i, c in enumerate(mine):
            try:
                rows, n_gops, decode_s = clip_gop_rows(
                    c.get("mp4_path"), gop_len=args.gop_len,
                    max_gops=args.max_gops, mb=args.mb,
                    npy_path=c.get("npy_path") or None,
                    target_frames=args.target_frames or None,
                    canonicalize=args.canonicalize,
                    max_frames=args.max_frames)
                for r in rows:
                    r.update({"video_id": c["video_id"], "generator": c["generator"],
                              "label": c["label"], "is_real": c["is_real"],
                              "n_gops_clip": n_gops, "decode_s": round(decode_s, 6)})
                    w.writerow(r)
                n_rows += len(rows)
                n_ok += 1
            except Exception as e:  # keep the shard alive; report at the end
                n_fail += 1
                print(f"FAIL {c.get('video_id')}: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
            if (i + 1) % 25 == 0 or (i + 1) == len(mine):
                el = time.monotonic() - t_start
                print(f"progress done={i+1}/{len(mine)} ok={n_ok} fail={n_fail} "
                      f"rows={n_rows} elapsed_s={el:.0f}", flush=True)

    print(f"result shard={args.shard_index} clips_ok={n_ok} clips_fail={n_fail} "
          f"gop_rows={n_rows} out={args.out}", flush=True)
    if n_ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
