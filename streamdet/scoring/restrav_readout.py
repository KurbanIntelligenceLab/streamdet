"""RF1 baseline readout: LOGO L2-LR over the per-clip ReStraV features.

Reuses the audited split path verbatim (streaming_scores.logo_splits: seed 42,
real 80/20, the 75/25 calibration carve-out) and the same readout recipe
(median-impute -> z-score -> balanced L2-LR), so the ReStraV point is evaluated
under exactly the protocol of every other stage: per-fold OOD scores on
test + calib, giving both an AUC and a gate decision accuracy at alpha=0.05.

Run (CPU, single task) after the restrav feature job:
    python Code/streamdet/restrav_readout.py \
        --features 'results/restrav-feat/sh_*.csv' --out $OUT
Output CSV: video_id, generator, label, is_real, split(calib|test),
held_generator, score  (clip-level; no gop_idx: ReStraV is offline).
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

import streamdet  # noqa: F401
from streamdet.scoring.e1_readout import read_table
from streamdet.scoring.streaming_scores import logo_splits, build_readout

FEATS = [f"f{i}" for i in range(21)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True,
                    help="glob of restrav feature shards")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    df = read_table(args.features)
    df["is_real"] = df["is_real"].astype(int)
    clips = df.drop_duplicates("video_id")[
        ["video_id", "generator", "label", "is_real"]].reset_index(drop=True)
    real_fit, real_cal, real_te, gens, gs = logo_splits(clips)
    print(f"restrav_readout v1 clips={len(clips)} generators={gens}", flush=True)

    feat = df.set_index("video_id")
    out_frames = []
    for held in gens:
        t0 = time.monotonic()
        held_tr, held_te = gs[held]
        others_tr = pd.concat([gs[g][0] for g in gens if g != held],
                              ignore_index=True)
        tr_clips = pd.concat([real_fit, others_tr], ignore_index=True)
        te_clips = pd.concat([real_te, held_te], ignore_index=True)

        tr = df[df["video_id"].isin(set(tr_clips["video_id"]))]
        te = df[df["video_id"].isin(set(te_clips["video_id"]))]
        cal = df[df["video_id"].isin(set(real_cal["video_id"]))]

        X_tr = tr[FEATS].astype("float64")
        y_tr = (tr["is_real"] == 0).astype(int).to_numpy()
        pipe = build_readout(len(FEATS)).fit(X_tr, y_tr)

        for split_name, part in (("test", te), ("calib", cal)):
            s = pipe.predict_proba(part[FEATS].astype("float64"))[:, 1]
            block = part[["video_id", "generator", "label", "is_real"]].copy()
            block["split"] = split_name
            block["held_generator"] = held
            block["score"] = s
            out_frames.append(block)
        print(f"fold held={held} train={len(tr)} test={len(te)} "
              f"cal={len(cal)} elapsed_s={time.monotonic()-t0:.0f}", flush=True)

    res = pd.concat(out_frames, ignore_index=True)
    res.to_csv(args.out, index=False)

    # concise stdout verdict: per-fold OOD AUC + gate decision accuracy
    import numpy as np
    from streamdet import metrics as SM
    aucs, accs = [], []
    for held in gens:
        d = res[(res.held_generator == held) & (res.split == "test")]
        y = (d.is_real == 0).astype(int).to_numpy()
        aucs.append(SM.roc_auc(d.score.to_numpy(), y))
        c = res[(res.held_generator == held) & (res.split == "calib")]
        tau = float(np.quantile(c.score.to_numpy(), 0.95))
        accs.append(float(((d.score.to_numpy() >= tau).astype(int) == y).mean()))
    print(f"result restrav LOGO AUC={np.mean(aucs):.4f} (sd {np.std(aucs):.4f}) "
          f"gate_acc@a05={np.mean(accs):.4f} folds={len(gens)} out={args.out}",
          flush=True)


if __name__ == "__main__":
    main()
