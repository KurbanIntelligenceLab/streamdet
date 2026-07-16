"""Stage-1 streaming scores: LOGO folds over per-GOP features.

Reproduces vidaudit's exact audited split path (seed 42, real 80/20
NON-stratified, per-generator 80/20 — see vidaudit.audit.protocol.run_logo) on
the CLIP level, then for each held-out generator trains the uniform readout
(median-impute -> z-score -> balanced L2-LR) on the PER-GOP rows of the train
clips and scores the per-GOP rows of the test clips. The result is, per fold,
one score per (test clip, GOP) — the s_t sequences the streaming metrics
consume (running max M_t, anytime AUC, gate calibration).

Run as a single task:
    python Code/streamdet/streaming_scores.py \
        --gop-features <merged_gop_features.csv> --out $OUT [--folds all]

Output CSV: video_id, generator, label, is_real, split(calib|test),
held_generator, gop_idx, score.  The calib rows are REAL clips held out of
BOTH training and test (a 25% carve-out of the real train pool): Prop 2's
threshold must be calibrated on a null sample the scorer was not fit on,
otherwise training bias deflates the calibration scores and the test-time
false-positive rate exceeds alpha. (This is the one deliberate departure from
vidaudit's run_logo, which trains its readout on the full real train pool;
E1's clip-level control uses vidaudit's protocol untouched.)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import streamdet  # noqa: F401  (path bootstrap)
from vidaudit.audit.protocol import build_readout
from vidaudit.data.cells import SEED
from sklearn.model_selection import train_test_split

META_COLS = {"video_id", "generator", "label", "is_real", "gop_idx",
             "n_gops_clip", "n_frames_gop", "n_frames_chunk", "iframe_ok",
             "decode_s", "feat_s", "split", "held_generator", "score"}


def feature_cols(df: pd.DataFrame):
    """All numeric non-meta columns (mirrors vidaudit resolve_feature_cols)."""
    cols = [c for c in df.columns if c not in META_COLS]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def clip_table(gop_df: pd.DataFrame) -> pd.DataFrame:
    """One row per clip, in first-appearance order, with identity columns."""
    return (gop_df[["video_id", "generator", "label", "is_real"]]
            .drop_duplicates("video_id").reset_index(drop=True))


def logo_splits(clips: pd.DataFrame):
    """Mirror of vidaudit run_logo's split construction (protocol.py:75-86),
    plus a calibration carve-out: real_tr is further split 75/25 into the
    readout's real training pool and a held-out null sample for gate
    calibration (see module docstring)."""
    real = clips[clips["is_real"] == 1].reset_index(drop=True)
    real_tr, real_te = train_test_split(real, test_size=0.2, random_state=SEED)
    real_fit, real_cal = train_test_split(real_tr, test_size=0.25,
                                          random_state=SEED)
    gens = sorted(clips.loc[clips["is_real"] == 0, "generator"].astype(str)
                  .unique().tolist())
    gs = {g: train_test_split(
              clips[clips["generator"].astype(str) == g].reset_index(drop=True),
              test_size=0.2, random_state=SEED) for g in gens}
    return real_fit, real_cal, real_te, gens, gs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gop-features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", default="all",
                    help="'all' or comma-separated held-out generator names")
    ap.add_argument("--reducer", default="none",
                    help="'none' (13-d codec) or 'pca' (512-d CLIP -> 13, "
                         "vidaudit's convention for appearance features)")
    ap.add_argument("--n-components", type=int, default=13)
    ap.add_argument("--manifest", default=None,
                    help="full cell manifest; clips present here but absent "
                         "from the feature table ABSTAIN (score 0.0) so the "
                         "evaluated population stays the audited cell")
    ap.add_argument("--keep-features", default=None,
                    help="comma-separated feature-column subset (E7c "
                         "feature-set ablation); default uses all")
    args = ap.parse_args(argv)

    from e1_readout import read_table
    gop = read_table(args.gop_features)
    gop["is_real"] = gop["is_real"].astype(int)
    feats = feature_cols(gop)
    if args.keep_features:
        want = [c.strip() for c in args.keep_features.split(",")]
        feats = [c for c in feats if c in want]
        if not feats:
            raise SystemExit(f"--keep-features matched no columns: {want}")
    abstain = pd.DataFrame(columns=["video_id", "generator", "label", "is_real"])
    if args.manifest:
        man = pd.read_csv(args.manifest)[["video_id", "generator", "label",
                                          "is_real"]]
        man["is_real"] = man["is_real"].astype(int)
        missing = ~man["video_id"].isin(set(gop["video_id"]))
        abstain = man[missing].reset_index(drop=True)
        print(f"abstaining clips (no scorable chunk): {len(abstain)} "
              f"by gen: {abstain.groupby('generator').size().to_dict()}",
              flush=True)
        gop = pd.concat([gop, abstain.assign(gop_idx=0)], ignore_index=True)
    clips = clip_table(gop)
    real_fit, real_cal, real_te, gens, gs = logo_splits(clips)
    folds = gens if args.folds == "all" else args.folds.split(",")

    print(f"streaming_scores v2 seed={SEED} clips={len(clips)} "
          f"gop_rows={len(gop)} n_features={len(feats)} reducer={args.reducer} "
          f"generators={gens} folds={folds}", flush=True)

    by_vid = gop.set_index("video_id")
    out_frames = []
    for held in folds:
        t0 = time.monotonic()
        held_tr, held_te = gs[held]
        others_tr = pd.concat([gs[g][0] for g in gens if g != held],
                              ignore_index=True)
        tr_clips = pd.concat([real_fit, others_tr], ignore_index=True)
        te_clips = pd.concat([real_te, held_te], ignore_index=True)

        tr = gop[gop["video_id"].isin(set(tr_clips["video_id"]))]
        te = gop[gop["video_id"].isin(set(te_clips["video_id"]))]
        cal = gop[gop["video_id"].isin(set(real_cal["video_id"]))]

        tr = tr[tr[feats].notna().any(axis=1)]      # abstainers never train
        X_tr = tr[feats].astype("float64")
        y_tr = (tr["is_real"] == 0).astype(int).to_numpy()
        pipe = build_readout(len(feats), reducer=args.reducer,
                             n_components=args.n_components).fit(X_tr, y_tr)

        for split_name, part in (("test", te), ("calib", cal)):
            X = part[feats].astype("float64")
            s = pipe.predict_proba(X)[:, 1]
            # a clip with no scorable chunk ABSTAINS: floor score, never flags
            s[X.isna().all(axis=1).to_numpy()] = 0.0
            block = part[["video_id", "generator", "label", "is_real",
                          "gop_idx"]].copy()
            block["split"] = split_name
            block["held_generator"] = held
            block["score"] = s
            out_frames.append(block)

        el = time.monotonic() - t0
        print(f"fold held={held} train_gops={len(tr)} test_gops={len(te)} "
              f"cal_gops={len(cal)} elapsed_s={el:.0f}", flush=True)

    res = pd.concat(out_frames, ignore_index=True)
    res.to_csv(args.out, index=False)
    print(f"result folds={len(folds)} rows={len(res)} out={args.out}", flush=True)


if __name__ == "__main__":
    main()
