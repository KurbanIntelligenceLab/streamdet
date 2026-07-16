"""E7e motion-bias control: is stage-1 riding the 'AI moves less' shortcut?

The motion-bias critique (arXiv:2607.00948) shows motion-based AIGV detectors
can separate real from generated purely on motion magnitude. We test our
codec stage-1 against it two ways, per LOGO fold, on the FINAL running-max
score:
  (1) Motion-matched AUC: subsample real and generated test clips to a common
      mean-motion (mean_mv_mag) distribution (quantile bin matching), then
      recompute OOD AUC. A large drop = the signal was mostly motion magnitude.
  (2) Within-bin AUC: AUC computed inside each motion quantile bin (where
      real/gen motion is, by construction, comparable), then averaged. A signal
      that survives here is not the movement shortcut.
Also reports the raw AUC of mean_mv_mag ALONE as the shortcut's own strength.

Needs the per-chunk feature table (for mean_mv_mag per clip = mean over its
chunks) and the stage-1 streaming scores (for the final score + label).

Run: python Code/streamdet/analyze_motionbias.py \
        --scores <stream-scores.csv> --gop-features <gopfeat glob> \
        --out $OUT [--n-bins 10]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import streamdet  # noqa: F401
from streamdet import metrics as SM
from streamdet.analysis.streaming import score_matrix


def matched_indices(motion, y, n_bins, rng):
    """Quantile-bin motion; within each bin keep min(#real,#gen) of each class."""
    edges = np.quantile(motion, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    keep = []
    for b in range(n_bins):
        m = (motion >= edges[b]) & (motion < edges[b + 1])
        idx = np.where(m)[0]
        r = idx[y[idx] == 0]
        g = idx[y[idx] == 1]
        k = min(len(r), len(g))
        if k == 0:
            continue
        keep.append(rng.choice(r, k, replace=False))
        keep.append(rng.choice(g, k, replace=False))
    return np.concatenate(keep) if keep else np.array([], int)


def within_bin_auc(score, y, motion, n_bins):
    edges = np.quantile(motion, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    aucs, weights = [], []
    for b in range(n_bins):
        m = (motion >= edges[b]) & (motion < edges[b + 1])
        yy = y[m]
        if yy.sum() == 0 or (yy == 0).sum() == 0:
            continue
        aucs.append(SM.roc_auc(score[m], yy))
        weights.append(len(yy))
    if not aucs:
        return float("nan")
    return float(np.average(aucs, weights=weights))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--gop-features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-bins", type=int, default=10)
    args = ap.parse_args(argv)

    from streamdet.scoring.e1_readout import read_table
    sc = read_table(args.scores)
    gf = read_table(args.gop_features)
    motion = gf.groupby("video_id")["mean_mv_mag"].mean()      # per-clip motion
    folds = sorted(sc["held_generator"].unique())
    rng = np.random.default_rng(0)
    print(f"analyze_motionbias v1 folds={folds} n_bins={args.n_bins}", flush=True)

    rows = []
    for held in folds:
        te = sc[(sc["held_generator"] == held) & (sc["split"] == "test")]
        M, y, vids, _ = score_matrix(te)
        s = M[:, -1]
        mo = motion.reindex(vids).to_numpy()
        ok = ~np.isnan(mo)
        s, y, mo = s[ok], y[ok], mo[ok]
        raw_auc = SM.roc_auc(s, y)
        motion_only_auc = SM.roc_auc(mo, y)
        idx = matched_indices(mo, y, args.n_bins, rng)
        matched_auc = SM.roc_auc(s[idx], y[idx]) if len(idx) else float("nan")
        wb_auc = within_bin_auc(s, y, mo, args.n_bins)
        rows.append({"held_generator": held, "raw_auc": raw_auc,
                     "motion_only_auc": motion_only_auc,
                     "motion_matched_auc": matched_auc,
                     "within_bin_auc": wb_auc, "n": len(y),
                     "n_matched": len(idx)})
        print(f"fold held={held} raw={raw_auc:.4f} motion_only={motion_only_auc:.4f} "
              f"matched={matched_auc:.4f} within_bin={wb_auc:.4f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"result raw={out.raw_auc.mean():.4f} "
          f"motion_only={out.motion_only_auc.mean():.4f} "
          f"matched={out.motion_matched_auc.mean():.4f} "
          f"within_bin={out.within_bin_auc.mean():.4f} out={args.out}", flush=True)


if __name__ == "__main__":
    main()
