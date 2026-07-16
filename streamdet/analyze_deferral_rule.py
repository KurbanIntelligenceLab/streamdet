"""E7 (learned deferral rule vs confidence gate).

The confidence band W=[tau-w,tau) defers clips near the boundary. A learned
rule can instead predict, from the stage-1 evidence, WHICH clips stage-2 would
fix. We compare the two at MATCHED deferral budgets, per LOGO fold, via an
honest within-test 5-fold CV (the rule is trained and evaluated on disjoint
test-clip folds, never on the clips it scores):
  target  = stage-1 wrong AND stage-2 right   (a clip worth escalating)
  features= [final stage-1 score m, |m-tau|, per-clip mean stage-1 codec feats]
For a grid of deferral budgets p, the learned rule escalates its top-p clips by
predicted escalation value; the band escalates the p clips closest below tau.
Report cascade accuracy of each at each budget -> which selection is better.

Run: python Code/streamdet/analyze_deferral_rule.py \
        --stage1 <codec scores> --stage2 <pixel scores> \
        --gop-features <gopfeat glob> --out $OUT
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import streamdet  # noqa: F401
import streamdet_metrics as SM
from analyze_streaming import score_matrix
from vidaudit.features import FEATURE_NAMES
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def fold_frame(sc, gf, held, tau_alpha):
    te = sc[(sc["held_generator"] == held) & (sc["split"] == "test")]
    M, y, vids, _ = score_matrix(te)
    s = M[:, -1]
    feat = gf.groupby("video_id")[list(FEATURE_NAMES)].mean().reindex(vids)
    ok = feat.notna().all(axis=1).to_numpy()
    return s[ok], y[ok], vids[ok], feat[ok].to_numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--gop-features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--budgets", default="0.02,0.05,0.1,0.15,0.2,0.3")
    args = ap.parse_args(argv)

    from e1_readout import read_table
    s1 = read_table(args.stage1)
    s2 = read_table(args.stage2)
    gf = read_table(args.gop_features)
    budgets = [float(b) for b in args.budgets.split(",")]
    folds = sorted(set(s1["held_generator"]) & set(s2["held_generator"]))
    print(f"analyze_deferral_rule v1 folds={folds} budgets={budgets}", flush=True)

    rows = []
    for held in folds:
        # stage-1 side
        te1 = s1[(s1["held_generator"] == held) & (s1["split"] == "test")]
        cal1 = s1[(s1["held_generator"] == held) & (s1["split"] == "calib")]
        Mc, _, _, _ = score_matrix(cal1)
        tau = float(np.quantile(Mc[:, -1], 1.0 - args.alpha))
        s, y, vids, X = fold_frame(s1, gf, held, tau)
        # stage-2 verdict aligned to same clips
        M2, y2, v2, _ = score_matrix(s2[(s2["held_generator"] == held) &
                                        (s2["split"] == "test")])
        pred2_by_vid = dict(zip(v2, (M2[:, -1] >= 0.5).astype(int)))
        pred2 = np.array([pred2_by_vid.get(v, 0) for v in vids])

        pred1 = (s >= tau).astype(int)
        worth = ((pred1 != y) & (pred2 == y)).astype(int)   # escalation target
        Xr = np.column_stack([s, np.abs(s - tau), np.nan_to_num(X)])

        # learned escalation value via 5-fold within-test CV (disjoint eval)
        val = np.zeros(len(s))
        if worth.sum() >= 5 and (worth == 0).sum() >= 5:
            skf = StratifiedKFold(5, shuffle=True, random_state=0)
            for tr, ev in skf.split(Xr, worth):
                clf = LogisticRegression(max_iter=1000, class_weight="balanced")
                clf.fit(Xr[tr], worth[tr])
                val[ev] = clf.predict_proba(Xr[ev])[:, 1]
        else:
            val = -np.abs(s - tau)      # degenerate: fall back to band proximity

        band_prox = -np.abs(s - (tau - 1e-9))   # closeness just below tau
        band_prox[s >= tau] = -np.inf           # already-flagged never deferred
        for p in budgets:
            k = int(round(p * len(s)))
            if k == 0:
                continue
            for name, order in (("band", band_prox), ("learned", val)):
                defer = np.zeros(len(s), bool)
                defer[np.argsort(order)[::-1][:k]] = True
                pred = pred1.copy()
                pred[defer] = pred2[defer]
                rows.append({"held_generator": held, "budget": p, "rule": name,
                             "defer_rate": defer.mean(),
                             "accuracy": 1 - SM.error_rate(pred, y),
                             "err_stage1": SM.error_rate(pred1, y),
                             "n": len(y)})
        print(f"fold held={held} tau={tau:.4f} worth={int(worth.sum())}/{len(s)}",
              flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    piv = out.groupby(["budget", "rule"])["accuracy"].mean().unstack()
    print(piv.to_csv(float_format="%.4f"), flush=True)
    print(f"result folds={len(folds)} rows={len(out)} out={args.out}", flush=True)


if __name__ == "__main__":
    main()
