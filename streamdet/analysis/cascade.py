"""E3/E4 cascade analysis: deferral sweep, compute accounting, deferral gain.

Consumes TWO streaming_scores outputs over the SAME cell and folds — stage 1
(codec) and stage 2 (pixel CLIP or VLM) — and, per fold:
  * forms each clip's stage-1 decision score (running max at its final prefix)
    and stage-2 prediction on the same clip;
  * calibrates tau on the stage-1 calib split (Prop 2);
  * sweeps the one-sided deferral band W = [tau - w, tau) over widths w;
  * reports, per width: deferral rate, expected compute E[C] = tbar*C1 + p*C2
    vs Monte-Carlo compute, stage-1 error, cascade error, and the deferral
    gain err1^W - err2^W (Prop 3's condition, tested not assumed);
  * verifies the exact decomposition err_casc = err1 - (err1^W - err2^W).

Costs C1 (per-chunk stage-1) and C2 (one stage-2 call) come from the latency
benchmark (--c1-ms/--c2-ms) or MACs (--c1/--c2), reported in whatever unit is
given. tbar is measured (mean chunks processed per clip).

Run: python Code/streamdet/analyze_cascade.py --stage1 <scores.csv> \
        --stage2 <scores.csv> --out $OUT [--alpha 0.05] [--c1 1 --c2 200]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import streamdet  # noqa: F401
from streamdet import metrics as SM
from streamdet.analysis.streaming import score_matrix


def final_scores(df: pd.DataFrame):
    """Per-clip final running-max score + label, index video_id."""
    M, y, vids, t_clip = score_matrix(df)
    return pd.DataFrame({"video_id": vids, "label": y, "m_final": M[:, -1],
                         "t_clip": t_clip}).set_index("video_id")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--c1", type=float, default=1.0,
                    help="stage-1 cost per chunk (ms or MACs)")
    ap.add_argument("--c2", type=float, default=200.0,
                    help="stage-2 cost per invocation (same unit)")
    ap.add_argument("--widths", default="0,0.02,0.05,0.1,0.15,0.2,0.3,0.5,0.7,0.9")
    args = ap.parse_args(argv)

    from streamdet.scoring.e1_readout import read_table
    s1 = read_table(args.stage1)
    s2 = read_table(args.stage2)
    widths = [float(w) for w in args.widths.split(",")]
    folds = sorted(set(s1["held_generator"]) & set(s2["held_generator"]))
    print(f"analyze_cascade v1 alpha={args.alpha} C1={args.c1} C2={args.c2} "
          f"folds={folds}", flush=True)

    rows = []
    for held in folds:
        d1 = s1[s1["held_generator"] == held]
        d2 = s2[s2["held_generator"] == held]
        te1 = final_scores(d1[d1["split"] == "test"])
        te2 = final_scores(d2[d2["split"] == "test"])
        cal1 = final_scores(d1[d1["split"] == "calib"])
        common = te1.index.intersection(te2.index)
        te1, te2 = te1.loc[common], te2.loc[common]
        y = te1["label"].to_numpy()
        s = te1["m_final"].to_numpy()
        tau = float(np.quantile(cal1["m_final"], 1.0 - args.alpha))
        tbar = float(te1["t_clip"].mean())
        # stage-2 verdict: Bayes cut on its (roughly calibrated) probability
        pred2 = (te2["m_final"].to_numpy() >= 0.5).astype(int)

        pred1 = (s >= tau).astype(int)
        err1 = SM.error_rate(pred1, y)
        for w in widths:
            window = (tau - w, tau - 1e-12)
            p_def = SM.deferral_rate(s, window)
            ec = SM.expected_compute(args.c1, args.c2, tbar, p_def)
            pred_c, deferred = SM.cascade_decision(s, tau, window, pred2)
            err_c = SM.error_rate(pred_c, y)
            e1w = SM.error_mass(pred1, y, deferred)
            e2w = SM.error_mass(pred2, y, deferred)
            ec_mc = float(tbar * args.c1 + deferred.mean() * args.c2)
            assert abs(err_c - (err1 - (e1w - e2w))) < 1e-9, "Prop 3 decomposition"
            rows.append({"held_generator": held, "width": w, "tau": tau,
                         "tbar": tbar, "defer_rate": p_def,
                         "expected_compute": ec, "mc_compute": ec_mc,
                         "err_stage1": err1, "err_cascade": err_c,
                         "err1_W": e1w, "err2_W": e2w,
                         "deferral_gain": e1w - e2w,
                         "condition_holds": bool(e2w <= e1w),
                         "n_test": len(y)})
        best = min((r for r in rows if r["held_generator"] == held),
                   key=lambda r: r["err_cascade"])
        print(f"fold held={held} tau={tau:.4f} err1={err1:.4f} "
              f"best: w={best['width']} defer={best['defer_rate']:.3f} "
              f"err_casc={best['err_cascade']:.4f} gain={best['deferral_gain']:.4f} "
              f"cond={best['condition_holds']}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    agg = out.groupby("width").agg(defer=("defer_rate", "mean"),
                                   EC=("expected_compute", "mean"),
                                   err1=("err_stage1", "mean"),
                                   errC=("err_cascade", "mean"),
                                   cond=("condition_holds", "mean"))
    print(agg.to_csv(float_format="%.4f"), flush=True)
    print(f"result folds={len(folds)} rows={len(out)} out={args.out}", flush=True)


if __name__ == "__main__":
    main()
