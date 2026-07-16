"""Fill the VLM cells: baseline AUC + codec->VLM cascade on the subsample.

Joins the VLM per-clip scores to the stage-1 codec scores (by video_id, taking
each clip's final running-max codec score from the fold where it is a test clip)
and to the pixel per-clip scores, on the SAME subsample population. Reports:
  * VLM standalone AUC (the expensive upper-bound baseline) and mean latency;
  * codec and pixel AUC on the subsample (for an apples-to-apples cascade);
  * codec->VLM and codec->pixel cascade accuracy over a deferral sweep, so the
    E4 "which escalation recovers more per unit compute" question is answered on
    one population.

Run: python Code/streamdet/vlm_analysis.py --vlm results/vlm-scores/sh_*.csv \
        --stage1 results/stream-scores/sh_0000.csv \
        --pixel results/pixel-scores/sh_0000.csv --out $OUT
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import streamdet  # noqa: F401
from streamdet import metrics as SM


def bal_acc(pred, y):
    """Balanced accuracy 0.5(TPR+TNR); imbalance-invariant, so the gen-heavy VLM
    subsample is comparable to the balanced main-cell cascade."""
    pred = np.asarray(pred).astype(int); y = np.asarray(y).astype(int)
    tpr = (pred[y == 1] == 1).mean() if (y == 1).any() else 0.0
    tnr = (pred[y == 0] == 0).mean() if (y == 0).any() else 0.0
    return float(0.5 * (tpr + tnr))


def final_codec_scores(stage1_csv):
    """video_id -> per-clip final running-max codec score.

    A clip's final running-max in one fold is the max over its per-GOP scores
    there. Reals appear in every fold's test set (each with that fold's OOD
    readout), so we average their per-fold finals; a generated clip appears only
    in its own held-out fold, so its mean is just that one score. This yields one
    consistent codec decision score per clip for the cascade."""
    df = pd.read_csv(stage1_csv)
    te = df[df.split == "test"]
    per_fold_final = te.groupby(["video_id", "held_generator"]).score.max()
    clip_final = per_fold_final.groupby("video_id").mean()
    is_real = te.groupby("video_id").is_real.first()
    return clip_final, is_real


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vlm", required=True)
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--pixel", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--widths", default="0,0.05,0.1,0.15,0.2,0.3,0.5")
    args = ap.parse_args(argv)

    from streamdet.scoring.e1_readout import read_table
    vlm = read_table(args.vlm)
    vlm["is_real"] = vlm["is_real"].astype(int)
    vlm = vlm.drop_duplicates("video_id").set_index("video_id")
    y_all = (vlm["is_real"] == 0).astype(int)
    print(f"vlm_analysis v1 vlm_clips={len(vlm)} "
          f"real={int((vlm.is_real==1).sum())} gen={int((vlm.is_real==0).sum())} "
          f"mean_latency_s={vlm.latency_s.mean():.1f}", flush=True)

    vlm_auc = SM.roc_auc(vlm["score"].to_numpy(), y_all.to_numpy())
    m, lo, hi = SM.bootstrap_ci(SM.roc_auc, vlm["score"].to_numpy(),
                                y_all.to_numpy(), n_boot=1000, seed=1)
    vlm_acc = ((vlm["score"] >= 0.5).astype(int) == y_all).mean()
    print(f"[VLM baseline] AUC={vlm_auc:.3f} boot95[{lo:.3f},{hi:.3f}] "
          f"acc@0.5={vlm_acc:.3f} mean_latency_s={vlm.latency_s.mean():.1f}", flush=True)

    codec_final, codec_real = final_codec_scores(args.stage1)
    # calibrate tau on the FULL test-real codec population (stable), not the
    # handful of reals that happen to be in the subsample
    real_finals = codec_final[codec_real == 1].to_numpy()
    tau = float(np.quantile(real_finals, 1 - args.alpha))
    # cascade population = subsample clips with an unbiased (test-fold) codec
    # score; VLM is zero-shot so its baseline AUC above used ALL subsample clips
    common = vlm.index.intersection(codec_final.index)
    v = vlm.loc[common]
    s1 = codec_final.loc[common].to_numpy()
    y = (v["is_real"] == 0).astype(int).to_numpy()
    vlm_pred = (v["score"].to_numpy() >= 0.5).astype(int)
    codec_auc = SM.roc_auc(s1, y)

    stage2 = {"vlm": vlm_pred}
    if args.pixel:
        px = read_table(args.pixel)
        pf = px[px.split == "test"].groupby(["video_id", "held_generator"]).score.max()
        px_final = pf.groupby("video_id").mean()
        pcom = np.array([px_final.get(i, np.nan) for i in common])
        stage2["pixel"] = (pcom >= 0.5).astype(int)

    codec_bacc = bal_acc((s1 >= tau).astype(int), y)
    widths = [float(w) for w in args.widths.split(",")]
    rows = []
    print(f"[cascade on test subsample] n={len(common)} tau={tau:.3f} "
          f"codec_AUC={codec_auc:.3f} codec_only_bal_acc={codec_bacc:.3f}", flush=True)
    # E4: on the near-boundary (deferred) clips, which stage-2 is more accurate?
    for w in [0.1, 0.15]:
        deferred = (s1 >= tau - w) & (s1 < tau)
        if deferred.sum() >= 10:
            line = (f"  [deferred band w={w}] n={int(deferred.sum())} "
                    f"codec_bacc={bal_acc((s1>=tau).astype(int)[deferred],y[deferred]):.3f} "
                    f"vlm_bacc={bal_acc(vlm_pred[deferred],y[deferred]):.3f}")
            if 'pixel' in stage2:
                line += f" pixel_bacc={bal_acc(stage2['pixel'][deferred],y[deferred]):.3f}"
            print(line, flush=True)
    for name, pred2 in stage2.items():
        for w in widths:
            window = (tau - w, tau - 1e-12)
            pred, deferred = SM.cascade_decision(s1, tau, window, pred2)
            rows.append({"stage2": name, "width": w, "defer": float(deferred.mean()),
                         "bal_acc": bal_acc(pred, y), "acc": 1 - SM.error_rate(pred, y)})
        best = max((r for r in rows if r["stage2"] == name), key=lambda r: r["bal_acc"])
        print(f"  stage2={name}: best defer={best['defer']:.3f} bal_acc={best['bal_acc']:.4f} "
              f"(codec-only bal_acc={codec_bacc:.4f})", flush=True)

    out = pd.DataFrame(rows)
    out["vlm_auc"] = vlm_auc; out["codec_auc"] = codec_auc
    out["vlm_mean_latency_s"] = vlm.latency_s.mean()
    out.to_csv(args.out, index=False)
    print(f"result vlm_auc={vlm_auc:.3f} codec_auc_sub={codec_auc:.3f} "
          f"out={args.out}", flush=True)


if __name__ == "__main__":
    main()
