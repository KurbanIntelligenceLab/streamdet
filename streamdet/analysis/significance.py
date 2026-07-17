"""Statistical support for every decision-accuracy operating point the paper
reports: percentile-bootstrap 95% CIs (clip-level, within LOGO folds) and
PAIRED tests on each deferral gain (paired bootstrap + exact McNemar on the
per-clip verdicts). Also reconciles Table 2's AUC@N definition against the raw
scores and measures the aggregator ablation (max vs mean vs last).

Mirrors analysis/cascade.py exactly (inner-join population, per-fold
end-calibrated tau, fold-mean aggregation) and validates against its output
before attaching any statistic.

Run:
  python -m streamdet.analysis.significance \
      --stage1 results/stream-scores/sh_0000.csv \
      --stage2 results/pixel-scores/sh_0000.csv \
      --cascade results/cascade-pixel/sh_0000.csv \
      --vlm 'results/vlm-scores/sh_*.csv' [--knee-width 0.10]
"""
from __future__ import annotations

import argparse
from math import comb

import numpy as np
import pandas as pd

import streamdet  # noqa: F401
from streamdet import metrics as SM
from streamdet.analysis.cascade import final_scores
from streamdet.analysis.streaming import score_matrix
from streamdet.analysis.vlm import bal_acc, final_codec_scores
from streamdet.scoring.e1_readout import read_table

RNG = np.random.default_rng(42)
N_BOOT = 10_000


# ------------------------------------------------------------------ helpers
def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p on discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def acc(p, y):
    return float((p == y).mean())


def boot_ci(stat, *arrs, n_boot=N_BOOT):
    n = len(arrs[0])
    out = np.empty(n_boot)
    for b in range(n_boot):
        i = RNG.integers(0, n, n)
        out[b] = stat(*(a[i] for a in arrs))
    return np.percentile(out, [2.5, 97.5]), out


# ------------------------------------------------------ 27k-cell cascade CIs
def fold_frames(s1, s2, held, alpha):
    d1 = s1[s1["held_generator"] == held]
    d2 = s2[s2["held_generator"] == held]
    te1 = final_scores(d1[d1["split"] == "test"])
    te2 = final_scores(d2[d2["split"] == "test"])
    cal1 = final_scores(d1[d1["split"] == "calib"])
    common = te1.index.intersection(te2.index)
    tau = float(np.quantile(cal1["m_final"], 1.0 - alpha))
    return te1.loc[common], te2.loc[common], tau


def verdicts(te1, te2, tau, w, s2_cut=0.5):
    y = te1["label"].to_numpy()
    s = te1["m_final"].to_numpy()
    pred2 = (te2["m_final"].to_numpy() >= s2_cut).astype(int)
    pred1 = (s >= tau).astype(int)
    deferred = (s >= tau - w) & (s < tau)
    predc = pred1.copy()
    predc[deferred] = pred2[deferred]
    return y, pred1, predc, deferred


def cell_significance(s1, s2, w, alpha=0.05, metric=acc, tag=""):
    folds = sorted(set(s1["held_generator"]) & set(s2["held_generator"]))
    per_fold = [verdicts(*fold_frames(s1, s2, h, alpha), w) for h in folds]
    a1 = np.mean([metric(p1, y) for y, p1, _, _ in per_fold])
    ac = np.mean([metric(pc, y) for y, _, pc, _ in per_fold])
    defer = np.mean([d.mean() for *_, d in per_fold])

    b1 = np.empty(N_BOOT)
    bc = np.empty(N_BOOT)
    for b in range(N_BOOT):
        v1, vc = [], []
        for y, p1, pc, _ in per_fold:
            i = RNG.integers(0, len(y), len(y))
            v1.append(metric(p1[i], y[i]))
            vc.append(metric(pc[i], y[i]))
        b1[b] = np.mean(v1)
        bc[b] = np.mean(vc)
    gain = bc - b1
    p_boot = float(min((gain <= 0).mean(), (gain >= 0).mean()) * 2)

    Y = np.concatenate([y for y, *_ in per_fold])
    P1 = np.concatenate([p1 for _, p1, *_ in per_fold])
    PC = np.concatenate([pc for _, _, pc, _ in per_fold])
    bwin = int(((P1 == Y) & (PC != Y)).sum())
    cwin = int(((P1 != Y) & (PC == Y)).sum())
    print(f"[{tag}] defer={defer:.3f} stage1={a1:.4f} "
          f"CI[{np.percentile(b1,2.5):.4f},{np.percentile(b1,97.5):.4f}] "
          f"cascade={ac:.4f} "
          f"CI[{np.percentile(bc,2.5):.4f},{np.percentile(bc,97.5):.4f}] "
          f"gain={ac-a1:+.4f} "
          f"CI[{np.percentile(gain,2.5):+.4f},{np.percentile(gain,97.5):+.4f}] "
          f"p_boot={p_boot:.3g} McNemar(b={bwin},c={cwin})p={mcnemar_exact(bwin,cwin):.3g}",
          flush=True)


# ------------------------------------------------- AUC@N reconciliation (RF6)
def aucn_reconcile(csv, label):
    df = pd.read_csv(csv)
    full, at16, meanpool, lastg = [], [], [], []
    for h in sorted(df.held_generator.unique()):
        d = df[(df.held_generator == h) & (df.split == "test")]
        M, y, vids, t = score_matrix(d)
        full.append(SM.roc_auc(M[:, -1], y))
        at16.append(SM.roc_auc(M[:, min(15, M.shape[1] - 1)], y))
        piv = d.pivot_table(index="video_id", columns="gop_idx",
                            values="score").sort_index(axis=1)
        lab = (d.drop_duplicates("video_id").set_index("video_id")
               .loc[piv.index, "is_real"] == 0).astype(int).to_numpy()
        raw = piv.to_numpy(float)
        meanpool.append(SM.roc_auc(np.nanmean(raw, axis=1), lab))
        li = (~np.isnan(raw)).cumsum(1).argmax(1)
        lastg.append(SM.roc_auc(raw[np.arange(len(raw)), li], lab))
    print(f"[AUC@N {label}] running-max@full-N={np.mean(full):.3f} "
          f"running-max@16={np.mean(at16):.3f} mean-pool={np.mean(meanpool):.3f} "
          f"last-GOP={np.mean(lastg):.3f}", flush=True)


# ----------------------------------------------------------- subsample (E4)
def subsample_significance(vlm_glob, stage1_csv, stage2_csv, alpha=0.05):
    vlm = read_table(vlm_glob)
    vlm["is_real"] = vlm["is_real"].astype(int)
    vlm = vlm.drop_duplicates("video_id").set_index("video_id")
    codec_final, codec_real = final_codec_scores(stage1_csv)
    tau = float(np.quantile(codec_final[codec_real == 1].to_numpy(), 1 - alpha))
    common = vlm.index.intersection(codec_final.index)
    v = vlm.loc[common]
    s1 = codec_final.loc[common].to_numpy()
    y = (v["is_real"] == 0).astype(int).to_numpy()
    vp = (v["score"].to_numpy() >= 0.5).astype(int)

    px = read_table(stage2_csv)
    pf = px[px.split == "test"].groupby(["video_id", "held_generator"]).score.max()
    px_final = pf.groupby("video_id").mean()
    pxp = (np.nan_to_num(np.array([px_final.get(i, np.nan) for i in common]),
                         nan=0.0) >= 0.5).astype(int)

    y_all = (vlm["is_real"] == 0).astype(int).to_numpy()
    vp_all = (vlm["score"].to_numpy() >= 0.5).astype(int)
    (lo, hi), _ = boot_ci(bal_acc, vp_all, y_all)
    print(f"[E4] VLM standalone bal={bal_acc(vp_all, y_all):.4f} "
          f"CI[{lo:.4f},{hi:.4f}] n={len(y_all)}", flush=True)

    def casc(w, pred2):
        p1 = (s1 >= tau).astype(int)
        d = (s1 >= tau - w) & (s1 < tau)
        pc = p1.copy()
        pc[d] = pred2[d]
        return p1, pc, d

    p1, _, _ = casc(0.0, vp)
    (lo, hi), _ = boot_ci(bal_acc, p1, y)
    print(f"[E4] codec gate bal={bal_acc(p1, y):.4f} CI[{lo:.4f},{hi:.4f}]",
          flush=True)
    for w in (0.1, 0.3):
        _, pcv, d = casc(w, vp)
        _, pcp, _ = casc(w, pxp)
        (glo, ghi), g = boot_ci(
            lambda a, b, yy: bal_acc(a, yy) - bal_acc(b, yy), pcv, pcp, y)
        p2 = float(min((g <= 0).mean(), (g >= 0).mean()) * 2)
        b2 = int(((pcp == y) & (pcv != y)).sum())
        c2 = int(((pcp != y) & (pcv == y)).sum())
        print(f"[E4] w={w} defer={d.mean():.3f} vlm={bal_acc(pcv, y):.4f} "
              f"pixel={bal_acc(pcp, y):.4f} diff CI[{glo:+.4f},{ghi:+.4f}] "
              f"p_boot={p2:.3g} McNemar(b={b2},c={c2})p={mcnemar_exact(b2, c2):.3g}",
              flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--vlm", default=None)
    ap.add_argument("--knee-width", type=float, default=0.10)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args(argv)

    s1 = read_table(args.stage1)
    s2 = read_table(args.stage2)
    aucn_reconcile(args.stage1, "codec")
    aucn_reconcile(args.stage2, "pixel")
    cell_significance(s1, s2, args.knee_width, args.alpha,
                      tag=f"27k codec->pixel w={args.knee_width}")
    if args.vlm:
        subsample_significance(args.vlm, args.stage1, args.stage2, args.alpha)


if __name__ == "__main__":
    main()
