"""Compute every number the paper cites + emit pgfplots coordinates.

Reads the pulled result CSVs and prints (a) Table 2 values, (b) headline AUCs
with percentile bootstrap CIs, (c) the cascade FUSED-score AUC, (d) anytime
AUC(t) curve coordinates for Fig 3a, (e) the compute-accuracy Pareto for Fig 3b.
Run: python Code/streamdet/paper_numbers.py > scratch/paper_numbers.txt
"""
import sys
import numpy as np
import pandas as pd

sys.path[:0] = ["Code", "Code/streamdet", "Paper"]
from streamdet import metrics as SM
from streamdet.analysis.streaming import score_matrix
from vidaudit.features import FEATURE_NAMES


def fold_final(df, held, split="test"):
    d = df[(df.held_generator == held) & (df.split == split)]
    M, y, v, tc = score_matrix(d)
    return M, y, v


def cell_auc(csv, label):
    df = pd.read_csv(csv)
    folds = sorted(df.held_generator.unique())
    finals, earlys, ps, py = [], [], [], []
    for h in folds:
        M, y, v = fold_final(df, h)
        finals.append(SM.roc_auc(M[:, -1], y))
        earlys.append(SM.roc_auc(M[:, 0], y))
        ps.append(M[:, -1]); py.append(y)
    s = np.concatenate(ps); yy = np.concatenate(py)
    m, lo, hi = SM.bootstrap_ci(SM.roc_auc, s, yy, n_boot=1000, seed=1)
    print(f"[{label}] AUC@N={np.mean(finals):.3f} (fold sd {np.std(finals):.3f}, "
          f"boot95 [{lo:.3f},{hi:.3f}]); sAUC@t1={np.mean(earlys):.3f}")
    return np.mean(finals), (lo, hi), np.mean(earlys)


def anytime_curve(csv, label, cap=16):
    df = pd.read_csv(csv)
    folds = sorted(df.held_generator.unique())
    T = cap
    curves = []
    for h in folds:
        M, y, v = fold_final(df, h)
        t = min(M.shape[1], T)
        aucs = [SM.roc_auc(M[:, i], y) for i in range(t)]
        aucs += [aucs[-1]] * (T - t)
        curves.append(aucs)
    mean = np.nanmean(curves, axis=0)
    print(f"[{label} anytime AUC(t) t=1..{T}] " +
          " ".join(f"{v:.3f}" for v in mean))
    coords = " ".join(f"({i+1},{v:.4f})" for i, v in enumerate(mean))
    print(f"  pgfcoords: {coords}")
    return mean


def cascade_fused(stage1_csv, stage2_csv, label, alpha=0.05):
    s1 = pd.read_csv(stage1_csv); s2 = pd.read_csv(stage2_csv)
    folds = sorted(set(s1.held_generator) & set(s2.held_generator))
    pts = {}   # width -> (defer, macs, acc, fused_auc)
    C1, C2 = 1e5, 1.76e10
    widths = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.90]
    for w in widths:
        defers, accs, macs_l, fs, fy = [], [], [], [], []
        for h in folds:
            M1, y1, v1 = fold_final(s1, h)
            M2, y2, v2 = fold_final(s2, h)
            cal = s1[(s1.held_generator == h) & (s1.split == "calib")]
            Mc, _, _, _ = score_matrix(cal)
            tau = float(np.quantile(Mc[:, -1], 1 - alpha))
            s = M1[:, -1]
            p2 = dict(zip(v2, M2[:, -1]))
            s2v = np.array([p2.get(v, 0.0) for v in v1])
            deferred = (s >= tau - w) & (s < tau)
            pred1 = (s >= tau).astype(int)
            pred = pred1.copy(); pred[deferred] = (s2v[deferred] >= 0.5).astype(int)
            defers.append(deferred.mean())
            accs.append(1 - SM.error_rate(pred, y1))
            macs_l.append(C1 + deferred.mean() * C2)
            # fused SCORE for AUC: stage-2 prob where deferred, else stage-1
            fused = s.copy(); fused[deferred] = s2v[deferred]
            fs.append(fused); fy.append(y1)
        s = np.concatenate(fs); yy = np.concatenate(fy)
        pts[w] = (np.mean(defers), np.mean(macs_l), np.mean(accs),
                  SM.roc_auc(s, yy))
    print(f"[{label} cascade]  w   defer   MACs        acc     fusedAUC")
    for w, (d, mc, a, au) in pts.items():
        print(f"    {w:.2f}  {d:.3f}  {mc:.3e}  {a:.4f}  {au:.4f}")
    # Pareto coords (compute, accuracy) up to the knee
    coords = " ".join(f"({mc:.3e},{a:.4f})" for w, (d, mc, a, au) in pts.items())
    print(f"  pareto pgfcoords: {coords}")
    return pts


def main(results="results"):
    """Recompute every number the paper cites, from the stage outputs."""
    r = lambda p: f"{results}/{p}"
    print("=" * 70)
    cell_auc(r("stream-scores/sh_0000.csv"), "codec matched")
    cell_auc(r("pixel-scores/sh_0000.csv"), "pixel matched")
    cell_auc(r("stream-scores-aigvd/sh_0000.csv"), "AIGVDBench")
    cell_auc(r("stream-scores-longform/sh_0000.csv"), "long-form")
    print("=" * 70)
    anytime_curve(r("stream-scores/sh_0000.csv"), "codec")
    anytime_curve(r("pixel-scores/sh_0000.csv"), "pixel")
    anytime_curve(r("stream-scores-longform/sh_0000.csv"), "longform")
    print("=" * 70)
    cascade_fused(r("stream-scores/sh_0000.csv"),
                  r("pixel-scores/sh_0000.csv"), "codec->pixel")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results",
                    help="directory holding the per-stage outputs")
    main(**vars(ap.parse_args()))
