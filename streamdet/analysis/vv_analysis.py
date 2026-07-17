"""VideoVeritas baseline cell: standalone accuracy on the VLM subsample.

Mirrors vlm_analysis.py so the numbers are population-comparable: same
subsample clips, same balanced-accuracy convention, bootstrap CIs computed the
same way. VideoVeritas is an offline full-clip detector (16 sampled frames
through Qwen3-VL-8B), so it contributes a single full-prefix point on the
compute-accuracy plane, not a cascade stage.

Run: python Code/streamdet/vv_analysis.py --vv results/vv-scores/merged.csv \
        --vlm 'results/vlm-scores/sh_*.csv' --out <csv>
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import streamdet  # noqa: F401
import streamdet_metrics as SM


def bal_acc(pred, y):
    pred = np.asarray(pred).astype(int); y = np.asarray(y).astype(int)
    tpr = (pred[y == 1] == 1).mean() if (y == 1).any() else 0.0
    tnr = (pred[y == 0] == 0).mean() if (y == 0).any() else 0.0
    return float(0.5 * (tpr + tnr))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vv", required=True)
    ap.add_argument("--vlm", default=None,
                    help="VLM shards; verifies the populations coincide")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from e1_readout import read_table
    vv = read_table(args.vv)
    vv["is_real"] = vv["is_real"].astype(int)
    vv = vv.drop_duplicates("video_id").set_index("video_id")
    y = (vv["is_real"] == 0).astype(int).to_numpy()
    s = vv["score"].to_numpy()
    pred = (vv["verdict"].astype(str).str.lower() == "fake").astype(int).to_numpy()
    print(f"vv_analysis v1 clips={len(vv)} real={int((y==0).sum())} "
          f"gen={int((y==1).sum())} mean_latency_s={vv.latency_s.mean():.1f}",
          flush=True)

    auc = SM.roc_auc(s, y)
    _, alo, ahi = SM.bootstrap_ci(SM.roc_auc, s, y, n_boot=1000, seed=1)
    ba = bal_acc(pred, y)
    rng = np.random.default_rng(1)
    idx = np.arange(len(y))
    bs = [bal_acc(pred[i], y[i])
          for i in (rng.choice(idx, len(idx)) for _ in range(1000))]
    blo, bhi = np.percentile(bs, [2.5, 97.5])
    acc05 = float(((s >= 0.5).astype(int) == y).mean())
    print(f"[VideoVeritas] AUC={auc:.3f} boot95[{alo:.3f},{ahi:.3f}] "
          f"bal_acc={ba:.3f} boot95[{blo:.3f},{bhi:.3f}] acc@0.5={acc05:.3f} "
          f"mean_latency_s={vv.latency_s.mean():.1f} "
          f"median_latency_s={vv.latency_s.median():.1f}", flush=True)

    if args.vlm:
        vlm = read_table(args.vlm).drop_duplicates("video_id")
        both = vv.index.intersection(vlm.video_id)
        print(f"[population] vv={len(vv)} vlm={len(vlm)} common={len(both)}",
              flush=True)

    per = (vv.assign(y=y, ok=(pred == y))
             .groupby("generator")
             .agg(n=("ok", "size"), acc=("ok", "mean"),
                  mean_score=("score", "mean")))
    for g, r in per.iterrows():
        print(f"  gen={g} n={int(r.n)} acc={r.acc:.3f} "
              f"mean_score={r.mean_score:.3f}", flush=True)

    out = per.reset_index()
    out["auc"] = auc; out["auc_lo"] = alo; out["auc_hi"] = ahi
    out["bal_acc"] = ba; out["bal_lo"] = blo; out["bal_hi"] = bhi
    out["mean_latency_s"] = vv.latency_s.mean()
    out.to_csv(args.out, index=False)
    print(f"result auc={auc:.3f} bal_acc={ba:.3f} out={args.out}", flush=True)


if __name__ == "__main__":
    main()
