"""Streaming analysis: per-GOP scores -> anytime curves, gate calibration, FPR.

Consumes streaming_scores.py output (long CSV of per-GOP scores per LOGO fold)
and produces, per fold and pooled across folds:
  * anytime AUC(t) of the running max M_t at each prefix t          (E2)
  * sAUC(B) at latency budgets (prefix counts)                      (E2)
  * recall@FPR(t) with the single end-calibrated threshold          (E5)
  * stopping-time FPR: end-calibrated vs per-prefix foil            (E5/E7, Prop 2)
  * decision-latency distribution under the confidence gate         (E5)

Clips have variable GOP counts, so the (clip x prefix) matrix is built by
carrying each clip's running max forward past its last GOP (a finished stream's
aggregate is frozen; this is exactly M_{min(t, T_clip)}).

The threshold tau is calibrated per fold on the calib split (held-out real
clips' final running max, (1-alpha) quantile) — never on test.

Run: python Code/streamdet/analyze_streaming.py \
        --scores <scores.csv> --out-prefix $OUT_DIR/stage1 [--alpha 0.05]
Writes <prefix>_anytime.csv, <prefix>_gate.csv, <prefix>_latency_hist.csv and
prints the headline numbers.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import streamdet  # noqa: F401
from streamdet import metrics as SM


def score_matrix(df: pd.DataFrame):
    """Long (video_id, gop_idx, score) -> (M, labels, vids, T_clip).

    M is the (N, T_max) RUNNING-MAX matrix, forward-filled past each clip's
    final GOP. labels: 1 = generated. T_clip: per-clip true GOP count.
    """
    piv = df.pivot_table(index="video_id", columns="gop_idx", values="score")
    piv = piv.sort_index(axis=1)
    raw = piv.to_numpy(dtype=float)
    lab = (df.drop_duplicates("video_id").set_index("video_id")
           .loc[piv.index, "is_real"] == 0).astype(int).to_numpy()
    t_clip = (~np.isnan(raw)).sum(axis=1)
    # running max ignoring NaN, then forward-fill the frozen aggregate
    M = np.where(np.isnan(raw), -np.inf, raw)
    M = np.maximum.accumulate(M, axis=1)
    M[np.isinf(M)] = np.nan  # leading NaN (shouldn't happen: gop 0 always scored)
    M = pd.DataFrame(M).ffill(axis=1).to_numpy()
    return M, lab, piv.index.to_numpy(), t_clip


def analyze_fold(te: pd.DataFrame, cal: pd.DataFrame, alpha: float,
                 max_prefix: int = 64):
    M, y, vids, t_clip = score_matrix(te)
    Mc, _, _, _ = score_matrix(cal)
    T = M.shape[1]

    tau = float(np.quantile(Mc[:, -1], 1.0 - alpha))       # end-calibrated (Prop 2)
    # per-prefix foil thresholds, aligned to the TEST horizon: calib clips can
    # stream longer/shorter than test clips, so slice or edge-pad (a frozen
    # running max keeps its last quantile past the calib horizon)
    taus_pp = np.quantile(Mc, 1.0 - alpha, axis=0)
    if len(taus_pp) >= T:
        taus_pp = taus_pp[:T]
    else:
        taus_pp = np.concatenate([taus_pp,
                                  np.full(T - len(taus_pp), taus_pp[-1])])

    T_curve = min(T, max_prefix)                            # curve horizon only
    auc_t = np.array([SM.roc_auc(M[:, t], y) for t in range(T_curve)])
    flag = M >= tau
    tpr_t = flag[y == 1].mean(axis=0)
    fpr_t = flag[y == 0].mean(axis=0)
    st_fpr = float((M[y == 0] >= tau).any(axis=1).mean())
    st_fpr_pp = float((M[y == 0] >= taus_pp[None, :]).any(axis=1).mean())
    st_tpr = float((M[y == 1] >= tau).any(axis=1).mean())

    # decision latency: first prefix where M_t >= tau (else censored at T_clip)
    crossed = M >= tau
    first = np.where(crossed.any(axis=1), crossed.argmax(axis=1), -1)
    lat = pd.DataFrame({"video_id": vids, "label": y, "t_clip": t_clip,
                        "decided_at": first})

    anytime = pd.DataFrame({"prefix_gops": np.arange(1, T_curve + 1),
                            "auc": auc_t, "tpr": tpr_t[:T_curve],
                            "fpr": fpr_t[:T_curve]})
    gate = {"tau": tau, "alpha": alpha,
            "stopping_fpr_end_calibrated": st_fpr,
            "stopping_fpr_perprefix_foil": st_fpr_pp,
            "stopping_tpr": st_tpr,
            "auc_final": float(SM.roc_auc(M[:, -1], y)),
            "n_test": int(len(y)), "n_gen": int(y.sum())}
    return anytime, gate, lat


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--max-prefix", type=int, default=64,
                    help="horizon for the anytime AUC curve (gate metrics "
                         "always use the full horizon)")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.scores)
    folds = sorted(df["held_generator"].unique())
    print(f"analyze_streaming v1 alpha={args.alpha} folds={folds} "
          f"rows={len(df)}", flush=True)

    all_any, all_gate, all_lat = [], [], []
    for held in folds:
        d = df[df["held_generator"] == held]
        anytime, gate, lat = analyze_fold(
            d[d["split"] == "test"], d[d["split"] == "calib"], args.alpha,
            max_prefix=args.max_prefix)
        anytime["held_generator"] = held
        lat["held_generator"] = held
        gate["held_generator"] = held
        all_any.append(anytime); all_gate.append(gate); all_lat.append(lat)
        print(f"fold held={held} auc_final={gate['auc_final']:.4f} tau={gate['tau']:.4f} "
              f"stFPR={gate['stopping_fpr_end_calibrated']:.4f} "
              f"stFPR_foil={gate['stopping_fpr_perprefix_foil']:.4f} "
              f"stTPR={gate['stopping_tpr']:.4f}", flush=True)

    anyt = pd.concat(all_any, ignore_index=True)
    gate = pd.DataFrame(all_gate)
    lat = pd.concat(all_lat, ignore_index=True)
    anyt.to_csv(f"{args.out_prefix}_anytime.csv", index=False)
    gate.to_csv(f"{args.out_prefix}_gate.csv", index=False)
    lat.to_csv(f"{args.out_prefix}_latency.csv", index=False)

    mean_any = anyt.groupby("prefix_gops")["auc"].mean()
    print("pooled anytime AUC(t): " +
          " ".join(f"t{t}={v:.3f}" for t, v in mean_any.items() if t <= 8), flush=True)
    print(f"result logo_ood_final={gate['auc_final'].mean():.4f} "
          f"stFPR={gate['stopping_fpr_end_calibrated'].mean():.4f} "
          f"stFPR_foil={gate['stopping_fpr_perprefix_foil'].mean():.4f} "
          f"out={args.out_prefix}_*.csv", flush=True)


if __name__ == "__main__":
    main()
