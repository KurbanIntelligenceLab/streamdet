"""Pick real exemplar clips for the Figure-1 redesign, from the actual logs.

Finds, in the held=cogvideo LOGO fold: (a) a generated clip whose running max
crosses tau at GOP 2-3 with a visible rise, (b) a real clip staying low over
>=8 GOPs, (c) a clip whose final lands in the deferral band. Prints the fold's
end-calibrated tau and each candidate's per-GOP running max.
"""
import sys

import numpy as np
import pandas as pd

sys.path[:0] = ["."]  # run from repo root; needs the streamdet package


def main():
    df = pd.read_csv("results/stream-scores/sh_0000.csv")
    h = "cogvideo"
    te = df[(df.held_generator == h) & (df.split == "test")]
    ca = df[(df.held_generator == h) & (df.split == "calib")]
    piv = ca.pivot_table(index="video_id", columns="gop_idx",
                         values="score").sort_index(axis=1)
    M = np.maximum.accumulate(
        np.where(np.isnan(piv.to_numpy()), -np.inf, piv.to_numpy()), axis=1)
    finals = pd.DataFrame(M).ffill(axis=1).to_numpy()[:, -1]
    tau = float(np.quantile(finals, 0.95))
    w = 0.10
    print(f"tau={tau:.4f} band=[{tau-w:.4f},{tau:.4f})")

    t = te.pivot_table(index="video_id", columns="gop_idx",
                       values="score").sort_index(axis=1)
    lab = te.drop_duplicates("video_id").set_index("video_id").loc[t.index]
    raw = t.to_numpy()
    run = np.maximum.accumulate(np.where(np.isnan(raw), -np.inf, raw), axis=1)
    run = pd.DataFrame(run, index=t.index).replace(-np.inf, np.nan).ffill(axis=1)

    gen = lab.is_real == 0
    real = lab.is_real == 1
    cand = []
    for v in t.index[gen]:
        r = run.loc[v].dropna().to_numpy()
        if len(r) < 6:
            continue
        cross = int(np.argmax(r >= tau)) if (r >= tau).any() else -1
        if cross in (1, 2) and r[0] < tau - 0.15:
            cand.append((v, cross, r[:8]))
    print("early-exit candidates:", len(cand))
    for v, c, r in cand[:6]:
        print(f"  GEN cross@{c} {np.round(r[:6],3)} :: {v[:64]}")

    rc = []
    for v in t.index[real]:
        r = run.loc[v].dropna().to_numpy()
        if len(r) >= 8 and r[-1] < tau - w - 0.15:
            rc.append((v, r[:8]))
    print("low reals:", len(rc))
    for v, r in rc[:4]:
        print(f"  REAL {np.round(r[:6],3)} :: {v[:64]}")

    uc = []
    for v in t.index:
        r = run.loc[v].dropna().to_numpy()
        if len(r) >= 6 and (tau - w) <= r[-1] < tau and not (r >= tau).any():
            uc.append((v, bool(lab.loc[v, "is_real"] == 0), r[:8]))
    print("deferred:", len(uc))
    for v, g, r in uc[:6]:
        print(f"  DEF gen={g} {np.round(r[:6],3)} :: {v[:64]}")


if __name__ == "__main__":
    main()
