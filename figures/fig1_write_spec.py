"""Write fig1_clips.json: exemplar candidates with real running-max
trajectories. Round 2: several candidates per role so the final pick can be
made on visual content; roles real1..N / unc1..N plus the chosen gen."""
import json
import sys

import numpy as np
import pandas as pd

sys.path[:0] = ["."]  # run from repo root; needs the streamdet package

GEN = "00980___a picture of a mountain with a train on it"
N_ALT = 5


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

    t = te.pivot_table(index="video_id", columns="gop_idx",
                       values="score").sort_index(axis=1)
    lab = te.drop_duplicates("video_id").set_index("video_id").loc[t.index]
    raw = t.to_numpy()
    run = np.maximum.accumulate(np.where(np.isnan(raw), -np.inf, raw), axis=1)
    run = pd.DataFrame(run, index=t.index).replace(-np.inf,
                                                   np.nan).ffill(axis=1)
    real = lab.is_real == 1

    clips = [{"video_id": GEN, "role": "gen",
              "running_max": [round(float(x), 4) for x in
                              run.loc[GEN].dropna().to_numpy()[:8]]}]

    # low reals with a gently RISING then flat trajectory (visually legible)
    rc = []
    for v in t.index[real]:
        r = run.loc[v].dropna().to_numpy()
        if len(r) >= 8 and 0.05 < r[-1] < tau - w - 0.25 and r[0] < 0.1:
            rc.append((v, r[:8]))
    for i, (v, r) in enumerate(rc[5:5 + N_ALT]):
        clips.append({"video_id": v, "role": f"real{i+1}",
                      "running_max": [round(float(x), 4) for x in r]})
        print(f"real{i+1}", np.round(r[:5], 3), v[:56])

    # deferred REAL clips with a RISING trajectory into the band
    uc = []
    for v in t.index[real]:
        r = run.loc[v].dropna().to_numpy()
        if (len(r) >= 6 and (tau - w) <= r[-1] < tau
                and not (r >= tau).any() and r[-1] - r[0] > 0.1):
            uc.append((v, r[:8]))
    for i, (v, r) in enumerate(uc[:N_ALT]):
        clips.append({"video_id": v, "role": f"unc{i+1}",
                      "running_max": [round(float(x), 4) for x in r]})
        print(f"unc{i+1}", np.round(r[:5], 3), v[:56])

    spec = {"tau": round(tau, 4), "width": w, "fold": h, "clips": clips}
    with open("figures/fig1_clips.json", "w") as f:
        json.dump(spec, f, indent=1)
    print("wrote fig1_clips.json with", len(clips), "clips")


if __name__ == "__main__":
    main()
