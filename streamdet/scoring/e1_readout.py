"""E1 readout: run vidaudit's audited LOGO + RvR on a 13-d clip feature table.

The positive control for the streaming paper: the reference toolkit's published
leave-one-generator-out numbers on the matched 27k cell (OOD AUC 0.832, RvR
0.643) must reproduce before any streaming result is trusted. Uses vidaudit.audit.protocol.run_logo/run_rvr
verbatim — same seed, same split path, same readout.

Run (single task):
    python Code/streamdet/e1_readout.py --features <table.csv> \
        [--subset <subset.csv>] --out $OUT
"""
from __future__ import annotations

import argparse
import glob

import pandas as pd


def read_table(path_or_glob: str) -> pd.DataFrame:
    """Read one CSV, or concat a shard glob (results/<job>/sh_*.csv).

    Globs are only safe when the producing job is COMPLETE — run the consumer
    only once the producing stage has finished, so every shard
    landed before this executes.
    """
    paths = sorted(glob.glob(path_or_glob)) if any(c in path_or_glob for c in "*?[") \
        else [path_or_glob]
    if not paths:
        raise FileNotFoundError(f"no files match {path_or_glob}")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

import streamdet  # noqa: F401
from vidaudit.audit.protocol import run_logo, run_rvr
from vidaudit.features import FEATURE_NAMES


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--subset", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    df = read_table(args.features)
    n0 = len(df)
    if args.subset:
        sub = pd.read_csv(args.subset)
        keys = set(zip(sub["video_id"].astype(str), sub["generator"].astype(str)))
        df = df[[(v, g) in keys for v, g in
                 zip(df["video_id"].astype(str), df["generator"].astype(str))]]
    df = df.reset_index(drop=True)
    print(f"e1_readout v1 features={args.features} subset={args.subset} "
          f"rows={len(df)}/{n0} gens={sorted(df.loc[df.is_real==0,'generator'].unique())}",
          flush=True)

    res = run_logo(df, list(FEATURE_NAMES))
    rvr = run_rvr(df, list(FEATURE_NAMES))

    rows = [{"held_generator": g, **{k: v for k, v in d.items() if k != "n_real_test"},
             "n_real_test": d["n_real_test"]} for g, d in res["per_generator"].items()]
    out = pd.DataFrame(rows)
    out["logo_id_mean"] = res["logo_id"]
    out["logo_ood_mean"] = res["logo_ood"]
    out["rvr_auc"] = rvr["auc"] if rvr else float("nan")
    out.to_csv(args.out, index=False)

    for g, d in res["per_generator"].items():
        print(f"fold held={g} ID={d['ID_auc']:.4f} OOD={d['OOD_auc']:.4f}", flush=True)
    print(f"result logo_id={res['logo_id']:.4f} logo_ood={res['logo_ood']:.4f} "
          f"rvr={rvr['auc'] if rvr else float('nan'):.4f} out={args.out}", flush=True)


if __name__ == "__main__":
    main()
