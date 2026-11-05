"""Summarize saved IOI causal CSVs (no model needed).

  python greaterthan/ioi_summarize.py                       # all results_ioi_pos/ioi_*_17491__*.csv
  python greaterthan/ioi_summarize.py path1.csv path2.csv
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats


def summarize(path):
    df = pd.read_csv(path)
    keys = df["is_keys" if "is_keys" in df.columns else "keys"].astype(bool).values
    n_rd = len([c for c in df.columns if c.startswith("ok_rd")])
    print(f"\n===== {Path(path).stem} =====")
    print(f"baseline first-token accuracy {df.ok_base.mean():.3f} ({int(df.ok_base.sum())}/{len(df)})")
    for name, m in {"all": np.ones(len(df), bool), "keys prompts": keys, "other prompts": ~keys}.items():
        d = df[m]
        fx = int(((d.ok_base == 0) & (d.ok_int == 1)).sum()); br = int(((d.ok_base == 1) & (d.ok_int == 0)).sum())
        p = sstats.binomtest(fx, fx + br, 0.5).pvalue if fx + br else float("nan")
        print(f"{name:14s} n={len(d):3d} | accuracy {d.ok_base.mean():.3f} -> {d.ok_int.mean():.3f} | "
              f"logit_diff {d.ld_base.mean():+.2f} -> {d.ld_int.mean():+.2f} | gained {fx}, lost {br}, paired p = {p:.2g}")
        if n_rd:
            acc = [d[f"ok_rd{j}"].mean() for j in range(n_rd)]; ld = [d[f"ld_rd{j}"].mean() for j in range(n_rd)]
            print(f"{'':14s}   size-matched random direction: accuracy {np.mean(acc):.3f} "
                  f"(range {min(acc):.3f}..{max(acc):.3f}), logit_diff {np.mean(ld):+.2f} (range {min(ld):+.2f}..{max(ld):+.2f})")


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(glob.glob("results_ioi_pos/ioi_*_17491__*.csv"))
    if not paths:
        raise SystemExit("no CSVs found in results_ioi_pos/")
    for p in paths:
        summarize(p)
