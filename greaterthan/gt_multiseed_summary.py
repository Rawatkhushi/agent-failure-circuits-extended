"""Summarize the multi-seed control: is the Greater-Than picture stable across corpus seeds?

For each results folder: failure rates (overall / century 12 / other), the top pooled features,
the rank and d of the features we care about, and the 23414 ablation effect at end_cc.
Also: overlap of each seed's top-20 with the first folder's top-20, and features in the top-50 of EVERY seed.

  python greaterthan/gt_multiseed_summary.py --dirs results_gt_hard_sae results_gt_seed1 results_gt_seed2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(dirs, feats, abl_file):
    rows, tops50, top20_ref = [], [], None
    for k, d in enumerate(dirs):
        d = Path(d)
        p = pd.read_csv(d / "predictions.csv")
        fs = pd.read_csv(d / "feature_stats_last3.csv")          # sorted by |d| desc
        fail = p.success == 0
        c12 = p.century == 12
        rank = {int(f): i + 1 for i, f in enumerate(fs.feature)}
        dval = dict(zip(fs.feature.astype(int), fs.cohens_d))
        top20 = list(fs.feature.astype(int).head(20))
        if top20_ref is None:
            top20_ref = set(top20)
        tops50.append(set(fs.feature.astype(int).head(50)))
        row = {"folder": d.name, "N": len(p), "fails": int(fail.sum()),
               "fail%": round(100 * fail.mean(), 1), "c12 fail%": round(100 * fail[c12].mean(), 1),
               "other fail%": round(100 * fail[~c12].mean(), 1),
               "n_|d|>.8": int((fs.abs_d > 0.8).sum()),
               "top3": ", ".join(f"{int(f)}({dval[int(f)]:+.2f})" for f in top20[:3]),
               "top20 overlap w/ first": len(top20_ref & set(top20))}
        for f in feats:
            row[f"rank {f}"] = rank.get(f)
            row[f"d {f}"] = round(float(dval.get(f, float('nan'))), 2)
        a = d / abl_file
        if a.exists():
            c = pd.read_csv(a)
            m12, mo = c.century == 12, c.century != 12
            row["c12 fail% base->abl"] = f"{100*c.fail_base[m12].mean():.1f}->{100*c.fail_abl[m12].mean():.1f}"
            row["other fail% base->abl"] = f"{100*c.fail_base[mo].mean():.1f}->{100*c.fail_abl[mo].mean():.1f}"
            row["c12 fixed/broken"] = f"{int(((c.fail_base==1)&(c.fail_abl==0)&m12).sum())}/{int(((c.fail_base==0)&(c.fail_abl==1)&m12).sum())}"
        rows.append(row)
    df = pd.DataFrame(rows).set_index("folder")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    print(df.T.to_string())
    common = set.intersection(*tops50)
    print(f"\nfeatures in the top-50 (by |d|) of EVERY folder: {sorted(common)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--features", type=int, nargs="+", default=[23414, 8350, 7807])
    p.add_argument("--ablation-file", default="causal_ablate_23414__end_cc.csv")
    a = p.parse_args()
    main(a.dirs, a.features, a.ablation_file)
