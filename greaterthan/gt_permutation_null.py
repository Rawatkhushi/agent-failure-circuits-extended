"""Permutation null for the failure-vs-success feature statistics (offline).

Why: with few failures, sparse SAE features that fire on 1-2 prompts get huge Cohen's d, and a
t-test (df = n_fail-1) is not calibrated for them. We recompute d under shuffled failure labels:
  FULL null   : labels shuffled across all prompts           -> how many features pass |d|>0.8 by chance
  STRATIFIED  : labels shuffled WITHIN strata (default century x start_yy), preserving each stratum's
                failure count -> what the known confounds alone can produce. A feature that merely
                tracks century or start year looks 'significant' under FULL but not under STRATIFIED.
Per-feature p-values use the max-|d| statistic (family-wise): p_f = P(max_g |d_g|^null >= |d_f|).

Scopes: the paper's last-3 pooling and every single token role. Subsets: all prompts, century 12 removed.

  python greaterthan/gt_permutation_null.py --dir results_gt_hard_sae --n-perm 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROLES = ["noun", "lasted", "from", "the1", "year1", "start_cc", "start_yy", "to", "the2", "year2", "end_cc"]


class Stat:
    def __init__(self, X):
        self.X = X.astype(np.float64)
        self.X2 = self.X ** 2
        self.st, self.st2, self.n = self.X.sum(0), self.X2.sum(0), len(X)

    def d(self, w):
        n, X, X2 = self.n, self.X, self.X2
        nf = w.sum(); ns = n - nf
        sf, sf2 = w @ X, w @ X2
        ss, ss2 = self.st - sf, self.st2 - sf2
        mf, ms = sf / nf, ss / ns
        vf = np.maximum((sf2 - nf * mf ** 2) / (nf - 1), 0)
        vs = np.maximum((ss2 - ns * ms ** 2) / (ns - 1), 0)
        pooled = ((nf - 1) * vf + (ns - 1) * vs) / (n - 2)
        return np.divide(mf - ms, np.sqrt(pooled), out=np.zeros_like(pooled), where=pooled > 1e-12)


def run_scope(X, fail, strata_ids, n_perm, rng, thr=0.8):
    """Returns observed d, and null summaries under FULL and STRATIFIED shuffles."""
    S = Stat(X)
    w = fail.astype(np.float64)
    d_obs = S.d(w)
    groups = [np.where(strata_ids == g)[0] for g in np.unique(strata_ids)]
    out = {"d_obs": d_obs}
    for name in ["full", "strat"]:
        n_big, maxd = [], []
        for _ in range(n_perm):
            if name == "full":
                wp = rng.permutation(w)
            else:
                wp = w.copy()
                for idx in groups:
                    if len(idx) > 1:
                        wp[idx] = rng.permutation(w[idx])
            dn = np.abs(S.d(wp))
            n_big.append(int((dn > thr).sum())); maxd.append(float(dn.max()))
        out[name] = {"n_big": np.array(n_big), "max": np.array(maxd)}
    return out


def main(d_dir, n_perm, strata_cols, features, seed):
    d = Path(d_dir)
    preds = pd.read_csv(d / "predictions.csv").sort_values("id").reset_index(drop=True)
    pos = pd.read_csv(d / "positions.csv")
    A = sp.load_npz(d / "acts_pos.npz").tocsr()
    pooled = np.load(d / "acts_last3.npy")
    fail = (preds.success.values == 0)
    c12 = preds.century.values == 12
    strata_all = preds[strata_cols].astype(str).agg("|".join, axis=1).values
    rng = np.random.default_rng(seed)
    print(f"N={len(preds)}, failures={fail.sum()} (c12: {(fail & c12).sum()}), strata = {strata_cols}, "
          f"{len(np.unique(strata_all))} strata, {n_perm} permutations per null")

    scopes = {"pooled_last3": pooled}
    for r in ROLES:
        sub = pos[pos.role == r].sort_values("id")
        if len(sub) == len(preds):
            scopes[r] = A[sub.row.values].toarray()

    summary, feat_rows, saved = [], [], {}
    for scope, X in scopes.items():
        for sname, mask in [("all", np.ones(len(preds), bool)), ("no_c12", ~c12)]:
            Xs, fs, ss = X[mask], fail[mask], strata_all[mask]
            if fs.sum() < 3:
                continue
            res = run_scope(Xs, fs, ss, n_perm, rng)
            d_obs = res["d_obs"]
            obs_big = int((np.abs(d_obs) > 0.8).sum())
            row = {"scope": scope, "subset": sname, "n_fail": int(fs.sum()), "obs_n_|d|>.8": obs_big,
                   "obs_max|d|": round(float(np.abs(d_obs).max()), 2)}
            for nm in ["full", "strat"]:
                nb, mx = res[nm]["n_big"], res[nm]["max"]
                row[f"{nm}_null_n_mean"] = round(float(nb.mean()), 1)
                row[f"{nm}_null_n_95"] = int(np.percentile(nb, 95))
                row[f"{nm}_null_max_95"] = round(float(np.percentile(mx, 95)), 2)
                p_f = (1 + (mx[None, :] >= np.abs(d_obs)[:, None]).sum(1)) / (1 + n_perm)
                row[f"n_feat_FWER<.05_{nm}"] = int((p_f < 0.05).sum())
                res[nm]["p"] = p_f
            summary.append(row)
            for f in features:
                feat_rows.append({"scope": scope, "subset": sname, "feature": f, "d": round(float(d_obs[f]), 2),
                                  "p_maxT_full": round(float(res["full"]["p"][f]), 3),
                                  "p_maxT_strat": round(float(res["strat"]["p"][f]), 3)})
            if scope == "pooled_last3":
                saved[sname] = pd.DataFrame({"feature": np.arange(len(d_obs)), "d": d_obs,
                                             "p_maxT_full": res["full"]["p"], "p_maxT_strat": res["strat"]["p"]})
            print(f"  done {scope:13s} {sname}")

    out = d / "position_analysis"; out.mkdir(exist_ok=True)
    S = pd.DataFrame(summary); S.to_csv(out / "permutation_summary.csv", index=False)
    F = pd.DataFrame(feat_rows); F.to_csv(out / "permutation_target_features.csv", index=False)
    for k, v in saved.items():
        v.to_csv(out / f"permutation_pooled_{k}.csv", index=False)

    pd.set_option("display.width", 250)
    print("\n=== A. How many features pass |d|>0.8: observed vs chance ===")
    print("(full = labels shuffled everywhere; strat = shuffled within strata => what the confounds alone give)")
    print(S[["scope", "subset", "n_fail", "obs_n_|d|>.8", "full_null_n_mean", "full_null_n_95",
             "strat_null_n_mean", "strat_null_n_95"]].to_string(index=False))
    print("\n=== B. Max |d|: observed vs null 95th percentile; features with family-wise p<.05 ===")
    print(S[["scope", "subset", "obs_max|d|", "full_null_max_95", "strat_null_max_95",
             "n_feat_FWER<.05_full", "n_feat_FWER<.05_strat"]].to_string(index=False))
    print("\n=== C. Target features (d and max-T p-values; strat p<.05 => signal BEYOND the strata) ===")
    show = F[(F.d.abs() >= 0.5)]
    print(show.to_string(index=False) if len(show) else "(none with |d|>=0.5)")
    print(f"\nwrote CSVs to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results_gt_hard_sae")
    p.add_argument("--n-perm", type=int, default=200)
    p.add_argument("--strata", default="century,start_yy", help="comma-separated columns of predictions.csv")
    p.add_argument("--features", type=int, nargs="+", default=[23414, 8350, 7807, 11842, 5868, 17942, 20679])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.dir, a.n_perm, a.strata.split(","), a.features, a.seed)
