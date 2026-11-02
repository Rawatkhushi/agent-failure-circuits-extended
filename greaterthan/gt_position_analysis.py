"""Per-position + confound analysis for the Greater-Than SAE run.

Reads the output folder of run_gt.py plus feature_stats_last3.csv (from src/analyze.py).

1. ROLE SUMMARY: for every token role (bos, noun, start_cc, start_yy, end_cc, ...), the
   per-feature failure-vs-success statistics (Cohen's d, Welch t, Holm) using the SAE
   activation at THAT position only, instead of the mean of the last 3 tokens.
2. TOP-FEATURE x ROLE: for the top pooled features, d at each role -> which position
   carries each feature's signal.
3. CONFOUND CHECK: for the top pooled features, d within century-12 prompts, d within
   the other centuries, and correlation with the century-12 indicator and with start_yy.
4. CONDITIONAL ANALYSIS: the pooled analysis repeated without century 12 (the analogue
   of the IOI keys-free analysis), plus overlap with the full-corpus top list.

Usage (from repo root):
  python greaterthan/gt_position_analysis.py --dir results_gt_hard_sae
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

ROLE_ORDER = ["bos", "the0", "noun_pre", "noun", "lasted", "from", "the1", "year1",
              "start_cc", "start_yy", "to", "the2", "year2", "end_cc"]


def holm(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, min(1.0, p[idx] * (n - rank)))
        adj[idx] = run
    return adj


def feature_stats(X: np.ndarray, fail: np.ndarray):
    """Same conventions as src/analyze.py (Welch t, df=min(n)-1, pooled-SD Cohen's d)."""
    nf, ns = int(fail.sum()), int((~fail).sum())
    if nf < 2 or ns < 2:
        return None
    A, B = X[fail].astype(np.float64), X[~fail].astype(np.float64)
    mf, ms = A.mean(0), B.mean(0)
    vf, vs = A.var(0, ddof=1), B.var(0, ddof=1)
    se = np.sqrt(vf / nf + vs / ns)
    t = np.divide(mf - ms, se, out=np.zeros_like(se), where=se > 0)
    p = 2 * stats.t.sf(np.abs(t), df=min(nf, ns) - 1)
    pooled = ((nf - 1) * vf + (ns - 1) * vs) / (nf + ns - 2)
    d = np.divide(mf - ms, np.sqrt(pooled), out=np.zeros_like(pooled), where=pooled > 0)
    return d, p


def summarize(d, p):
    ph = holm(p)
    top = int(np.argmax(np.abs(d)))
    return {"n_sig_holm": int((ph < 0.05).sum()), "n_large_d": int((np.abs(d) > 0.8).sum()),
            "max_abs_d": float(np.abs(d).max()), "top_feature": top, "top_d": float(d[top])}


def main(d_dir: str, top_n: int, out_dir: str | None) -> None:
    d_dir = Path(d_dir)
    out = Path(out_dir) if out_dir else d_dir / "position_analysis"
    out.mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(d_dir / "predictions.csv").sort_values("id").reset_index(drop=True)
    pos = pd.read_csv(d_dir / "positions.csv")
    A = sp.load_npz(d_dir / "acts_pos.npz").tocsr()
    pooled = np.load(d_dir / "acts_last3.npy")
    fs = pd.read_csv(d_dir / "feature_stats_last3.csv")

    fail = preds.success.values == 0
    c12 = preds.century.values == 12
    yy = preds.start_yy.values.astype(float)
    n = len(preds)
    print(f"N={n}, failures={fail.sum()}, century-12 prompts={c12.sum()}, "
          f"failures in c12={(fail & c12).sum()}, failures outside c12={(fail & ~c12).sum()}")

    roles = [r for r in ROLE_ORDER if r in set(pos.role)]
    role_rows = {}
    for r in roles:
        sub = pos[pos.role == r].sort_values("id")
        if len(sub) != n or not (sub.id.values == preds.id.values).all():
            print(f"  (skipping role '{r}': not present exactly once per prompt)")
            continue
        role_rows[r] = sub.row.values

    top_feats = fs.head(top_n).feature.values.astype(int)

    # ---- 1 & 2: role summary (all prompts, and without century 12) ----
    summ, D_top = [], {}
    for subset_name, mask in [("all", np.ones(n, bool)), ("no_c12", ~c12)]:
        for r, rows in role_rows.items():
            X = A[rows].toarray()
            res = feature_stats(X[mask], fail[mask])
            if res is None:
                continue
            d, p = res
            s = summarize(d, p); s.update(role=r, subset=subset_name)
            summ.append(s)
            if subset_name == "all":
                D_top[r] = d[top_feats]
    role_summary = pd.DataFrame(summ)[["subset", "role", "n_sig_holm", "n_large_d", "max_abs_d", "top_feature", "top_d"]]
    role_summary.to_csv(out / "role_summary.csv", index=False)
    print("\n=== 1. Per-position discrimination (failure vs success), by token role ===")
    print(role_summary.round(2).to_string(index=False))

    tf = pd.DataFrame(D_top, index=top_feats)
    tf["peak_role"] = tf.abs().idxmax(axis=1)
    tf.to_csv(out / "top_feature_role_d.csv")
    print(f"\n=== 2. Cohen's d of the top-{top_n} pooled features at each token role ===")
    print(tf.round(2).to_string())

    # ---- 3: confound check on the pooled (last-3) matrix ----
    rows3 = []
    for f in top_feats:
        col = pooled[:, f].astype(np.float64)
        r_all = feature_stats(col[:, None], fail)
        r_n12 = feature_stats(col[~c12][:, None], fail[~c12])
        r_12 = feature_stats(col[c12][:, None], fail[c12])
        rows3.append({
            "feature": f,
            "d_all": r_all[0][0] if r_all else np.nan,
            "d_no_c12": r_n12[0][0] if r_n12 else np.nan,
            "d_c12_only": r_12[0][0] if r_12 else np.nan,
            "corr_c12": np.corrcoef(col, c12.astype(float))[0, 1] if col.std() > 0 else np.nan,
            "corr_start_yy": np.corrcoef(col, yy)[0, 1] if col.std() > 0 else np.nan,
            "mean_act_c12": col[c12].mean(), "mean_act_other": col[~c12].mean(),
        })
    conf = pd.DataFrame(rows3)
    conf.to_csv(out / "confound_top_features.csv", index=False)
    print("\n=== 3. Confound check on top pooled features ===")
    print("(d_no_c12 / d_c12_only: effect of failure WITHIN a century group; corr_*: link to the confounds)")
    print(conf.round(2).to_string(index=False))

    # ---- 4: conditional analysis without century 12 (pooled window) ----
    res = feature_stats(pooled[~c12], fail[~c12])
    d_nc, p_nc = res
    s = summarize(d_nc, p_nc)
    order = np.argsort(-np.abs(d_nc))[:20]
    overlap = len(set(order) & set(fs.head(20).feature.astype(int)))
    print("\n=== 4. Conditional analysis: pooled window, century 12 removed ===")
    print(f"prompts={int((~c12).sum())}, failures={int(fail[~c12].sum())}")
    print(f"Holm-significant: {s['n_sig_holm']}, |d|>0.8: {s['n_large_d']}, max |d| = {s['max_abs_d']:.2f}")
    print("top 5 features (d):", [(int(f), round(float(d_nc[f]), 2)) for f in order[:5]])
    print(f"overlap between this top-20 and the full-corpus top-20: {overlap}/20")

    # ---- figures ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        for k, name in enumerate(["all", "no_c12"]):
            sub = role_summary[role_summary.subset == name]
            ax[k].bar(sub.role, sub.max_abs_d, color="steelblue")
            ax[k].set_title(f"Max |d| by token role ({'all prompts' if name == 'all' else 'century 12 removed'})")
            ax[k].tick_params(axis="x", rotation=60)
            ax[k].axhline(0.8, color="firebrick", ls="--", lw=0.8)
        fig.tight_layout(); fig.savefig(out / "role_max_d.png", dpi=150)
        fig, ax = plt.subplots(figsize=(8, 0.4 * len(top_feats) + 1.5))
        M = tf.drop(columns="peak_role").values.astype(float)
        im = ax.imshow(M, cmap="coolwarm", vmin=-3, vmax=3, aspect="auto")
        ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(tf.columns[:-1], rotation=60)
        ax.set_yticks(range(len(top_feats))); ax.set_yticklabels(top_feats)
        ax.set_title("Cohen's d of top pooled features, per token role")
        fig.colorbar(im); fig.tight_layout(); fig.savefig(out / "top_feature_role_heatmap.png", dpi=150)
        print(f"\nfigures + CSVs written to {out}")
    except ImportError:
        print("(matplotlib missing; skipped figures)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results_gt_hard_sae")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    main(a.dir, a.top_n, a.out)
