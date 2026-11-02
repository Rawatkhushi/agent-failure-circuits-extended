"""Control 2: does the SAE basis predict failure better than raw activations, or than the
known confounds alone?  (Offline: reads files written by run_gt.py; needs scikit-learn.)

Predictors compared (failure = 1), repeated stratified 5-fold CV, ROC AUC:
  metadata-only   : century (one-hot) | + start_yy | + noun       <- the known confounds
  raw residual    : 768-d, mean over last 3 tokens
  SAE all features: 24,576-d (optional, slow: --all-sae)
  SAE top-K LEAKY : top-K by |d| chosen on ALL data (what the original repo does -> optimistic)
  SAE top-K NESTED: top-K chosen on each TRAINING fold only (the correct protocol)
Optionally (--per-role): raw residual at each single token role, showing where the
discriminating information lives.

  python greaterthan/gt_raw_baseline.py --dir results_gt_hard_sae --per-role
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

warnings.filterwarnings("ignore")


def abs_d(X, y):
    A, B = X[y == 1].astype(np.float64), X[y == 0].astype(np.float64)
    nf, ns = len(A), len(B)
    vf, vs = A.var(0, ddof=1), B.var(0, ddof=1)
    pooled = ((nf - 1) * vf + (ns - 1) * vs) / (nf + ns - 2)
    return np.abs(np.divide(A.mean(0) - B.mean(0), np.sqrt(pooled),
                            out=np.zeros_like(pooled), where=pooled > 0))


def lr():
    return LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")


def cv_auc(X, y, splits, select_k=None):
    aucs = []
    for tr, te in splits:
        cols = slice(None)
        if select_k is not None:
            cols = np.argsort(-abs_d(X[tr], y[tr]))[:select_k]
        m = lr().fit(X[tr][:, cols], y[tr])
        aucs.append(roc_auc_score(y[te], m.decision_function(X[te][:, cols])))
    return float(np.mean(aucs)), float(np.std(aucs))


def main(d_dir, ks, per_role, all_sae):
    d = Path(d_dir)
    preds = pd.read_csv(d / "predictions.csv").sort_values("id").reset_index(drop=True)
    y = (preds.success.values == 0).astype(int)
    acts = np.load(d / "acts_last3.npy")
    raw = np.load(d / "raw_last3.npy")
    print(f"N={len(y)}, failures={y.sum()}")
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0).split(raw, y))

    cent = pd.get_dummies(preds.century, prefix="c").values.astype(float)
    yyz = ((preds.start_yy - preds.start_yy.mean()) / preds.start_yy.std()).values[:, None]
    noun = pd.get_dummies(preds.noun, prefix="n").values.astype(float)
    rows = []

    def add(name, X, k=None):
        m, s = cv_auc(X, y, splits, select_k=k)
        rows.append({"predictor": name, "dim": X.shape[1] if k is None else k, "AUC": round(m, 3), "sd": round(s, 3)})
        print(f"  {name:38s} AUC = {m:.3f} (sd {s:.3f})")

    print("\n-- known confounds only --")
    add("century", cent)
    add("century + start_yy", np.hstack([cent, yyz]))
    add("century + start_yy + noun", np.hstack([cent, yyz, noun]))
    print("\n-- representations (mean of last 3 tokens) --")
    add("raw residual (768)", raw)
    if all_sae:
        add("SAE all features", acts)
    glob_order = np.argsort(-abs_d(acts, y))
    for k in ks:
        m, s = cv_auc(acts[:, glob_order[:k]], y, splits)
        rows.append({"predictor": f"SAE top-{k} LEAKY (original protocol)", "dim": k, "AUC": round(m, 3), "sd": round(s, 3)})
        print(f"  {'SAE top-%d LEAKY (selected on all data)' % k:38s} AUC = {m:.3f} (sd {s:.3f})")
        add(f"SAE top-{k} NESTED (in-fold selection)", acts, k=k)

    if per_role:
        print("\n-- raw residual at a SINGLE token role (where does the information live?) --")
        pos = pd.read_csv(d / "positions.csv")
        raw_pos = np.load(d / "raw_pos.npy")
        for role in ["noun", "lasted", "from", "the1", "year1", "start_cc", "start_yy", "to", "the2", "year2", "end_cc"]:
            sub = pos[pos.role == role].sort_values("id")
            if len(sub) != len(y):
                continue
            add(f"raw @ {role}", raw_pos[sub.row.values])

    out = d / "position_analysis"; out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "baseline_auc.csv", index=False)
    print(f"\nwrote {out / 'baseline_auc.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results_gt_hard_sae")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 10, 50, 100])
    p.add_argument("--per-role", action="store_true")
    p.add_argument("--all-sae", action="store_true")
    a = p.parse_args()
    main(a.dir, a.ks, a.per_role, a.all_sae)
