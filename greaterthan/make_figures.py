"""Build the five figures for the write-up from saved result files (no model needed).

  python greaterthan/make_figures.py                       # defaults: results_gt_grid, results_ioi_pos -> figures_ext/
Missing files are skipped with a message.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, RED, GREY, GREEN = "#3b6ea8", "#c0392b", "#8c8c8c", "#2e8b57"
ROLE_ORDER = ["noun", "lasted", "from", "the1", "year1", "start_cc", "start_yy", "to", "the2", "year2", "end_cc"]


def rd(path):
    p = Path(path)
    if not p.exists():
        print(f"  (missing {p}; skipping what needs it)")
        return None
    return pd.read_csv(p)


def keycol(df):
    return df["is_keys"] if "is_keys" in df.columns else df["keys"]


def rd_mean(d, prefix):
    cols = [c for c in d.columns if c.startswith(prefix)]
    return float(np.mean([d[c].mean() for c in cols])) if cols else np.nan


def fig1_roles(gt, out):
    df = rd(gt / "position_analysis" / "role_summary.csv")
    if df is None:
        return
    a = df[df.subset == "all"].set_index("role").reindex(ROLE_ORDER)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    cols = [RED if r in ("the2", "year2", "end_cc") else BLUE for r in ROLE_ORDER]
    ax[0].bar(ROLE_ORDER, a.n_sig_holm, color=cols)
    ax[0].set_title("Holm-significant features per token role\n(red = the paper's last-3 pooling window)")
    ax[1].bar(ROLE_ORDER, a.max_abs_d, color=cols)
    ax[1].axhline(0.8, color="k", ls="--", lw=0.8)
    ax[1].set_title("Max |Cohen's d| per token role")
    for x in ax:
        x.tick_params(axis="x", rotation=60)
    fig.tight_layout(); fig.savefig(out / "fig1_gt_position_signal.png", dpi=170); plt.close(fig)


def fig2_gt_causal(gt, out):
    ab = rd(gt / "causal_ablate_23414__end_cc.csv"); inj = rd(gt / "inject_23414__end_cc.csv")
    a8 = rd(gt / "causal_ablate_8350__year2.csv"); a7 = rd(gt / "causal_ablate_7807__end_cc.csv")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    # (a) ablations on century 12
    labels, base, abl = [], [], []
    for name, d in [("23414\n@end_cc", ab), ("8350\n@year2", a8), ("7807\n@end_cc", a7)]:
        if d is None:
            continue
        c = d[d.century == 12]
        labels.append(name); base.append(100 * c.fail_base.mean()); abl.append(100 * c.fail_abl.mean())
    x = np.arange(len(labels))
    ax[0].bar(x - 0.2, base, 0.4, color=GREY, label="baseline"); ax[0].bar(x + 0.2, abl, 0.4, color=RED, label="feature ablated")
    if ab is not None:
        rnd = 100 * rd_mean(ab[ab.century == 12], "fail_rd")
        ax[0].scatter([0.2], [rnd], marker="D", color="k", zorder=5, label="same-norm random direction")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels); ax[0].set_ylabel("failure rate, century-12 prompts (%)")
    ax[0].set_title("(a) Ablation: only 23414 matters"); ax[0].set_ylim(0, 100); ax[0].legend(fontsize=8, loc="upper right")
    # (b) dose-response
    if inj is not None:
        scales = [float(c.split("_s")[1]) for c in inj.columns if c.startswith("fail_s")]
        for name, m, col in [("other centuries (injection)", inj.century != 12, RED), ("century 12 (s<1 = partial ablation)", inj.century == 12, BLUE)]:
            d = inj[m]
            ys = [100 * d.fail_base.mean()] + [100 * d[f"fail_s{s}"].mean() for s in scales]
            ax[1].plot([0] + scales, ys, "o-", color=col, label=name)
        rr = 100 * rd_mean(inj[inj.century != 12], "fail_rd")
        ax[1].scatter([1.0], [rr], marker="D", color="k", zorder=5, label="random dir., same norm (other centuries)")
        ax[1].set_xlabel("feature level (x its century-12 activation)"); ax[1].set_ylabel("failure rate (%)")
        ax[1].set_title("(b) Dose-response of 23414 at the final token"); ax[1].legend(fontsize=8)
    # (c) prob_diff by century under ablation
    if ab is not None:
        g = ab.groupby("century")[["pd_base", "pd_abl"]].mean()
        xx = np.arange(len(g))
        ax[2].bar(xx - 0.2, g.pd_base, 0.4, color=GREY, label="baseline"); ax[2].bar(xx + 0.2, g.pd_abl, 0.4, color=RED, label="23414 ablated")
        ax[2].set_xticks(xx); ax[2].set_xticklabels(g.index); ax[2].set_xlabel("century"); ax[2].set_ylabel("mean prob_diff (valid minus invalid year mass)")
        ax[2].set_title("(c) Ablation restores century 12 to normal"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig2_gt_causal.png", dpi=170); plt.close(fig)


def fig3_cross_task(gt, ioi, out):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    rows = []   # (label, baseline, ablated, random) in failure %
    ab = rd(gt / "causal_ablate_23414__end_cc.csv")
    if ab is not None:
        c = ab[ab.century == 12]
        rows.append(("GT 23414\n@end_cc", 100 * c.fail_base.mean(), 100 * c.fail_abl.mean(), 100 * rd_mean(c, "fail_rd")))
    for pos, lab in [("all", "IOI 17491\nall pos."), ("obj", "IOI 17491\n@obj"), ("to", "IOI 17491\n@to")]:
        d = rd(ioi / f"ioi_ablate_17491__{pos}.csv")
        if d is None:
            continue
        k = d[keycol(d).astype(bool)]
        rnd = 100 * (1 - np.mean([k[c].mean() for c in k.columns if c.startswith("ok_rd")])) if any(c.startswith("ok_rd") for c in k.columns) else np.nan
        rows.append((lab, 100 * (1 - k.ok_base.mean()), 100 * (1 - k.ok_int.mean()), rnd))
    if rows:
        x = np.arange(len(rows))
        ax[0].bar(x - 0.27, [r[1] for r in rows], 0.27, color=GREY, label="baseline")
        ax[0].bar(x, [r[2] for r in rows], 0.27, color=RED, label="feature ablated")
        ax[0].bar(x + 0.27, [r[3] for r in rows], 0.27, color=BLUE, label="same-norm random direction")
        ax[0].set_xticks(x); ax[0].set_xticklabels([r[0] for r in rows], fontsize=8)
        ax[0].set_ylabel("failure rate on the high-failure subset (%)\n(GT: century 12; IOI: keys prompts)")
        ax[0].set_title("(a) Necessity: ablating the top lexical feature"); ax[0].set_ylim(0, 118); ax[0].legend(fontsize=8, loc="upper right")
    rows = []
    inj = rd(gt / "inject_23414__end_cc.csv")
    if inj is not None:
        o = inj[inj.century != 12]
        rows.append(("GT 23414\n@end_cc", 100 * o.fail_base.mean(), 100 * o["fail_s1.0"].mean()))
    for fn, lab in [("ioi_add_17491__obj.csv", "IOI 17491\n@obj"), ("ioi_add_17491__last3.csv", "IOI 17491\n@obj+to")]:
        d = rd(ioi / fn)
        if d is None:
            continue
        o = d[~keycol(d).astype(bool)]
        rows.append((lab, 100 * (1 - o.ok_base.mean()), 100 * (1 - o.ok_int.mean())))
    if rows:
        x = np.arange(len(rows))
        ax[1].bar(x - 0.2, [r[1] for r in rows], 0.4, color=GREY, label="baseline")
        ax[1].bar(x + 0.2, [r[2] for r in rows], 0.4, color=RED, label="feature injected at its high-failure level")
        ax[1].set_xticks(x); ax[1].set_xticklabels([r[0] for r in rows], fontsize=8)
        ax[1].set_ylabel("failure rate on the other prompts (%)")
        ax[1].set_title("(b) Sufficiency: injecting the feature elsewhere"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig3_ioi_vs_gt.png", dpi=170); plt.close(fig)


def fig4_auc(gt, out):
    df = rd(gt / "position_analysis" / "baseline_auc.csv")
    if df is None:
        return
    df = df[~df.predictor.str.contains("raw @")]
    df = df[~df.predictor.str.contains("LEAKY")].iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(df) + 1.2))
    cols = [BLUE if p.startswith(("century", "metadata")) else (GREEN if p.startswith("raw") else RED) for p in df.predictor]
    ax.barh(df.predictor, df.AUC, xerr=df.sd, color=cols)
    ax.set_xlim(0.5, 1.0); ax.set_xlabel("cross-validated ROC AUC (failure)")
    ax.set_title("Prompt variables alone vs. raw residual vs. SAE features (Greater-Than)")
    fig.tight_layout(); fig.savefig(out / "fig4_auc_baselines.png", dpi=170); plt.close(fig)


def fig5_perm(gt, out):
    df = rd(gt / "position_analysis" / "permutation_summary.csv")
    if df is None:
        return
    a = df[df.subset == "all"].set_index("scope").reindex(["pooled_last3"] + ROLE_ORDER[5:])
    a = a.dropna(subset=["obs_n_|d|>.8"])
    x = np.arange(len(a))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, a["obs_n_|d|>.8"], 0.4, color=RED, label="observed")
    ax.bar(x + 0.2, a["strat_null_n_mean"], 0.4, color=GREY, yerr=a["strat_null_n_95"] - a["strat_null_n_mean"], label="null: labels shuffled within century x start-year (mean, 95th pct)")
    ax.set_xticks(x); ax.set_xticklabels(a.index, rotation=30)
    ax.set_ylabel("features with |d| > 0.8")
    ax.set_title("Most large-effect features are explained by the known confounds alone"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig5_permutation_null.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gt-dir", default="results_gt_grid")
    p.add_argument("--ioi-dir", default="results_ioi_pos")
    p.add_argument("--out", default="figures_ext")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    gt, ioi = Path(a.gt_dir), Path(a.ioi_dir)
    fig1_roles(gt, out); fig2_gt_causal(gt, out); fig3_cross_task(gt, ioi, out); fig4_auc(gt, out); fig5_perm(gt, out)
    print(f"figures written to {out}/")
