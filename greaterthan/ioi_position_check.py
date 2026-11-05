"""IOI per-position check (the analysis the original paper did not report).

Logs SAE activations at the last --last-n token positions of every IOI prompt and asks:
  * WHERE does feature 17491 ('keys') fire?  (prediction: at the ' keys' token, not at the final ' to')
  * At which relative position does failure-vs-success separate most?
Positions are RELATIVE to the end of the prompt (-1 = final ' to', -2 = object noun, -3 = ' the'),
because IOI prompts differ in length. Uses the paper's own success flags (results/predictions.csv).

  python greaterthan/ioi_position_check.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sae_lens import SAE, HookedSAETransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gt_position_analysis import feature_stats, summarize  # noqa: E402


def main(tasks_path, preds_path, out_dir, feature, last_n, layer, device):
    sae_id = f"blocks.{layer}.hook_resid_pre"
    model = HookedSAETransformer.from_pretrained("gpt2", device=device)
    model.eval()
    sae = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id=sae_id, device=device)
    if isinstance(sae, tuple):
        sae = sae[0]
    sae.eval()

    tasks = [json.loads(l) for l in open(tasks_path)]
    preds = pd.read_csv(preds_path).set_index("id")
    ids = [t["id"] for t in tasks]
    fail = (preds.loc[ids, "success"].values == 0)
    keys = np.array([t["object"] == "the keys" for t in tasks])
    print(f"N={len(tasks)}, failures={fail.sum()}, keys prompts={keys.sum()}, failures on keys={(fail & keys).sum()}")

    mats = {r: [] for r in range(1, last_n + 1)}      # r = positions from the end (1 = final token)
    toks_at = {r: [] for r in range(1, last_n + 1)}
    for i, t in enumerate(tasks):
        toks = model.to_tokens(t["prompt"])
        strs = model.to_str_tokens(t["prompt"])
        assert strs[-1] == " to", strs
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=[sae_id])
            f = sae.encode(cache[sae_id][0]).float().cpu().numpy()      # (T, d_sae)
        for r in range(1, last_n + 1):
            mats[r].append(f[-r])
            toks_at[r].append(strs[-r])
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for r in mats:
        mats[r] = np.stack(mats[r])
    sp.save_npz(out / "acts_rel.npz", sp.csr_matrix(np.concatenate([mats[r] for r in range(1, last_n + 1)], axis=0)))

    rows = []
    for r in range(1, last_n + 1):
        X = mats[r]
        d, p = feature_stats(X, fail)
        s = summarize(d, p)
        d_k = feature_stats(X[keys][:, [feature]], fail[keys])
        d_n = feature_stats(X[~keys][:, [feature]], fail[~keys])
        rows.append({
            "pos_from_end": -r, "common_token_keys": Counter(np.array(toks_at[r])[keys]).most_common(1)[0][0],
            "n_|d|>.8": s["n_large_d"], "n_Holm_sig": s["n_sig_holm"], "max|d|": round(s["max_abs_d"], 2),
            "top_feat": s["top_feature"], f"d[{feature}] all": round(float(d[feature]), 2),
            f"d[{feature}] keys-only": round(float(d_k[0][0]), 2) if d_k else np.nan,
            f"d[{feature}] other-only": round(float(d_n[0][0]), 2) if d_n else np.nan,
            f"act[{feature}] keys": round(float(X[keys, feature].mean()), 2),
            f"act[{feature}] other": round(float(X[~keys, feature].mean()), 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out / "ioi_position_summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("\n=== IOI: failure-vs-success by position from the end of the prompt ===")
    print(df.to_string(index=False))
    print(f"\nWhere does feature {feature} fire? (mean activation on keys prompts by position; compare with 'other')")
    print("If it peaks at -2 (the ' keys' token) and not at -1, the paper's last-3 pooling picked it up from a lexical token.")
    print(f"wrote {out / 'ioi_position_summary.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="data/tasks.jsonl")
    p.add_argument("--preds", default="results/predictions.csv")
    p.add_argument("--out", default="results_ioi_pos")
    p.add_argument("--feature", type=int, default=17491)
    p.add_argument("--last-n", type=int, default=6)
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    main(a.tasks, a.preds, a.out, a.feature, a.last_n, a.layer, a.device)
