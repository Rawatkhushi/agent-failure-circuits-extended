"""Control 1: causal tests of chosen SAE features on Greater-Than, with POSITION control.

MODES
  ablate (default): subtract the feature's decoded contribution at the chosen positions
        (resid <- resid - f_i(x) * W_dec[i]; same rule as src/ablation.py).   [necessity test]
  add:  SET the feature's activation, at the chosen token roles, to its mean value on
        century-12 prompts (for every prompt) by adding (target - current) * W_dec[i].
        Look at 'other centuries': do failures appear?                        [sufficiency test]

CONTROLS
  --random-k K     ablate K random features active at the target positions (small-feature null)
  --random-dir N   ablate N random DIRECTIONS with the same norm as the removed contribution,
                   per prompt and position (size-matched null; ablate mode only)

Reports failure rate and mean prob_diff (valid-year mass minus invalid-year mass), baseline vs
intervened, for all / century 12 / other centuries, plus prompts FIXED (fail->ok) vs BROKEN
(ok->fail) and an exact paired (McNemar-style) p-value.

  python greaterthan/gt_ablate.py --features 23414 --positions end_cc --random-dir 5
  python greaterthan/gt_ablate.py --features 23414 --positions start_cc --random-dir 5
  python greaterthan/gt_ablate.py --features 23414 --positions start_cc end_cc --mode add
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as sstats
from sae_lens import SAE, HookedSAETransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gt import roles_for  # noqa: E402


def main(tasks_path, out_dir, features, positions, layer, device, random_k, n_random,
         seed, mode, n_randdir):
    sae_id = f"blocks.{layer}.hook_resid_pre"
    if mode == "add" and (positions == ["all"] or len(features) != 1):
        raise SystemExit("--mode add needs exactly one feature and explicit --positions roles (not 'all').")
    model = HookedSAETransformer.from_pretrained("gpt2", device=device)
    model.eval()
    sae = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id=sae_id, device=device)
    if isinstance(sae, tuple):
        sae = sae[0]
    sae.eval()
    W_dec = sae.W_dec

    year_ids = {}
    for yy in range(100):
        t = model.to_tokens(f"{yy:02d}", prepend_bos=False)[0]
        if len(t) == 1:
            year_ids[yy] = int(t[0])
    cand = sorted(year_ids)
    ids_t = torch.tensor([year_ids[y] for y in cand], device=device)
    yys = torch.tensor(cand, device=device)

    tasks = [json.loads(l) for l in open(tasks_path)]
    prep = []
    for t in tasks:
        toks = model.to_tokens(t["prompt"])
        roles = roles_for(model.to_str_tokens(t["prompt"]))
        if positions == ["all"]:
            pidx = list(range(len(roles)))
        else:
            pidx = [i for i, r in enumerate(roles) if r in positions]
            if not pidx:
                raise SystemExit(f"no token with role in {positions}; roles are {sorted(set(roles))}")
        prep.append((toks, pidx, roles))

    def score(toks, pidx, kind, feats_idx=None, target=None, gen=None):
        mask = torch.zeros(toks.shape[1], device=device)
        mask[pidx] = 1.0

        def hook(act, hook):
            if kind == "none":
                return act
            f = sae.encode(act)                                   # (1, T, d_sae)
            if kind in ("ablate", "randdir"):
                contrib = f[..., feats_idx] @ W_dec[feats_idx, :]  # (1, T, d_model)
                if kind == "randdir":
                    nrm = contrib.norm(dim=-1, keepdim=True)
                    r = torch.randn(contrib.shape, generator=gen).to(act.device)
                    r = r / r.norm(dim=-1, keepdim=True)
                    contrib = nrm * r
                return act - contrib * mask[None, :, None]
            if kind == "add":
                cur = f[..., int(feats_idx[0])]                    # (1, T)
                delta = (target[None, :] - cur) * mask[None, :]
                return act + delta[..., None] * W_dec[int(feats_idx[0]), :]
            raise ValueError(kind)

        with torch.no_grad():
            lg = model.run_with_hooks(toks, fwd_hooks=[(sae_id, hook)])[0, -1]
            return torch.softmax(lg.float(), dim=-1)[ids_t]

    def metrics(pr, s):
        p_gt, p_le = float(pr[yys > s].sum()), float(pr[yys <= s].sum())
        return p_gt - p_le, int(p_gt < p_le)

    # --- extra preparation ---
    mean_c12 = {}
    if mode == "add":
        acc = {r: [] for r in positions}
        for t, (toks, pidx, roles) in zip(tasks, prep):
            if t["century"] != 12:
                continue
            with torch.no_grad():
                _, cache = model.run_with_cache(toks, names_filter=[sae_id])
                f = sae.encode(cache[sae_id][0])
            for p in pidx:
                acc[roles[p]].append(float(f[p, features[0]]))
        mean_c12 = {r: float(np.mean(v)) for r, v in acc.items()}
        print(f"target activation of feature {features[0]} (mean on century-12 prompts) by role: {mean_c12}")

    pool = None
    if random_k:
        counts = None
        for toks, pidx, _ in prep:
            with torch.no_grad():
                _, cache = model.run_with_cache(toks, names_filter=[sae_id])
                f = sae.encode(cache[sae_id][0][pidx]) > 0
            a = f.any(dim=0).cpu().numpy().astype(int)
            counts = a if counts is None else counts + a
        pool = np.array([i for i in np.where(counts >= 0.05 * len(prep))[0] if i not in set(features)])
        print(f"random-feature pool: {len(pool)} features active at target positions in >=5% of prompts")

    rng = np.random.default_rng(seed)
    fidx = torch.tensor(features, device=device, dtype=torch.long)
    rand_runs = [rng.choice(pool, size=random_k, replace=False) for _ in range(n_random)] if random_k else []
    rows = []
    for i, (t, (toks, pidx, roles)) in enumerate(zip(tasks, prep)):
        s = t["start_yy"]
        pd0, f0 = metrics(score(toks, pidx, "none"), s)
        if mode == "ablate":
            pd1, f1 = metrics(score(toks, pidx, "ablate", fidx), s)
        else:
            tgt = torch.zeros(toks.shape[1], device=device)
            for p in pidx:
                tgt[p] = mean_c12[roles[p]]
            pd1, f1 = metrics(score(toks, pidx, "add", fidx, target=tgt), s)
        row = {"id": t["id"], "century": t["century"], "start_yy": s,
               "pd_base": pd0, "fail_base": f0, "pd_abl": pd1, "fail_abl": f1}
        for j, r in enumerate(rand_runs):
            pdr, fr = metrics(score(toks, pidx, "ablate", torch.tensor(r, device=device, dtype=torch.long)), s)
            row[f"pd_rand{j}"], row[f"fail_rand{j}"] = pdr, fr
        if mode == "ablate":
            for j in range(n_randdir):
                g = torch.Generator().manual_seed(seed * 100003 + i * 101 + j)
                pdr, fr = metrics(score(toks, pidx, "randdir", fidx, gen=g), s)
                row[f"pd_rd{j}"], row[f"fail_rd{j}"] = pdr, fr
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(tasks)}")

    df = pd.DataFrame(rows)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = f"{mode}_" + "_".join(map(str, features)) + "__" + "-".join(positions)
    df.to_csv(out / f"causal_{tag}.csv", index=False)

    label = "ABLATION" if mode == "ablate" else "ADDITION (set to century-12 level)"
    print(f"\n===== {label} of features {features} at positions {positions} =====")
    subsets = {"all": df.century > 0, "century 12": df.century == 12, "other centuries": df.century != 12}
    for name, m in subsets.items():
        d = df[m]
        fixed = int(((d.fail_base == 1) & (d.fail_abl == 0)).sum())
        broken = int(((d.fail_base == 0) & (d.fail_abl == 1)).sum())
        pval = sstats.binomtest(fixed, fixed + broken, 0.5).pvalue if fixed + broken > 0 else float("nan")
        print(f"{name:16s} n={len(d):3d} | fail rate {d.fail_base.mean():.3f} -> {d.fail_abl.mean():.3f} "
              f"(delta {d.fail_abl.mean() - d.fail_base.mean():+.3f}) | prob_diff "
              f"{d.pd_base.mean():.3f} -> {d.pd_abl.mean():.3f} (delta {d.pd_abl.mean() - d.pd_base.mean():+.3f})")
        print(f"{'':16s}   fixed (fail->ok) {fixed}, broken (ok->fail) {broken}, exact paired p = {pval:.2g}")
        for lab, fk, pk, cnt in [(f"random-{random_k} features", "fail_rand", "pd_rand", len(rand_runs)),
                                 ("size-matched random direction", "fail_rd", "pd_rd", n_randdir if mode == "ablate" else 0)]:
            if cnt:
                dr = [d[f"{fk}{j}"].mean() - d.fail_base.mean() for j in range(cnt)]
                dp = [d[f"{pk}{j}"].mean() - d.pd_base.mean() for j in range(cnt)]
                print(f"{'':16s}   {lab}: fail delta {np.mean(dr):+.3f} (range {min(dr):+.3f}..{max(dr):+.3f}), "
                      f"prob_diff delta {np.mean(dp):+.3f} (range {min(dp):+.3f}..{max(dp):+.3f})")
    print(f"\nwrote {out / f'causal_{tag}.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="data/gt_tasks_hard.jsonl")
    p.add_argument("--out", default="results_gt_hard_sae")
    p.add_argument("--features", type=int, nargs="+", required=True)
    p.add_argument("--positions", nargs="+", default=["all"])
    p.add_argument("--mode", choices=["ablate", "add"], default="ablate")
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--random-k", type=int, default=0)
    p.add_argument("--n-random", type=int, default=3)
    p.add_argument("--random-dir", type=int, default=0, dest="random_dir")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.tasks, a.out, a.features, a.positions, a.layer, a.device, a.random_k, a.n_random,
         a.seed, a.mode, a.random_dir)
