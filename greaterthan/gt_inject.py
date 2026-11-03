"""Sufficiency tests for one SAE feature: dose-response injection + size-matched random control.

For each prompt and each SCALE s, at the chosen token roles the feature's activation is SET to
  s * (its mean activation on century-12 prompts at that role)
by adding (target - current) * W_dec[feature] to blocks.<layer>.hook_resid_pre.
  - on prompts where the feature is ~silent (other centuries) this is a dose-controlled INJECTION
  - on century-12 prompts, s < 1 is a PARTIAL ABLATION (they already sit near the target at s = 1)
--random-dir N (applied at scale 1.0): instead of the feature direction, add a RANDOM direction with
the same norm as the injected vector (per prompt and position). If random pushes of the same size
do not raise failures, the effect is specific to this feature's direction.

  python greaterthan/gt_inject.py --feature 23414 --positions end_cc --random-dir 5
  python greaterthan/gt_inject.py --feature 23414 --positions start_cc end_cc --random-dir 5
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


def main(tasks_path, out_dir, feature, positions, scales, n_randdir, layer, device, seed):
    sae_id = f"blocks.{layer}.hook_resid_pre"
    if positions == ["all"]:
        raise SystemExit("give explicit roles, e.g. --positions end_cc")
    model = HookedSAETransformer.from_pretrained("gpt2", device=device)
    model.eval()
    sae = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id=sae_id, device=device)
    if isinstance(sae, tuple):
        sae = sae[0]
    sae.eval()
    w = sae.W_dec[feature, :]
    w_norm = float(w.norm())

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
        pidx = [i for i, r in enumerate(roles) if r in positions]
        if not pidx:
            raise SystemExit(f"no token with role in {positions}; roles are {sorted(set(roles))}")
        prep.append((toks, pidx, roles))

    def score(toks, pidx, kind, target=None, gen=None):
        mask = torch.zeros(toks.shape[1], device=device)
        mask[pidx] = 1.0

        def hook(act, hook):
            if kind == "none":
                return act
            f = sae.encode(act)
            delta = (target[None, :] - f[..., feature]) * mask[None, :]       # (1, T)
            if kind == "add":
                return act + delta[..., None] * w
            if kind == "randadd":
                nrm = delta.abs() * w_norm
                r = torch.randn((1, toks.shape[1], act.shape[-1]), generator=gen).to(act.device)
                r = r / r.norm(dim=-1, keepdim=True)
                return act + nrm[..., None] * r
            raise ValueError(kind)

        with torch.no_grad():
            lg = model.run_with_hooks(toks, fwd_hooks=[(sae_id, hook)])[0, -1]
            return torch.softmax(lg.float(), dim=-1)[ids_t]

    def metrics(pr, s):
        p_gt, p_le = float(pr[yys > s].sum()), float(pr[yys <= s].sum())
        return p_gt - p_le, int(p_gt < p_le)

    acc = {r: [] for r in positions}
    for t, (toks, pidx, roles) in zip(tasks, prep):
        if t["century"] != 12:
            continue
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=[sae_id])
            f = sae.encode(cache[sae_id][0])
        for p in pidx:
            acc[roles[p]].append(float(f[p, feature]))
    mean_c12 = {r: float(np.mean(v)) for r, v in acc.items()}
    print(f"feature {feature}: mean activation on century-12 prompts by role = {mean_c12}")

    rows = []
    for i, (t, (toks, pidx, roles)) in enumerate(zip(tasks, prep)):
        s0 = t["start_yy"]
        row = {"id": t["id"], "century": t["century"], "start_yy": s0}
        row["pd_base"], row["fail_base"] = metrics(score(toks, pidx, "none"), s0)
        for sc in scales:
            tgt = torch.zeros(toks.shape[1], device=device)
            for p in pidx:
                tgt[p] = sc * mean_c12[roles[p]]
            row[f"pd_s{sc}"], row[f"fail_s{sc}"] = metrics(score(toks, pidx, "add", target=tgt), s0)
            if sc == 1.0:
                for j in range(n_randdir):
                    g = torch.Generator().manual_seed(seed * 100003 + i * 101 + j)
                    row[f"pd_rd{j}"], row[f"fail_rd{j}"] = metrics(score(toks, pidx, "randadd", target=tgt, gen=g), s0)
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(tasks)}")

    df = pd.DataFrame(rows)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = f"{feature}__{'-'.join(positions)}"
    df.to_csv(out / f"inject_{tag}.csv", index=False)

    print(f"\n===== DOSE-RESPONSE: feature {feature} set to scale x century-12 level at {positions} =====")
    for name, m in {"other centuries": df.century != 12, "century 12 (s<1 = partial ablation)": df.century == 12}.items():
        d = df[m]
        print(f"\n{name}  (n={len(d)})")
        print(f"  {'scale':>8} | fail rate | prob_diff | fixed | broken | paired p")
        print(f"  {'baseline':>8} | {d.fail_base.mean():9.3f} | {d.pd_base.mean():9.3f} |")
        for sc in scales:
            fx = int(((d.fail_base == 1) & (d[f'fail_s{sc}'] == 0)).sum())
            br = int(((d.fail_base == 0) & (d[f'fail_s{sc}'] == 1)).sum())
            p = sstats.binomtest(fx, fx + br, 0.5).pvalue if fx + br else float("nan")
            print(f"  {sc:>8} | {d[f'fail_s{sc}'].mean():9.3f} | {d[f'pd_s{sc}'].mean():9.3f} | {fx:5d} | {br:6d} | {p:.2g}")
        if n_randdir:
            fr = [d[f"fail_rd{j}"].mean() for j in range(n_randdir)]
            pr = [d[f"pd_rd{j}"].mean() for j in range(n_randdir)]
            print(f"  random direction, same norm (scale 1.0, {n_randdir} draws): fail rate {np.mean(fr):.3f} "
                  f"(range {min(fr):.3f}..{max(fr):.3f}), prob_diff {np.mean(pr):.3f} (range {min(pr):.3f}..{max(pr):.3f})")
    print(f"\nwrote {out / f'inject_{tag}.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="data/gt_tasks_hard.jsonl")
    p.add_argument("--out", default="results_gt_hard_sae")
    p.add_argument("--feature", type=int, required=True)
    p.add_argument("--positions", nargs="+", required=True)
    p.add_argument("--scales", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    p.add_argument("--random-dir", type=int, default=0, dest="random_dir")
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.tasks, a.out, a.feature, a.positions, a.scales, a.random_dir, a.layer, a.device, a.seed)
