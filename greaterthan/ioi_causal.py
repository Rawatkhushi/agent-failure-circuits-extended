"""IOI causal tests of one SAE feature (default 17491) with the SAME tooling as the Greater-Than study.

Score (first predicted token, greedy): success = the argmax token, stripped, equals the indirect-object
name. Also reports logit_diff = logit[IO] - logit[subject] and the-gap = logit[" the"] - logit[IO] (the paper's keys failures
are mostly the model continuing "the ..." instead of a name; positive the-gap = " the" beats the IO name).
Baseline is validated against the paper's results/predictions.csv (generation-based).

POSITIONS (relative to the end of the prompt): all | to (-1) | obj (-2, the object noun, e.g. ' keys')
                                               | the_obj (-3) | last3 (-3..-1)
MODES
  ablate: subtract the feature's decoded contribution at those positions   [necessity]
  add:    set the feature to its mean activation on KEYS prompts at those positions, for every prompt [sufficiency]
--random-dir N: ablate mode; N random directions with the same norm as the removed contribution.

  python greaterthan/ioi_causal.py --positions all                       # replicates the paper's protocol
  python greaterthan/ioi_causal.py --positions obj --random-dir 5
  python greaterthan/ioi_causal.py --positions to --random-dir 5
  python greaterthan/ioi_causal.py --positions obj --mode add
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as sstats
from sae_lens import SAE, HookedSAETransformer

REL = {"to": [-1], "obj": [-2], "the_obj": [-3], "last3": [-3, -2, -1]}


def rel_to_abs(name, T):
    return list(range(T)) if name == "all" else [T + r for r in REL[name]]


def norm_tok(s):
    s = s.strip()
    return s.split()[0].rstrip(".,!?;:") if s else ""


def main(tasks_path, preds_path, out_dir, feature, position, mode, layer, device, n_randdir, seed):
    sae_id = f"blocks.{layer}.hook_resid_pre"
    if mode == "add" and position == "all":
        raise SystemExit("--mode add needs a specific --positions (to / obj / the_obj / last3)")
    model = HookedSAETransformer.from_pretrained("gpt2", device=device)
    model.eval()
    sae = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id=sae_id, device=device)
    if isinstance(sae, tuple):
        sae = sae[0]
    sae.eval()
    w = sae.W_dec[feature, :]
    fidx = torch.tensor([feature], device=device, dtype=torch.long)

    tasks = [json.loads(l) for l in open(tasks_path)]
    paper = pd.read_csv(preds_path).set_index("id")
    the_id = int(model.to_tokens(" the", prepend_bos=False)[0][0])
    prep = []
    for t in tasks:
        toks = model.to_tokens(t["prompt"])
        assert model.to_str_tokens(t["prompt"])[-1] == " to"
        io = int(model.to_tokens(" " + t["indirect_object"], prepend_bos=False)[0][0])
        sb = int(model.to_tokens(" " + t["subject"], prepend_bos=False)[0][0])
        prep.append((toks, rel_to_abs(position, toks.shape[1]), io, sb))

    def run(toks, pidx, kind, target=None, gen=None):
        mask = torch.zeros(toks.shape[1], device=device); mask[pidx] = 1.0

        def hook(act, hook):
            if kind == "none":
                return act
            f = sae.encode(act)
            if kind in ("ablate", "randdir"):
                contrib = f[..., fidx] @ sae.W_dec[fidx, :]
                if kind == "randdir":
                    nrm = contrib.norm(dim=-1, keepdim=True)
                    r = torch.randn(contrib.shape, generator=gen).to(act.device)
                    contrib = nrm * r / r.norm(dim=-1, keepdim=True)
                return act - contrib * mask[None, :, None]
            delta = (target[None, :] - f[..., feature]) * mask[None, :]       # add
            return act + delta[..., None] * w

        with torch.no_grad():
            return model.run_with_hooks(toks, fwd_hooks=[(sae_id, hook)])[0, -1].float()

    def score(lg, t, io, sb):
        top = norm_tok(model.to_string(lg.argmax().reshape(1)))
        return int(top == t["indirect_object"]), float(lg[io] - lg[sb]), float(lg[the_id] - lg[io])

    keys = np.array([t["object"] == "the keys" for t in tasks])
    mean_keys = {}
    if mode == "add":
        acc = {}
        for t, (toks, pidx, io, sb) in zip(tasks, prep):
            if t["object"] != "the keys":
                continue
            with torch.no_grad():
                _, cache = model.run_with_cache(toks, names_filter=[sae_id])
                f = sae.encode(cache[sae_id][0])
            for p in pidx:
                acc.setdefault(p - toks.shape[1], []).append(float(f[p, feature]))
        mean_keys = {r: float(np.mean(v)) for r, v in acc.items()}
        print(f"mean activation of {feature} on keys prompts by relative position: {mean_keys}")

    rows = []
    for i, (t, (toks, pidx, io, sb)) in enumerate(zip(tasks, prep)):
        row = {"id": t["id"], "is_keys": bool(keys[i])}
        row["ok_base"], row["ld_base"], row["gap_base"] = score(run(toks, pidx, "none"), t, io, sb)
        if mode == "ablate":
            lg = run(toks, pidx, "ablate")
        else:
            tgt = torch.zeros(toks.shape[1], device=device)
            for p in pidx:
                tgt[p] = mean_keys[p - toks.shape[1]]
            lg = run(toks, pidx, "add", target=tgt)
        row["ok_int"], row["ld_int"], row["gap_int"] = score(lg, t, io, sb)
        if mode == "ablate":
            for j in range(n_randdir):
                g = torch.Generator().manual_seed(seed * 100003 + i * 101 + j)
                row[f"ok_rd{j}"], row[f"ld_rd{j}"], row[f"gap_rd{j}"] = score(run(toks, pidx, "randdir", gen=g), t, io, sb)
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(tasks)}")

    df = pd.DataFrame(rows)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"ioi_{mode}_{feature}__{position}.csv", index=False)

    agree = int((df.ok_base.values == paper.loc[df.id, "success"].values).sum())
    print(f"\nbaseline (first-token argmax) accuracy {df.ok_base.mean():.3f} ({int(df.ok_base.sum())}/{len(df)}); "
          f"paper's generation-based accuracy {paper.loc[df.id, 'success'].mean():.3f}; "
          f"per-prompt agreement with paper's flags: {agree}/{len(df)}")
    print(f"\n===== IOI {mode.upper()} of feature {feature} at positions '{position}' =====")
    for name, m in {"all": np.ones(len(df), bool), "keys prompts": df.is_keys.values, "other prompts": ~df.is_keys.values}.items():
        d = df[m]
        fx = int(((d.ok_base == 0) & (d.ok_int == 1)).sum()); br = int(((d.ok_base == 1) & (d.ok_int == 0)).sum())
        p = sstats.binomtest(fx, fx + br, 0.5).pvalue if fx + br else float("nan")
        print(f"{name:14s} n={len(d):3d} | accuracy {d.ok_base.mean():.3f} -> {d.ok_int.mean():.3f} | "
              f"logit_diff {d.ld_base.mean():+.2f} -> {d.ld_int.mean():+.2f} | the-gap {d.gap_base.mean():+.2f} -> {d.gap_int.mean():+.2f} | gained {fx}, lost {br}, paired p = {p:.2g}")
        if mode == "ablate" and n_randdir:
            acc = [d[f"ok_rd{j}"].mean() for j in range(n_randdir)]; ld = [d[f"ld_rd{j}"].mean() for j in range(n_randdir)]
            gp = [d[f"gap_rd{j}"].mean() for j in range(n_randdir)]
            print(f"{'':14s}   size-matched random direction: accuracy {np.mean(acc):.3f} (range {min(acc):.3f}..{max(acc):.3f}), "
                  f"logit_diff {np.mean(ld):+.2f} (range {min(ld):+.2f}..{max(ld):+.2f}), the-gap {np.mean(gp):+.2f} (range {min(gp):+.2f}..{max(gp):+.2f})")
    print(f"\nwrote {out / f'ioi_{mode}_{feature}__{position}.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="data/tasks.jsonl")
    p.add_argument("--preds", default="results/predictions.csv")
    p.add_argument("--out", default="results_ioi_pos")
    p.add_argument("--feature", type=int, default=17491)
    p.add_argument("--positions", default="all", choices=["all", "to", "obj", "the_obj", "last3"])
    p.add_argument("--mode", default="ablate", choices=["ablate", "add"])
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--random-dir", type=int, default=0, dest="random_dir")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.tasks, a.preds, a.out, a.feature, a.positions, a.mode, a.layer, a.device, a.random_dir, a.seed)
