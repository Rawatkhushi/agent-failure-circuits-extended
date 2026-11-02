"""Run GPT-2 small on the Greater-Than corpus and log SAE activations at EVERY token position.

Outputs (in --out):
  predictions.csv   one row/prompt: scores + 'success' (= 1 - logit_fail), so the
                    original src/analyze.py works unchanged on it
  acts_last3.npy    (N, d_sae)  mean SAE encoding over the last 3 tokens
                    (the original paper's pooling; use with src/analyze.py)
  raw_last3.npy     (N, d_model) same pooling on the raw residual stream
  acts_pos.npz      scipy CSR, one row per (prompt, token position), d_sae columns
  raw_pos.npy       (R, d_model) raw residual per (prompt, position)
  positions.csv     row index -> id, pos, role, token   (rows align with the two files above)

Roles (assigned by token structure, robust to multi-token nouns):
  bos the0 noun_pre noun lasted from the1 year1 start_cc start_yy to the2 year2 end_cc

Usage (from repo root):
  python greaterthan/run_gt.py --tasks data/gt_tasks_hard.jsonl --out results_gt_hard_sae
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sae_lens import SAE, HookedSAETransformer


def roles_for(str_toks: list[str]) -> list[str]:
    i_lasted = str_toks.index(" lasted")
    assert str_toks[i_lasted + 1] == " from" and str_toks[i_lasted + 2] == " the", str_toks
    i_year1 = i_lasted + 3
    assert str_toks[i_year1] == " year", str_toks
    i_to = str_toks.index(" to", i_year1)
    assert len(str_toks) == i_to + 4, f"unexpected length: {str_toks}"
    roles = ["other"] * len(str_toks)
    roles[0] = "bos"
    roles[1] = "the0"
    for k in range(2, i_lasted):
        roles[k] = "noun" if k == i_lasted - 1 else "noun_pre"
    roles[i_lasted] = "lasted"
    roles[i_lasted + 1] = "from"
    roles[i_lasted + 2] = "the1"
    roles[i_year1] = "year1"
    start = list(range(i_year1 + 1, i_to))
    if len(start) == 2:
        roles[start[0]], roles[start[1]] = "start_cc", "start_yy"
    else:
        for n, k in enumerate(start):
            roles[k] = f"start_{n}"
    roles[i_to] = "to"
    roles[i_to + 1] = "the2"
    roles[i_to + 2] = "year2"
    roles[i_to + 3] = "end_cc"
    return roles


def main(tasks_path, out_dir, model_name, sae_release, layer, device, last_k):
    sae_id = f"blocks.{layer}.hook_resid_pre"
    print(f"device: {device}")
    model = HookedSAETransformer.from_pretrained(model_name, device=device)
    model.eval()
    sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)
    if isinstance(sae, tuple):          # older sae_lens versions return a tuple
        sae = sae[0]
    sae.eval()
    print(f"SAE {sae_release}/{sae_id}: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

    # candidate year tokens 00..99
    year_ids = {}
    for yy in range(100):
        t = model.to_tokens(f"{yy:02d}", prepend_bos=False)[0]
        if len(t) == 1:
            year_ids[yy] = int(t[0])
    id_to_yy = {v: k for k, v in year_ids.items()}
    cand = sorted(year_ids)
    ids_t = torch.tensor([year_ids[y] for y in cand], device=device)
    yys = torch.tensor(cand, device=device)

    tasks = [json.loads(l) for l in open(tasks_path)]
    print(f"{len(tasks)} tasks, hook {sae_id}")

    pred_rows, last3, rawlast3, sp_rows, raw_rows, pos_rows = [], [], [], [], [], []
    r = 0
    for i, t in enumerate(tasks):
        tokens = model.to_tokens(t["prompt"])
        str_toks = model.to_str_tokens(t["prompt"])
        roles = roles_for(str_toks)
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens, names_filter=[sae_id])
            resid = cache[sae_id][0]                    # (T, d_model)
            feats = sae.encode(resid)                   # (T, d_sae)
            lg = logits[0, -1]
            probs = torch.softmax(lg.float(), dim=-1)
        yp = probs[ids_t]
        s = t["start_yy"]
        p_gt = float(yp[yys > s].sum())
        p_le = float(yp[yys <= s].sum())
        top1 = int(torch.argmax(lg))
        top1_yy = id_to_yy.get(top1, -1)
        logit_fail = int(p_gt < p_le)
        greedy_fail = int(not (top1_yy > s))
        pred_rows.append({
            "id": t["id"], "prompt": t["prompt"], "noun": t["noun"], "century": t["century"],
            "start_yy": s, "p_gt": p_gt, "p_le": p_le, "prob_diff": p_gt - p_le,
            "logit_fail": logit_fail, "greedy_fail": greedy_fail,
            "success": 1 - logit_fail, "n_tokens": len(str_toks),
        })
        f_np = feats.float().cpu().numpy()
        r_np = resid.float().cpu().numpy()
        last3.append(f_np[-last_k:].mean(axis=0))
        rawlast3.append(r_np[-last_k:].mean(axis=0))
        sp_rows.append(sp.csr_matrix(f_np))
        raw_rows.append(r_np)
        for pos, (role, tok) in enumerate(zip(roles, str_toks)):
            pos_rows.append({"row": r, "id": t["id"], "pos": pos, "role": role, "token": tok})
            r += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(tasks)}")

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pred_rows).to_csv(out / "predictions.csv", index=False)
    np.save(out / "acts_last3.npy", np.stack(last3))
    np.save(out / "raw_last3.npy", np.stack(rawlast3))
    sp.save_npz(out / "acts_pos.npz", sp.vstack(sp_rows).tocsr())
    np.save(out / "raw_pos.npy", np.concatenate(raw_rows, axis=0))
    pd.DataFrame(pos_rows).to_csv(out / "positions.csv", index=False)

    df = pd.DataFrame(pred_rows)
    print(f"\nwrote files to {out}")
    print(f"rows (prompt x position): {r}; acts_pos nnz = {sp.vstack(sp_rows).nnz}")
    print(f"logit-fail {df.logit_fail.sum()}/{len(df)} = {df.logit_fail.mean():.3f}; "
          f"greedy-fail {df.greedy_fail.sum()}/{len(df)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="data/gt_tasks_hard.jsonl")
    p.add_argument("--out", default="results_gt_hard_sae")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--sae-release", default="gpt2-small-res-jb")
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--last-k", type=int, default=3)
    a = p.parse_args()
    main(a.tasks, a.out, a.model, a.sae_release, a.layer, a.device, a.last_k)
