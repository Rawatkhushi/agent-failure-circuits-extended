"""Sanity check: how well does GPT-2 small do Greater-Than, and where does it fail?

No SAE needed. For each prompt:
  p_gt   = prob mass on two-digit tokens YY' > start_yy   (candidates 00..99)
  p_le   = prob mass on two-digit tokens YY' <= start_yy
  logit_fail  = 1 if p_gt < p_le           (PRIMARY failure definition)
  greedy_fail = 1 unless full-vocab argmax is a year token > start_yy
Also records whether the start year tokenized cleanly as [' CC','YY'].

Usage (from repo root):
    python greaterthan/gt_sanity.py --tasks data/gt_tasks.jsonl --out results_gt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformer_lens import HookedTransformer


def start_year_tokens(str_toks):
    """Tokens between the first ' year' and the following ' to'."""
    i = str_toks.index(" year")
    j = str_toks.index(" to", i)
    return str_toks[i + 1 : j]


def main(tasks_path: str, out_dir: str, model_name: str, device: str) -> None:
    print(f"device: {device}")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()

    year_ids, missing = {}, []
    for yy in range(0, 100):
        s = f"{yy:02d}"
        toks = model.to_tokens(s, prepend_bos=False)[0]
        if len(toks) == 1:
            year_ids[yy] = int(toks[0])
        else:
            missing.append(s)
    print(f"year tokens found: {len(year_ids)}/100; not single-token: {missing}")
    id_to_yy = {v: k for k, v in year_ids.items()}
    cand_yy = sorted(year_ids)
    ids_t = torch.tensor([year_ids[y] for y in cand_yy], device=device)
    yys = torch.tensor(cand_yy, device=device)

    tasks = [json.loads(l) for l in open(tasks_path)]
    print(f"{len(tasks)} tasks")
    print("example tokenization:", model.to_str_tokens(tasks[0]["prompt"]))

    rows = []
    for i, t in enumerate(tasks):
        tokens = model.to_tokens(t["prompt"])
        str_toks = model.to_str_tokens(t["prompt"])
        with torch.no_grad():
            logits = model(tokens)[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        yp = probs[ids_t]
        s = t["start_yy"]
        p_gt = float(yp[yys > s].sum())
        p_le = float(yp[yys <= s].sum())
        top1 = int(torch.argmax(logits))
        top1_is_year = top1 in id_to_yy
        top1_yy = id_to_yy.get(top1, -1)
        st = start_year_tokens(str_toks)
        clean = int(st == [f" {t['century']}", f"{s:02d}"])
        rows.append({
            "id": t["id"], "noun": t["noun"], "century": t["century"], "start_yy": s,
            "start_tokens": "|".join(st), "clean_tok": clean,
            "p_gt": p_gt, "p_le": p_le, "p_year_mass": p_gt + p_le, "prob_diff": p_gt - p_le,
            "logit_fail": int(p_gt < p_le),
            "top1_is_year": int(top1_is_year), "top1_yy": top1_yy,
            "greedy_fail": int(not (top1_is_year and top1_yy > s)),
            "n_tokens": int(tokens.shape[1]),
        })
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(tasks)}")

    df = pd.DataFrame(rows)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "gt_sanity.csv", index=False)

    print("\n===== Summary =====")
    print(f"clean start-year tokenization: {df.clean_tok.mean():.3f}")
    print(f"mean year-token mass (near 1 is good): {df.p_year_mass.mean():.3f}")
    print(f"logit-fail rate  : {df.logit_fail.mean():.3f}  ({df.logit_fail.sum()} / {len(df)})")
    print(f"greedy-fail rate : {df.greedy_fail.mean():.3f}  ({df.greedy_fail.sum()} / {len(df)})")
    print(f"top1 is a year token: {df.top1_is_year.mean():.3f}")
    df["yy_bin"] = pd.cut(df.start_yy, bins=[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 98])
    print("\nlogit-fail by clean tokenization:")
    print(df.groupby("clean_tok").logit_fail.agg(["mean", "sum", "count"]).round(3).to_string())
    print("\nlogit-fail by start_yy bin:")
    print(df.groupby("yy_bin", observed=True).logit_fail.agg(["mean", "sum", "count"]).round(3).to_string())
    print("\nlogit-fail by noun:")
    print(df.groupby("noun").logit_fail.agg(["mean", "sum", "count"]).round(3).to_string())
    print("\nlogit-fail by century:")
    print(df.groupby("century").logit_fail.agg(["mean", "sum", "count"]).round(3).to_string())
    print("\nfailures: century x yy_bin (counts)")
    print(pd.crosstab(df.century, df.yy_bin, values=df.logit_fail, aggfunc="sum").fillna(0).astype(int).to_string())
    if df.start_yy.nunique() <= 20:
        print("\nlogit-fail by exact start_yy:")
        print(df.groupby("start_yy").logit_fail.agg(["mean", "sum", "count"]).round(3).to_string())
    print("\nstart_yy values of failures:", sorted(df[df.logit_fail == 1].start_yy.tolist()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="data/gt_tasks.jsonl")
    p.add_argument("--out", default="results_gt")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    main(a.tasks, a.out, a.model, a.device)
