"""Build the Greater-Than task set (after Hanna et al., 2023).

Prompt:  "The war lasted from the year 1732 to the year 17"
Task:    the next token should be a two-digit year token YY' with YY' > 32.

IMPORTANT (found in sanity check): GPT-2 does NOT always split a 4-digit year
as ' CC' + 'YY'. E.g. 1196 -> ' 119','6'. Prompts with a messy split are a
confound, so by default we keep only prompts whose start year tokenizes as
exactly [' CC', 'YY'] (rejection sampling, still deterministic given --seed).

Each record logs start_yy, century and noun so results can be STRATIFIED.

Usage (from repo root):
    python greaterthan/build_gt_set.py --n 1000 --seed 42 --out data/gt_tasks.jsonl
    python greaterthan/build_gt_set.py --no-clean-only ...   # keep everything
    python greaterthan/build_gt_set.py --n 600 --yy-min 90 --yy-max 98 --out data/gt_tasks_hard.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

NOUNS = ["war", "conflict", "siege", "expedition", "voyage", "famine", "drought", "reign"]
CENTURIES = [11, 12, 13, 14, 15, 16, 17]
YY_MIN, YY_MAX = 2, 98


def is_clean(tok, cc: int, yy: int) -> bool:
    ids = tok.encode(f" {cc}{yy:02d}")
    return len(ids) == 2 and tok.decode(ids[0]) == f" {cc}" and tok.decode(ids[1]) == f"{yy:02d}"


def build_gt(rng: random.Random, yy_min: int = YY_MIN, yy_max: int = YY_MAX) -> dict:
    noun = rng.choice(NOUNS)
    cc = rng.choice(CENTURIES)
    yy = rng.randint(yy_min, yy_max)
    prompt = f"The {noun} lasted from the year {cc}{yy:02d} to the year {cc}"
    return {
        "prompt": prompt, "noun": noun, "century": cc, "start_yy": yy,
        "rule": "next two-digit token must be > start_yy", "n_steps": 1,
    }


def main(n: int, seed: int, out_path: str, clean_only: bool, yy_min: int, yy_max: int) -> None:
    tok = None
    if clean_only:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
    rng = random.Random(seed)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = rejected = 0
    with out.open("w") as f:
        while kept < n:
            t = build_gt(rng, yy_min, yy_max)
            if clean_only and not is_clean(tok, t["century"], t["start_yy"]):
                rejected += 1
                continue
            t["id"] = kept
            f.write(json.dumps(t) + "\n")
            kept += 1
    print(f"Wrote {kept} Greater-Than tasks to {out_path} (rejected {rejected} messy-tokenization candidates)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/gt_tasks.jsonl")
    p.add_argument("--no-clean-only", dest="clean_only", action="store_false")
    p.add_argument("--yy-min", type=int, default=YY_MIN, help="lowest start year (2..98)")
    p.add_argument("--yy-max", type=int, default=YY_MAX, help="highest start year (2..98)")
    a = p.parse_args()
    main(a.n, a.seed, a.out, a.clean_only, a.yy_min, a.yy_max)
