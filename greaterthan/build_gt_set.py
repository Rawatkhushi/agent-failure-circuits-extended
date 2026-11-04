"""Build the Greater-Than task set (after Hanna et al., 2023).

Prompt:  "The war lasted from the year 1732 to the year 17"
Task:    the next token should be a two-digit year token YY' with YY' > 32.

IMPORTANT (found in sanity check): GPT-2 does NOT always split a 4-digit year as ' CC' + 'YY'
(e.g. 1196 -> ' 119','6'). By default we keep only prompts whose start year tokenizes as exactly
[' CC','YY'].

TWO SAMPLING MODES
  random (default): draws prompts WITH REPLACEMENT from the (noun x century x start-year) space.
        WARNING: that space is small (hard range: 8 x 6 x 9 = 432 prompts), so a 600-prompt draw has
        ~45% duplicates. Duplicates inflate p-values and leak across CV folds.
  --grid: enumerates EVERY (noun, century, start-year) combination in the range exactly once
        (shuffled by --seed). No duplicates, perfectly balanced. Use this for final results.

Usage (from repo root):
    python greaterthan/build_gt_set.py --grid --yy-min 90 --yy-max 98 --out data/gt_grid.jsonl
    python greaterthan/build_gt_set.py --n 600 --seed 42 --yy-min 90 --yy-max 98 --out data/gt_tasks_hard.jsonl
    python greaterthan/build_gt_set.py --no-clean-only ...   # keep messy tokenizations too
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


def make_task(noun: str, cc: int, yy: int) -> dict:
    return {
        "prompt": f"The {noun} lasted from the year {cc}{yy:02d} to the year {cc}",
        "noun": noun, "century": cc, "start_yy": yy,
        "rule": "next two-digit token must be > start_yy", "n_steps": 1,
    }


def build_gt(rng: random.Random, yy_min: int = YY_MIN, yy_max: int = YY_MAX) -> dict:
    return make_task(rng.choice(NOUNS), rng.choice(CENTURIES), rng.randint(yy_min, yy_max))


def main(n: int, seed: int, out_path: str, clean_only: bool, yy_min: int, yy_max: int, grid: bool) -> None:
    tok = None
    if clean_only:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
    rng = random.Random(seed)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = rejected = 0
    with out.open("w") as f:
        if grid:
            combos = [(nn, cc, yy) for nn in NOUNS for cc in CENTURIES for yy in range(yy_min, yy_max + 1)]
            rng.shuffle(combos)
            for nn, cc, yy in combos:
                if clean_only and not is_clean(tok, cc, yy):
                    rejected += 1
                    continue
                t = make_task(nn, cc, yy); t["id"] = kept
                f.write(json.dumps(t) + "\n"); kept += 1
        else:
            while kept < n:
                t = build_gt(rng, yy_min, yy_max)
                if clean_only and not is_clean(tok, t["century"], t["start_yy"]):
                    rejected += 1
                    continue
                t["id"] = kept
                f.write(json.dumps(t) + "\n"); kept += 1
    mode = "grid (every combination once)" if grid else "random with replacement"
    print(f"Wrote {kept} Greater-Than tasks to {out_path} [{mode}] (rejected {rejected} messy-tokenization candidates)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/gt_tasks.jsonl")
    p.add_argument("--no-clean-only", dest="clean_only", action="store_false")
    p.add_argument("--yy-min", type=int, default=YY_MIN, help="lowest start year (2..98)")
    p.add_argument("--yy-max", type=int, default=YY_MAX, help="highest start year (2..98)")
    p.add_argument("--grid", action="store_true", help="enumerate every combination exactly once (ignores --n)")
    a = p.parse_args()
    main(a.n, a.seed, a.out, a.clean_only, a.yy_min, a.yy_max, a.grid)
