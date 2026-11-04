#!/usr/bin/env bash
# Final-results pipeline on the DEDUPLICATED grid corpus (every noun x century x start-year once).
# Logs each step to results_gt_grid/logs/*.txt so the outputs are easy to paste. ~25 min on CPU.
#   source .venv/bin/activate && bash greaterthan/run_grid.sh
set -eo pipefail
D=results_gt_grid
mkdir -p $D/logs
python greaterthan/build_gt_set.py --grid --yy-min 90 --yy-max 98 --seed 42 --out data/gt_grid.jsonl | tee $D/logs/00_build.txt
python greaterthan/run_gt.py --tasks data/gt_grid.jsonl --out $D 2>&1 | sed "/Warning/d" | tee $D/logs/01_run.txt
python src/analyze.py --preds $D/predictions.csv --acts $D/acts_last3.npy \
    --out-stats $D/feature_stats_last3.csv --out-top $D/top_features_last3.md --out-fig $D/effect_last3.png 2>&1 | tee $D/logs/02_analyze.txt
python greaterthan/gt_position_analysis.py --dir $D 2>&1 | tee $D/logs/03_position.txt
python greaterthan/gt_raw_baseline.py --dir $D --per-role 2>&1 | tee $D/logs/04_baseline.txt
python greaterthan/gt_permutation_null.py --dir $D --n-perm 200 2>&1 | tee $D/logs/05_permutation.txt
python greaterthan/gt_ablate.py --tasks data/gt_grid.jsonl --out $D --features 23414 --positions end_cc --random-dir 5 2>&1 | sed "/Warning/d" | tee $D/logs/06_ablate.txt
python greaterthan/gt_inject.py --tasks data/gt_grid.jsonl --out $D --feature 23414 --positions end_cc --random-dir 5 2>&1 | sed "/Warning/d" | tee $D/logs/07_inject.txt
echo "DONE. Paste the contents of $D/logs/ (or run: cat $D/logs/0[2-7]*.txt)"
