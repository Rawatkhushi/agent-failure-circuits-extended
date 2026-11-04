#!/usr/bin/env bash
# Control 3: regenerate the hard corpus under several seeds and repeat the audit + the 23414 ablation.
#   bash greaterthan/run_multiseed.sh            # seeds 1 2 3 4 5
#   bash greaterthan/run_multiseed.sh 1 2        # chosen seeds
set -e
SEEDS="${@:-1 2 3 4 5}"
for s in $SEEDS; do
  D=results_gt_seed$s
  echo "=============== seed $s ==============="
  python greaterthan/build_gt_set.py --n 600 --seed $s --yy-min 90 --yy-max 98 --out data/gt_tasks_seed$s.jsonl
  python greaterthan/run_gt.py --tasks data/gt_tasks_seed$s.jsonl --out $D
  python src/analyze.py --preds $D/predictions.csv --acts $D/acts_last3.npy \
      --out-stats $D/feature_stats_last3.csv --out-top $D/top_features_last3.md --out-fig $D/effect_last3.png > $D/analyze_log.txt
  python greaterthan/gt_ablate.py --tasks data/gt_tasks_seed$s.jsonl --out $D --features 23414 --positions end_cc
done
DIRS="results_gt_hard_sae"
for s in $SEEDS; do DIRS="$DIRS results_gt_seed$s"; done
python greaterthan/gt_multiseed_summary.py --dirs $DIRS
