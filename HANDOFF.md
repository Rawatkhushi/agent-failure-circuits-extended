# HANDOFF v3: extending agent-failure-circuits (IOI -> Greater-Than + per-position SAE audit)

## Setup
- User Khushi (GitHub Rawatkhushi), Mac Air M4, runs LOCALLY on CPU, Python 3.13 venv (ALWAYS `source .venv/bin/activate`; prompt must show (.venv)). Claude's sandbox reaches GitHub/PyPI only, not HF/Neuronpedia. Claude writes code, user runs and pastes output. Fork: Rawatkhushi/agent-failure-circuits-extended (upstream mahdinaser/agent-failure-circuits, arXiv 2605.22719; GPT-2-small, SAE gpt2-small-res-jb, hook blocks.8.hook_resid_pre).
- Goal: does the paper's "correlate, not cause" result generalize to Greater-Than? Plus per-position analysis (paper mean-pools the last 3 tokens).
- All new code is in greaterthan/ (original src/ untouched). Tested: build_gt_set.py (--grid), gt_sanity.py, run_gt.py, gt_position_analysis.py, gt_raw_baseline.py, gt_permutation_null.py, gt_ablate.py, gt_inject.py, run_multiseed.sh, gt_multiseed_summary.py, run_grid.sh. If file cards fail to download, ask Claude to paste code via heredoc.

## Design (final)
- Prompt "The {noun} lasted from the year {CC}{YY} to the year {CC}"; 8 nouns; failure = valid-year mass (> start, tokens 00..99) < invalid mass (logit_fail); greedy_fail also logged. Only clean [' CC','YY'] tokenizations kept (century 11 mostly rejected).
- Random sampling WITH replacement gave ~45% duplicate prompts (universe <= 432), which inflates p-values and CV scores. FINAL corpus = GRID: every noun x century (12-17) x start yy (90-98) once = 432 prompts (data/gt_grid.jsonl, results_gt_grid/), 56 failures (43 in century 12). Earlier duplicated-corpus results (results_gt_hard_sae/, seeds 1-3) are superseded for p-values and AUCs; effect sizes agreed.
- Failure is a deterministic function of (noun, century, start yy); within a century x yy stratum only the noun varies.

## Grid results (cite these)
- Century 12 fails 59.7% vs 3.6% elsewhere. Pooled last-3 audit: 103 Holm-significant, 85 |d|>.8; top 23414 (+2.35), 8350 (+2.28), 7807 (-1.97).
- Per-position: bos/the0 d=0; positions before the century token carry only noun identity (no sig features; raw AUC .604 = noun lookup); signal starts at start_cc. Sig features: start_cc 38, start_yy 63, year2 58, end_cc 56, the2 13 (pooled window = the2, year2, end_cc). Per-position advantage over pooling is modest.
- CV AUC: century .884; +start_yy .949; +noun .976; raw last-3 .991; SAE top-100 nested .987 (in-fold selection ~ leaky here). SAE adds nothing over raw or over prompt variables.
- Permutation null: full-shuffle chance count of |d|>.8 is ~0; shuffling within century x yy strata reproduces ~69 of the 85 -> most "significant" features just track the strata. 23414/7807/11842/5868/17942/20679 have strata p 0.22-1.0. 8350 passes the strata null only at year2 (p=.005), fragile (no_c12 p=.10). Noun-position features pass it (noun is not in the strata).
- Feature 23414 = token " 12" detector (Neuronpedia "instances of the number 12", score 96; positive logits include 02,03,04,06). Fires only at start_cc/end_cc; corr(century12)=1.00; within-c12 d=-0.30; ~29 at end_cc on c12 prompts vs ~0 elsewhere.
- CAUSAL (grid): ablate 23414 at end_cc: c12 failure 59.7->9.7% (36 fixed/0 broken, p~3e-11); same-norm random direction makes it worse (+6.1pp). Inject at end_cc into other centuries: 3.6% -> 7.5/15.0/41.9% at 0.25/0.5/1.0x of the c12 level (138 broken/0 fixed at 1.0x; random same-norm: 8.7%). Partial ablation in c12: 16.7%/29.2% at .25/.5x. Ablation fixes only 5/13 non-c12 failures (p=.062): do NOT claim.
- NULL ablations (grid): 8350 at year2 = exactly 0 effect (0 fixed, 0 broken); 7807 at end_cc = no failure changes, prob_diff -0.008 (tiny, expected direction). Both features have small activations, so size-matched random controls are uninformative for them.
- Multi-seed (resamples of the same prompt universe, so weak evidence): 23414 rank 1-2, 8350 rank 1-3, 7807 rank 3-5 across 4 corpora; c12 failure 55-66%; ablation fixes 44-52 c12 failures with 0 broken each time.

## Answer to the RQ so far (calibrated)
Partly. Statistical audits (even stratified permutation) and interventions DISAGREE in both directions: the feature that looks like a pure lexical confound (23414) is a necessary and sufficient, dose-dependent cause at the last token; 8350 and 7807 are inert at the tested positions. The SAE adds nothing over raw or prompt variables (as in IOI). The contrast with IOI (17491 ablation did nothing) is NOT yet established: the paper ablated at all positions on 45 keys prompts; a same-tooling IOI rerun is needed. Caveats: one layer/SAE/model/task, templated 3-factor task, injected levels (~7-29) far above normal for other centuries, auto-generated labels.

## Pending (priority)
1. IOI cross-check with same tooling: ioi_position_check.py (per-position activations/stats, tests whether 17491 fires at ' keys' vs ' to') and ioi_causal.py (position-specific ablation + injection of 17491; validates baseline against results/predictions.csv). IOI corpus data/tasks.jsonl: 300 unique prompts; keys = object "the keys"; 37/42 keys failures predict "the".
2. Optional: layer sweep (6,8,10); start_cc-only injection on grid; inject 8350 at large scales as a sufficiency check.
3. Write-up + figures (role d plots, heatmap, ablation/injection dose-response, AUC comparison, permutation-null table).

## Cautions
Earlier Claude errors corrected: predicted 23414 would fail ablation (wrong); overstated the leaky-CV flaw (negligible); suspected permutation chance counts (wrong under full shuffle); duplicate-prompt design flaw. Cohen's d inflates for near-silent features. Report both failure definitions.

## UPDATE v4: IOI cross-check (supersedes the "Pending" list above)
- ioi_position_check.py, ioi_causal.py (--positions all|to|obj|the_obj|last3, --mode ablate|add, --random-dir), ioi_summarize.py (offline summary of saved CSVs in results_ioi_pos/). Baseline first-token accuracy 0.813 vs paper 0.797; 295/300 prompt flags agree.
- Position check: 17491 on keys prompts: 38.4 at the ' keys' token (-2), 6.3 at the final ' to' (-1), ~0 at -3..-6; ~0 everywhere on non-keys prompts. All failure-discriminating signal is in the last two tokens (79 and 57 features with |d|>.8; none at -3..-6). Within-keys d (0.45-0.73) is unreliable (3 successes among 45).
- IOI causal (first-token scoring): ablating 17491 at all / obj / to positions = 0 gained, 0 lost (keys accuracy 0.178 unchanged; logit_diff IO-S +1.77 -> +1.77/+1.80); random same-norm directions the same. Injection at the object token (to the keys level, 38.4) into all prompts: 2 gained, 0 lost (p=0.5). So 17491 is inert, unlike GT feature 23414.
- CROSS-TASK CONCLUSION: the audit ranks a lexical token detector first in both tasks, but it is inert in IOI and causal (necessary, sufficient, dose-dependent) in GT; "correlate, not cause" is NOT a general property of the method; statistics alone cannot distinguish them, interventions can.
- LIMITS: IOI keys failures are mostly the model predicting " the" (not name confusion), which logit_diff (IO-S) cannot see; the-gap = logit[" the"] - logit[IO] metric added to ioi_causal.py. Injection so far only at the object token; need injection at both obj and final token (--positions last3 --mode add). Rerun: ablate all with the-gap, add at last3.
- NEXT: finish those two reruns, then write-up and figures (role-d plots, heatmap, GT dose-response, IOI vs GT comparison table, AUC comparison, permutation-null table). Optional: layer sweep.

## UPDATE v5: experiments complete
- IOI the-gap reruns done: ablating 17491 at all positions: keys-prompt the-gap +0.86 -> +0.82 (random directions +0.82..+0.88), accuracy unchanged (0 gained/0 lost). Injecting at obj+to (38.4 and 6.3) into other prompts: the-gap -1.64 -> -1.62 (keys prompts sit at +0.86; keys-vs-other gap ~2.5 logits). So 17491 is neither necessary nor sufficient even on the metric matching IOI's failure mode (" the" continuation). Compare GT 23414: prob_diff +0.46 on century 12 when ablated.
- make_figures.py builds figures_ext/fig1..fig5 from saved CSVs (role signal, GT causal, IOI-vs-GT, AUC baselines, permutation null).
- Claim list: SUPPORTED: audit finds many large-d features; most explained by century x start-year strata; 23414 necessary/sufficient/dose-dependent at final token; IOI 17491 inert; SAE adds nothing over raw/prompt variables. QUALIFY: per-position advantage modest in GT, absent in IOI; "inert" = no detectable single-feature effect at tested position/layer (redundancy possible). AVOID: non-c12 fix claim; 8350 passing confound control; multi-seed as independent replication; generalizing beyond 1 model/SAE/layer/2 templated tasks.
- Optional robustness: joint ablation of top-20 IOI features (redundancy objection); layer sweep; look up Neuronpedia labels for 17491 and 7536.
- NEXT: write-up (abstract, methods, results with fig1-5, limitations, reproducibility section pointing to the commands in this repo).
