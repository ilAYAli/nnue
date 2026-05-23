# Enyo NNUE Improvement Plan

This file is the active working plan. Long experiment notes are archived in
`docs/archive/IMPROVEMENT_HISTORY_20260523.md`.

Goal: produce a stronger Enyo net without repeating already-failed NNUE farming
loops.

## Current State

No trained Enyo net is currently a keeper.

The current Berserk-derived Enyo net appears locally saturated for ordinary
fine-tuning. Static MAE/sign can improve while search behavior and SPRT do not.
The main failure pattern is:

- head/output changes can improve narrow diagnostics, but fail broad gates or
  mate-like tails.
- normal input/L1 fine-tunes receive gradients but usually do not cross exported
  quantization thresholds.
- forced sparse movement is possible, but has not improved engine-search move
  choice.
- scratch/native training works technically. Current Enyo self-play volume has
  not recovered reference move-choice strength, but the first Stockfish-binpack
  native run was materially better than prior Enyo-selfplay scratch attempts.

Current `build.json` state:
`native-searchaware-rootpolarity-exportcheck16-lr3e5-e80`.
This is a search-aware plumbing audit, not a candidate. It exists to prove or
disprove saved/exported target-gate persistence after the child-low audit
failed.

## Track Definitions

`nnue_reckless` is the near-term Elo lane:

- start from existing Enyo/Berserk-derived weights.
- make small existing-weight-compatible architectural/objective changes.
- use Bullet only when it preserves the same lane goal.
- do not turn this into a scratch Reckless-style net.

`nnue_native` is the long-term provenance lane:

- no Berserk initialization.
- train an Enyo-owned net from scratch.
- may borrow architecture ideas, but the result is native to Enyo.
- currently useful for background experiments, not promotion.

## Latest Result

Search-aware target construction and training are implemented on
`feature/nnue-search-aware-targets`:

- target set: `assets/failure_suite/search_aware_targets_20260523.jsonl`.
- size: `232` positions and `5,345` scored legal moves.
- reference-vs-reference search-gate sanity check has zero compare diffs.

Rejected search-aware sparse diagnostics:

- `search-aware-existing-preflight-lr1e7-e2`: exported only dense/output floats;
  search-gate failed (`candidate_better=39`, `reference_better=146`,
  `capped_sum_diff_cp=-17929`).
- `search-aware-sparseprobe-init-in800-l1x1000-e2`: moved sparse floats but
  exported an identical `.nn`.
- `search-aware-sparsecross-init-in3000-l1x3000-e2`: crossed export thresholds
  (`90/25200209` changed) but search-gate was unchanged and still failed.
- `search-aware-targetonly-sparse-overfit-in3000-l1x3000-e8`: moved
  `523527/25200209` exported values and improved internal child-ranking/static
  metrics, but engine-search gate still failed (`candidate_better=41`,
  `reference_better=147`, `capped_sum_diff_cp=-18167`).

Decision: stop search-aware sparse LR/multiplier sweeps. If this lane continues,
change target construction or objective so success is measured by engine-search
move choice, not only child-eval ranking.

## Latest Result: Bullet SF-binpack

`build.json` now defines `bullet-sfbinpack-legacy-floathead-lr1e7-sb1`.

Purpose:

- use a proven external Stockfish NNUE binpack as the next data source.
- keep the near-term lane existing-weight based.
- avoid another Enyo self-play or JSONL relabeling cycle.
- use Bullet directly on SF binpack data after first proving the legacy-layout
  Enyo init export is behaviorally identical to the reference.

Init-export parity passed:

- exported net size matched the current reference exactly (`25203012` bytes).
- all integer tensors matched; only four output floats differed by `1.49e-08`.
- search-gate compare was exactly zero on all/mate-like/non-mate subsets.

Rejected SF-binpack pressure setting:

- `bullet-sfbinpack-legacy-existing-init-preflight`: two-superbatch
  `lr=1e-6 -> 2e-7`, weight decay `1e-6`.
- exported movement: `505` input weights, no L1 movement, dense/output moved.
- search gate failed (`top1=84/232`, `capped_sum_diff_cp=-2749`,
  mate-like worst regression `-31591`).
- checkpoint 1 was also rejected (`top1=87/232`, `capped_sum_diff_cp=-2788`,
  mate-like worst regression `-31591`).

Rejected lower-pressure all-weight setting:

- `bullet-sfbinpack-legacy-lr1e7-sb1-nodecay`: one superbatch,
  `lr=1e-7 -> 2e-8`, no weight decay.
- exported movement still crossed `505` input weights, plus dense/output moved.
- search gate failed (`top1=82/232`, `capped_sum_diff_cp=-928`,
  mate-like worst regression `-31248`).
- non-mate subset was mixed (`candidate_better=32`, `reference_better=26`,
  worst regression `-155`), but top1 still regressed (`49/128` vs `53/128`).

Rejected float-head setting:

- `bullet-sfbinpack-legacy-floathead-lr1e7-sb1`: one superbatch,
  `lr=1e-7 -> 2e-8`, no weight decay, `trainable=float-head`.
- `net-diff` was intentional: sparse/L1 stayed identical; dense/output floats
  moved.
- search gate improved top1 overall (`91/232` vs reference baseline `89/232`)
  and non-mate top1 (`60/128` vs `53/128`), but failed mate-like behavior.
- failure-suite replay rejected it: `positions=913`, `candidate_better=73`,
  `reference_better=68`, `sum_diff_cp=-627`, `worst_regression_cp=-512`.

Scaled float-head deltas were also rejected:

- `25%` delta: search gate looked best (`top1=101/232`,
  `candidate_better=57`, `reference_better=30`,
  `capped_sum_diff_cp=+1301`), but failure-suite tail vetoed it
  (`sum_diff_cp=+1060`, `worst_regression_cp=-511`).
- `20%` delta: search gate stayed positive (`candidate_better=53`,
  `reference_better=33`, `capped_sum_diff_cp=+933`), but failure-suite replay
  rejected it (`sum_diff_cp=-559`, `worst_regression_cp=-512`).
- `5%`, `10%`, `15%`, `50%`, and `75%` deltas did not improve the overall gate
  enough to justify replay or SPRT.

Decision: no SPRT. Stop SF-binpack float-head delta scaling as a near-term Elo
lane. It contains some broad signal, but repeatedly creates or preserves
unacceptable mate/endgame tails.

## Latest Result: Native SF-binpack Continuation

`native-bullet-sfbinpack-continue-eval400-lr2e4-sb4096`

Purpose was to:

- continue the native SF-binpack lane after the first run showed clear
  baseline-building progress but failed promotion gates.
- initialize from the Enyo-owned
  `native-bullet-sfbinpack-scratch-eval400-sb4096/model.nn`.
- keep the data source and architecture fixed.
- use a lower continuation LR and save checkpoints for the same cheap move-gate
  sweep before any broader validation.

This run is a provenance/native-baseline test, not a promotion candidate by
default.

Checkpoint sweep selected `2048` as the least-bad checkpoint, but it still
failed the original search gate:

- `top1=68/232`, `candidate_better=58`, `reference_better=86`,
  `capped_sum_diff_cp=-4925`.
- the old gate showed a repeated `-31814cp` mate-like tail.

The search gate was then fixed to send `ucinewgame`/`isready` before each target
instead of reusing one UCI game/hash across unrelated positions. Fresh-hash
validation for checkpoint `2048` gave:

- `300k nodes`: `top1=65/232`, `candidate_better=39`,
  `reference_better=86`, `capped_sum_diff_cp=-5202`.
- `1M nodes`: `top1=78/232`, `candidate_better=48`,
  `reference_better=73`, `capped_sum_diff_cp=-2436`,
  `worst_regression_cp=-276`.
- `1M mate-like`: `candidate_better=21`, `reference_better=16`,
  `capped_sum_diff_cp=+503`, `worst_regression_cp=-237`.
- `1M non-mate`: `candidate_better=27`, `reference_better=57`,
  `capped_sum_diff_cp=-2939`, `median_nonzero_diff_cp=-25`.

Decision: no SPRT. Stop scalar native SF-binpack continuation. The useful next
diagnostic is non-mate search behavior, not another same-objective continuation
run.

Follow-up non-mate diagnostic:

- `native-nonmate-deep-diagnosis-20260523_055831`
- `3M nodes` on the `128` non-mate targets:
  `top1=42/128`, `candidate_better=21`, `reference_better=60`,
  `capped_sum_diff_cp=-3397`, `median_nonzero_diff_cp=-23`.

Decision: deeper search did not close the non-mate gap. The next native lane
needs a changed target/objective, not more scalar SF-binpack epochs.

## Latest Result: Native Search-Aware Non-Mate

`native-searchaware-nonmate-initdistill-w8-lr1e6-e6`

Purpose was to start from the best Enyo-owned SF-binpack checkpoint and train a
non-mate-weighted search-aware objective with broad init-net distillation as a
guardrail.

Result:

- training finished quickly, but internal target metrics did not improve:
  `target_top1` stayed around `32-33/232`.
- `net-diff`: exported input/L1/output tensors did not move. Only
  `191/512` L2 weights and `24/32` L2 biases changed.
- fresh-hash `300k` search gate failed badly:
  `top1=39/232`, `candidate_better=31`, `reference_better=151`,
  `capped_sum_diff_cp=-20372`.
- `non_mate`: `top1=12/128`, `candidate_better=13`,
  `reference_better=108`, `capped_sum_diff_cp=-15887`.
- `mate_like`: `capped_sum_diff_cp=-4485`, with reopened `-31924cp` tail.

Decision: no SPRT and no broader replay. This is not an Elo candidate and not a
useful native baseline. The next work is to fix search-aware objective behavior
so it can improve training-set move choice and move intended exported tensors;
do not launch another weighted-search config with the current plumbing.

Follow-up search-aware plumbing audits:

- `search-aware-target-overfit-audit-20260523_063642`: 16 non-mate targets,
  quantized target-only, no broad loss. It moved exported input/L1/dense values
  (`183835/25200209` total changed), but only reached `8/16` top1 after 80
  epochs and damaged broad init-net agreement. This proves exported sparse
  movement is possible, but the schedule is not yet a usable candidate recipe.
- `search-aware-4target-forward-audit-20260523_063827`: 4 non-mate targets,
  target-only. Float and quantized paths both overfit from `0/4` to `4/4`.
  This rules out a simple sign/orientation bug in the child-eval target loss.
- `search-aware-16target-quantized-audit-20260523_064051`: 16 non-mate
  targets, quantized target-only. It overfit from `0/16` to `16/16`, but broad
  init-net MAE drifted to about `1653cp`.
- `search-aware-16target-preserve-audit-20260523_064326`: simultaneous broad
  init-net preservation pins the targets. `w=0.05` ended at `1/16` top1 with
  broad MAE `8.10cp`; `w=0.20` ended at `1/16` top1 with broad MAE `28.15cp`.
- `native-searchaware-stagewarmup-audit16-lr3e6-e24`: first staged
  warmup/ramp attempt. It still failed to cross the target-choice boundary
  (`1/16` final top1), ended with broad MAE `39.89cp`, and exported only
  dense/output movement (`165/25200209` total changed; no input/L1 movement).
- `native-searchaware-stagewarmup-strong16-lr3e5-e96`: stronger staged
  warmup/ramp moved exported input/L1 (`169656/25200209` total changed), but
  still ended at `1/16` top1 with broad MAE about `2398cp`. This exposed a
  tooling gap: `search_target_limit=16` selected the first 16 mixed targets,
  not the intended first 16 non-mate targets.
- `native-searchaware-stagewarmup-strong16-lr3e5-e96` tag-filtered rerun:
  correctly used the first 16 non-mate targets and moved exported sparse/L1
  heavily (`2802198/25200209` total changed), but ended at `0/16` top1,
  `1/16` top3, and broad MAE about `150cp`. The schedule is destructive:
  sparse movement alone is not useful if it does not learn the search targets.
- `native-searchaware-recover16-w002-lr1e6-e80`: initialized from the previous
  target-fit checkpoint and distilled broad rows against the separate native
  checkpoint. It recovered broad MAE from about `2032cp` to `597cp`, but target
  choice stayed at `0/16` top1 and `1/16` top3. Direct reload checks showed the
  supposed target-fit checkpoint itself now evaluates as `0/16` from exported
  `.nn` and `0/16` from saved `.pt` quantized forward on the same targets.
- `native-searchaware-targetonly-exportcheck16-lr3e5-e240`: failed the
  required saved/exported target gate. Training stayed at `0/16` top1,
  `1/16` top3, `sum_gap_cp=1926`, `worst_gap_cp=544`; saved `.pt` and exported
  `.nn` matched that failure. The first 16 non-mate target IDs and first 8
  moves matched the earlier audit, so this was not a target-file mismatch.
- Reloading the old `search-aware-16target-quantized-audit-20260523_064051`
  artifacts also gives `0/16`, despite the old train log claiming `16/16`.
  Treat that old success as non-persistent and unusable.

Decision: direct child-low search-aware training is not a valid candidate lane.
The next audit uses `search_score_mode=root-high` because a manual diagnostic
can overfit the same 16 targets under root-pov high-is-good scoring. Passing
that audit would only prove target-training/export plumbing; it would not make
root-high child-position training semantically correct for Elo.

## Latest Result: Native SF-binpack Scratch

`native-bullet-sfbinpack-scratch-eval400-sb4096`

Completed with Bullet directly on the Stockfish NNUE binpack:

- no Berserk initialization.
- final training loss: about `0.0266`.
- training time: `0h 41m 47s`.
- checkpoints: `1024`, `2048`, `3072`, `4096`.

Checkpoint search-gate sweep:

- `1024`: `top1=62/232`, `candidate_better=46`,
  `reference_better=89`, `capped_sum_diff_cp=-5341`.
- `2048`: `top1=56/232`, `candidate_better=51`,
  `reference_better=95`, `capped_sum_diff_cp=-6724`.
- `3072`: `top1=64/232`, `candidate_better=53`,
  `reference_better=93`, `capped_sum_diff_cp=-5359`.
- `4096`: `top1=64/232`, `candidate_better=56`,
  `reference_better=91`, `capped_sum_diff_cp=-4663`.

Decision: no SPRT. This is not close to the reference, and every checkpoint
keeps the `-31814cp` mate-like tail. It is still materially better than prior
native Enyo-selfplay scratch checkpoints, so continue once with lower LR from
this Enyo-owned checkpoint before changing architecture or data again.

## Hard Rejections

Do not restart these as near-term Elo lanes:

- another same-architecture Stockfish-d16 Enyo self-play recipe.
- fresh d10/d12/mixed-depth self-play without new signal.
- Lichess blends as the main lever.
- bulk d18/d20 relabeling.
- head-only/output-only LR/objective sweeps.
- material/phase head masks.
- raw SF-binpack float-head update without tail scaling.
- SF-binpack float-head delta scaling.
- current king-bucket split variants.
- current king-pressure/check-state output bucket variants.
- target-only sparse multiplier sweeps.
- Bullet/Reckless-like scratch checkpoints as a direct replacement candidate.
- native scratch nets trained only on current Enyo self-play labels as promotion
  candidates.
- `native-bullet-sfbinpack-scratch-eval400-sb4096` as a promotion candidate:
  best checkpoint was still `top1=64/232`, `candidate_better=56`,
  `reference_better=91`, `capped_sum_diff_cp=-4663`.
- `native-bullet-sfbinpack-continue-eval400-lr2e4-sb4096` as a promotion
  candidate: fresh-hash 1M validation improved mate-like behavior, but non-mate
  stayed strongly negative (`candidate_better=27`, `reference_better=57`,
  `capped_sum_diff_cp=-2939`).
- scalar native SF-binpack continuation at deeper search: the `3M` non-mate
  diagnostic stayed negative (`candidate_better=21`, `reference_better=60`,
  `capped_sum_diff_cp=-3397`).
- `native-searchaware-nonmate-initdistill-w8-lr1e6-e6`: only L2 exported floats
  moved, internal target choice did not improve, and fresh-hash search gate
  collapsed (`capped_sum_diff_cp=-20372`).
- `search-aware-target-overfit-audit-20260523_063642` as a candidate recipe:
  it can move exported sparse/L1 tensors, but only reached `8/16` training-set
  top1 and caused very large broad init-net drift.
- simultaneous search-aware broad preservation as a candidate recipe:
  `search-aware-16target-preserve-audit-20260523_064326` kept broad drift small
  but did not cross the target-choice boundary.
- weak staged search-aware warmup as a candidate recipe:
  `native-searchaware-stagewarmup-audit16-lr3e6-e24` still moved only
  dense/output floats and ended at `1/16` top1.
- mixed-target strong staged search-aware warmup as a candidate recipe:
  `native-searchaware-stagewarmup-strong16-lr3e5-e96` moved sparse/L1 but
  targeted the wrong mixed 16-row subset and still ended at `1/16` top1.
- tag-filtered strong staged search-aware warmup as a candidate recipe:
  `native-searchaware-stagewarmup-strong16-lr3e5-e96` moved sparse/L1 heavily,
  but ended at `0/16` top1 on the intended non-mate subset.
- `native-searchaware-recover16-w002-lr1e6-e80` as a candidate recipe:
  broad behavior recovered, but target top1 stayed `0/16`; the input
  target-fit checkpoint was not export-persistent.
- `native-searchaware-targetonly-exportcheck16-lr3e5-e240` as a candidate
  recipe: the child-low target-only objective failed even the 16-target
  saved/exported gate.
- search-aware mateguard with broad init distillation and `mate_like=8`: it
  exported only dense/output changes and failed the search gate
  (`candidate_better=39`, `reference_better=146`,
  `capped_sum_diff_cp=-17929`).

## Promotion Gates

A candidate must pass cheap gates before match testing:

1. `net-diff`: exported movement must match the intended change.
2. Static validation: no clear sign regression; MAE/sign are rejection filters,
   not proof.
3. Search/move gate: improve top1/top3 or capped gap against reference without
   losing mate-like or non-mate subsets.
4. Failure-suite replay: no unacceptable tail regression.
5. SPRT only after the above are clean.

A narrow positive gate does not justify SPRT if a broader gate or mate-like tail
is negative.

## Next Decision

The next action should be one of these, in order:

1. Run `native-searchaware-rootpolarity-exportcheck16-lr3e5-e80` from
   `build.json`. This is an export-persistence and polarity audit, not a
   candidate.
2. If it fails, stop direct child-eval target training. The search-aware lane
   needs a different target/objective measured by actual engine-search behavior.
3. If it passes, record that target-training/export persistence works under
   root-high scoring, then decide whether the objective can be made
   semantically valid before any recovery or scale-up run.
4. No SPRT from the current search-aware runs.

Do not launch another training run until `build.json` names the lane,
hypothesis, data source, and gates.

## Workflow Rules

- Use `./build.py -c build.json` for candidate creation.
- Commit `build.json` with the experiment decision before running it.
- Keep run data under `runs/<run-name>/`.
- Use `nnue_reckless` for near-term existing-weight work.
- Use `nnue_native` for scratch/native work.
- Emit NNUE event notifications for long-running phases.
- Update this file only for durable conclusions or a changed next action.
- Put long evidence dumps in `docs/archive/`, not in this active plan.
