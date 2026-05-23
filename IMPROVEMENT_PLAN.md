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

Current `build.json` state: active next run is
`native-bullet-sfbinpack-continue-eval400-lr2e4-sb4096` in the `nnue_native`
lane. It continues from the Enyo-owned SF-binpack scratch net, not from Berserk.

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

## Active Next Run

`native-bullet-sfbinpack-continue-eval400-lr2e4-sb4096`

Purpose:

- continue the native SF-binpack lane after the first run showed clear
  baseline-building progress but failed promotion gates.
- initialize from the Enyo-owned
  `native-bullet-sfbinpack-scratch-eval400-sb4096/model.nn`.
- keep the data source and architecture fixed.
- use a lower continuation LR and save checkpoints for the same cheap move-gate
  sweep before any broader validation.

This run is a provenance/native-baseline test, not a promotion candidate by
default.

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

1. Run the committed native SF-binpack scratch preflight from `build.json`.
2. Sweep saved checkpoints against the search/move gate.
3. Continue once from the Enyo-owned SF-binpack checkpoint with lower LR because
   it is materially closer than prior Enyo-selfplay scratch nets.
4. If the continuation still does not materially close the search-gate gap, stop
   native scalar SF-binpack continuation and reassess data/objective before
   spending more GPU time.

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
