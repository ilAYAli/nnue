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
- scratch/native training works technically, but current Enyo self-play volume
  has not recovered reference move-choice strength.

Current `build.json` state: approved for a proven-data Bullet preflight in the
`nnue_reckless` lane. It starts from the existing Enyo/Berserk-derived net and
trains directly from the available Stockfish NNUE binpack instead of generating
more Enyo self-play labels.

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

## Active Experiment

`build.json` now defines `bullet-sfbinpack-existing-init-preflight`.

Purpose:

- use a proven external Stockfish NNUE binpack as the next data source.
- keep the near-term lane existing-weight based.
- avoid another Enyo self-play or JSONL relabeling cycle.
- use Bullet directly on SF binpack data, then export normal Enyo `.nn`.

This is a preflight, not a promotion candidate. It must pass `net-diff`, static,
composite search/move, and failure-suite gates before any SPRT.

## Hard Rejections

Do not restart these as near-term Elo lanes:

- another same-architecture Stockfish-d16 Enyo self-play recipe.
- fresh d10/d12/mixed-depth self-play without new signal.
- Lichess blends as the main lever.
- bulk d18/d20 relabeling.
- head-only/output-only LR/objective sweeps.
- material/phase head masks.
- current king-bucket split variants.
- current king-pressure/check-state output bucket variants.
- target-only sparse multiplier sweeps.
- Bullet/Reckless-like scratch checkpoints as a direct replacement candidate.
- native scratch nets trained only on current Enyo self-play labels as promotion
  candidates.

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

1. Run the `build.json` Bullet/SF-binpack preflight in `nnue_reckless`.
2. Gate the exported net; stop immediately if it only improves scalar metrics or
   creates a broad/mate-like tail regression.
3. If the proven-data preflight fails, redesign search-aware target/objective
   around engine-search behavior instead of launching another scalar training
   recipe.
4. Keep native/scratch as a background lane that needs substantially broader
   prepared data before it is considered a promotion path.

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
