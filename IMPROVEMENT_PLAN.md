# Enyo NNUE Improvement Plan

This is the active NNUE plan. It should stay concise. Detailed experiment
history belongs in `docs/archive/` or git history, not in this file.

## Current State

No trained Enyo net is currently a keeper.

As of 2026-05-25, no NNUE training process is running. `build.json` is disabled
and records the latest rejected native diagnostic.

The main conclusion is now clear: current scratch/native scalar scaling and
current local search-aware repair do not close the move-choice gap.

Evidence that scratch scale is not enough:

- SF-binpack scratch `4096` superbatches, about `1.07B` presentations:
  `64/232` top1, `56` candidate-better vs `91` reference-better.
- SF-binpack scratch long `12288`, about `3.22B` presentations:
  `77/232` top1, `68` vs `58`, but mate-like tail `-31311cp`.
- SF-binpack scratch long `24576`, about `6.44B` presentations:
  `80/232` top1, `63` vs `60`, but mate-like tail `-31804cp`.
- SF-binpack scratch long `32768`, about `8.59B` presentations:
  regressed to `69/232`, tail still `-31887cp`.
- External 800-target gate stayed flat: scratch `12288` was `387/800`,
  scratch `24576` was `386/800`, reference was `506/800`.

Evidence that the current search-aware objective family is not ready:

- `native-searchaware-lichess5k-mpe-preflight-lr3e6-e48`: `1255/5000`
  top1, below the `1600/5000` exported-model gate.
- `native-searchaware-lichess5k-targetonly-lr1e6-e80`: collapsed from
  `1054/5000` top1 to about `131/5000`.
- `native-searchaware-rankhinge16-audit-lr3e7-e80`: stayed at `4/16` top1.
- `native-reference-distill5k-targetonly-lr1e6-e80`: started below parent
  baseline (`1095/4937` vs parent `1146/4937`) and degraded to `957/4937`.

## Hard Stop

No more scratch scale runs, LR sweeps, WDL retunes, bucket-count scalar runs,
or broad search-aware training until exported one-pair learning is proven.

The next question is not "how many more rows?" It is:

Can the exported network learn one forcing move preference correctly?

## Active Bottleneck

The bottleneck is search-sensitive move ranking through the full
train/export/engine-eval path.

Scalar MAE/sign are now sanity checks only. They are not promotion evidence.

The persistent `-31k cp` mate-like tails are the strongest diagnostic signal.
They suggest one or more of:

- the objective does not strongly reward forcing distinctions.
- export quantization destroys the distinction.
- the representation cannot express the distinction cleanly.
- the training distribution lacks losing-side forcing alternatives.
- search amplifies small eval errors into catastrophic move choices.

## Track Definitions

`nnue_native`:

- Enyo-owned net trained from scratch.
- Long-term provenance and future Elo lane.
- Current role: diagnostic platform, not promotion candidate.

`nnue_reckless`:

- existing-weight-compatible near-term Elo lane.
- paused until there is a new written hypothesis that changes representation in
  a measurable way without being dense/head-only.

## Closed Lanes

Closed unless a new mechanistic hypothesis is written first:

- head-only, output-only, material-head, and float-head fitting.
- sparse/input LR multiplier sweeps from existing exported weights.
- pairwise/local repair loops on the old target construction.
- target-only policy preservation as a candidate recipe.
- scalar child-row blends and same-data search-aware patching.
- bucket-index sweeps without new feature geometry plus parity/NPS proof.
- same-architecture scratch scalar scaling as a promotion lane.
- Reckless deltas that only affect dense/head tensors or a small bucket mask.

## Next Task

Build and run a one-pair exported overfit diagnostic.

Target contents:

- one mate-like or forcing failure position.
- best forcing move.
- actual engine bad move.
- `2-4` plausible legal alternatives.
- one obvious blunder if available.

Pass criteria:

- `.pt` margin: best move at least `+100cp` over the actual bad move.
- exported `.nn` margin: best move at least `+50cp`.
- engine eval ordering matches exported `.nn`.
- exported net-diff is nonzero in the expected tensors.
- broad sanity: no new `>300cp` regression on `50-100` quick positions.

Failure criteria:

- `.pt` cannot learn the pair: objective/target formulation is wrong.
- `.pt` learns but `.nn` does not: export/quantization is the bottleneck.
- `.nn` learns but engine eval disagrees: loader/eval path is wrong.
- pair learns but broad sanity collapses: representation update is too
  destructive.

## Expansion Criteria

Only after the one-pair test passes:

4-pair pass:

- `4/4` correct in exported `.nn`.
- no pair worse than `-25cp` margin.
- no broad sanity collapse.

16-pair pass:

- at least `13/16` correct in exported `.nn`.
- capped aggregate margin positive.
- worst pair no worse than `-100cp`.
- broad sanity still acceptable.

If one-pair passes but `4` or `16` fails, the likely issue is target conflict,
representation capacity, or generalization.

## Tactical Coverage Audit

Run only after the one-pair path is proven.

Audit questions:

- Do mate-like failure structures exist in the broad training data?
- Does the side to move have plausible losing alternatives represented?
- Are legal alternatives scored/ranked, or only the final position eval?
- Which failure mechanisms dominate?

Suggested categories:

- forced win.
- defensive only move.
- perpetual/check motif.
- conversion precision.
- sacrificial attack.
- quiet only move.
- horizon-collapse case.

## Targeted Forcing Dataset

Build only after the one-pair and small-slice diagnostics pass.

Requirements:

- mined from failure-suite/search misses and similar structures.
- legal child move scores, not only root scalar labels.
- deeper Stockfish labels only on the targeted forcing subset.
- low-weight blend with a broad dataset.

Promotion preconditions:

- external move-choice gate improves.
- mate-like tail improves.
- broad scalar sanity does not collapse.
- no SPRT until these gates are clean.

## Gates

Candidate gates, in order:

1. exported `net-diff` matches the intended change.
2. static MAE/sign sanity does not collapse.
3. exported model move-choice gate improves.
4. engine search gate improves, including mate-like and non-mate splits.
5. failure-suite replay has no unacceptable tail regression.
6. SPRT only after all cheap gates are clean.

## Workflow Rules

- Use `./build.py -c build.json` for candidate creation.
- Commit `build.json` with the experiment decision before running it.
- Keep run data under `runs/<run-name>/`.
- Use `nnue_native` for scratch/native work.
- Use `nnue_reckless` for existing-weight work.
- Emit NNUE event notifications for long-running phases.
- Update this file only for durable conclusions or changed next action.
