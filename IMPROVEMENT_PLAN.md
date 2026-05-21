# Enyo NNUE Improvement Plan

`README.md` documents how to create a candidate. This file records the current
strategy for producing a stronger net.

Goal: add new signal. Do not keep rerunning the same architecture on the same
kind of Stockfish-labeled Enyo self-play.

## Current State

No trained Enyo net is currently a keeper.

Rejected lanes:

- d16/d18 relabeling of old/self-play pools: static metrics improved, SPRT did
  not confirm.
- fresh d10/d12 self-play: neutral or negative after smoke/screen tests.
- Lichess blends: sometimes improved static or early smoke direction, but did
  not hold up in longer SPRT.
- mixed-depth self-play, d12/d8/d6 with Stockfish d16 labels: only `0.4%`
  exact overlap with the old d12 pool, improved static MAE, but smoked at
  `-0.7 +/- 15.0`.
- hardcase/failure-suite fine-tunes: moved some target positions, but produced
  unacceptable tail regressions.
- old-pool instability/disagreement blends: diagnostically useful, but not
  enough for SPRT promotion.
- learned material/phase head input:
  - all-weights training improved static MAE slightly, but failed the
    failure-suite gate: `candidate_better=64`, `reference_better=55`,
    `sum_diff_cp=+561`, `worst_regression_cp=-563`.
  - float-head-only training improved scalar MAE much more, but regressed
    move choice: `candidate_better=70`, `reference_better=74`,
    `sum_diff_cp=-31`, `worst_regression_cp=-563`.
  - phase-column-only training was behaviorally identical to the reference:
    `candidate_better=0`, `reference_better=0`, `sum_diff_cp=0`.
  - Conclusion: this head-level material/phase signal is either harmful or
    too weak/no-op in the current architecture.
  - Process lesson: the phase-column-only no-op should have been caught by a
    known-FEN activation and export-delta check before training. The all-weights
    run changed move choices, so the feature path was not completely dead, but
    future architecture branches must prove the feature affects exported evals
    before spending training time.
- aggregate-positive/tail-negative experiments are a repeated failure mode:
  material/phase all-weights and hardcase fine-tunes both improved some
  positions while introducing unacceptable worst-case regressions. Tail risk is
  now a hard veto, not just a note.
- folded 8-king-bucket shortcut: clearly negative as a drop-in net.
- thread voting/arbitration search experiments: clearly negative in early SPRT.

Conclusion:

- The current architecture/training regime appears locally saturated.
- Further gains from relabeling or self-play refresh alone are expected to be
  low.
- Static MAE/sign is now only a rejection filter.
- Novel Enyo self-play alone was not enough.
- Do not launch another same-architecture Stockfish-d16-labeled Enyo self-play
  candidate unless a move-choice/failure-suite gate gives a concrete reason.

## Current Strategy

Priority order:

1. Architecture/features.
   - This is the primary lane.
   - First branch, learned material/phase head input, failed the pre-SPRT
     gates and should not be SPRT-tested.
   - Next branch: proper king-bucket refinement with full trainer/engine
     support, not another folded/drop-in shortcut.
   - Verify feature extraction, export/load, and roundtrip before training.
   - Verify at least one known FEN where the new feature changes the exported
     eval before training.
   - Benchmark NPS before training; pause and optimize first if NPS drops more
     than about `3-5%`.
   - If NPS loss is above that threshold, require much stronger pre-SPRT
     evidence before spending games.
   - Train the changed architecture properly. Do not treat folded/drop-in
     conversions as evidence.
   - Do not widen the net until at least one small feature/bucket experiment
     has failed cleanly.
   - Material/phase has now failed, so widening is allowed only as a
     fallback-of-last-resort after king-bucket failure analysis, not as the next
     default move.

2. Stronger or different teacher data.
   - Treat Stockfish d16 as the bulk baseline, not the ceiling.
   - Test d18/d20 only on high-value slices first: disagreement,
     PV-instability, failure-suite, and high-loss move-choice rows.
   - Do not spend a full bulk d20 label run unless a small slice improves
     move-choice gates, not just MAE.
   - External/prepared datasets are acceptable if converted once into the Enyo
     row format and stored with provenance under `runs/` or `assets/`.

3. Targeted move-choice data.
   - Expand the fixed failure-suite and disagreement/PV-instability samplers.
   - Train at most one isolated candidate from this signal at a time.
   - Tail regressions can veto a candidate even when aggregate sum diff is
     positive.
   - Longer-term goal: optimize search decision quality, not only scalar
     evaluation accuracy.
   - Search-aware signals to track before training from them:
     - top-move agreement.
     - top-3 move overlap.
     - eval ranking consistency for candidate moves.
     - disagreement/PV-instability weighting.
     - tactical surprise or large child-eval swing weighting.

4. Tooling.
   - Tooling work is justified only when it directly supports the lanes above.
   - New candidates must use `./build.py create`.
   - The reviewed active recipe lives in `build.json` and should be updated in
     the same commit as the experiment decision.
   - Manual step-by-step pipelines are historical/legacy only.
   - Planned recipes should be concrete `build.py create` commands, not prose.

## Next Concrete Experiment

Run exactly one architecture/feature branch first.

Next branch:

- proper king-bucket refinement.

Reason:

- materially changes the representation instead of adding another scalar head
  hint.
- directly targets the likely weakness: king-local piece-square context under
  search.
- must be trained properly with matching engine/trainer feature extraction.
- previous folded/drop-in bucket shortcut is not evidence against a real
  retrain.

Anti-confounding rule:

- Do not change architecture and data source in the same first candidate.
- Reuse the best-understood training source for the first architecture test:
  the current signed-balanced d12 self-play plus Stockfish-d16 labels.
- Use `build.py --labeled-jsonl` so the first architecture test repacks the
  existing labels with the new feature map instead of generating fresh
  self-play or relabeling.
- Because this moves the recipe through the new `build.py` pack/train path,
  run the pack/static/roundtrip sanity checks before training starts.

Fallback:

- If proper king-bucket refinement also fails gates, stop bulk training and
  reassess base net, feature family, and teacher/source assumptions.

## Candidate Workflow

Normal candidate creation:

```sh
./build.py -c build.json
```

Current `build.json` intent:

- candidate name: `arch-kingbucket-v1`
- selected branch: proper king-bucket refinement
- self-play depth: `12`
- self-play seed: `2026052101`
- skipped opening plies: `8`
- labeled input: existing imported `fresh_d12self18h64_d16_labels` JSONL
- label provenance: Stockfish depth `16`; `build.py` skips scoring because
  `labeled_jsonl` is set.
- objective: Huber, clamp `800`, beta `200`, lr `7e-7`, epochs `8`
- checkpoint selection: `sign`, patience `2`

Rules:

- Commit `build.json` changes so the current intended run is reviewable.
- Always record `--selfplay-seed`.
- Treat `--skip-plies` as an opening-distribution knob.
- Use `--select-metric` and `--patience`; do not blindly export the final epoch.
- Use `--trainable float-head` or `--trainable output` only for quick probes.
  Keeper attempts normally train all weights.
- Keep new run data under `runs/<run-name>/`.
- Do not assume old manual packed data and new `build.py` packed data are
  interchangeable until a roundtrip/static sanity check confirms it.

## Gates

Static validation:

- MAE/sign improvements are not success.
- Static validation can reject obviously bad candidates.
- A candidate still needs move-choice gates and SPRT.

Move-choice/failure-suite gate:

- Use candidate/reference/oracle replay CSV where possible.
- Current status: baseline is recorded in
  `assets/failure_suite/baseline_reference_b19794a.md`.
- Baseline: `913` positions, same-reference run,
  `candidate_better=0`, `reference_better=0`, `sum_diff_cp=0`,
  `worst_regression_cp=0`.
- Before architecture training starts, run the branch-specific feature
  activation and roundtrip checks. Do not skip them because the baseline exists.
- Track move-choice correlation metrics, not only scalar eval deltas:
  - top-move agreement.
  - top-3 overlap.
  - eval ranking consistency across legal or candidate moves.
  - near-threshold instability where several moves are close.
- Required direction: candidate better count should exceed reference better
  count, aggregate cp diff should be positive, and tail regressions must be
  controlled.
- Provisional numeric heuristics:
  - `candidate_better >= reference_better * 1.05`
  - `sum_diff_cp > +1000`
  - `worst_regression_cp > -250`
  - no new tactical regression worse than `-300cp`
- These numbers are starting heuristics, not law. The principle is the
  important part: do not spend SPRT on nets with ugly tails.
- A candidate that fails these gates can still be kept as diagnostic evidence,
  but it must not consume long SPRT time.

Failure taxonomy:

- Tag failure-suite positions by category so improvements/regressions are not
  treated as one flat bucket.
- Suggested initial tags:
  - tactic.
  - king attack.
  - fortress.
  - conversion.
  - pawn race.
  - zugzwang.
  - space.
  - initiative.
  - imbalance.
  - quiet maneuver.
- Every candidate should report deltas by category once the taxonomy exists.

SPRT:

- A fixed 1000-game smoke is only a cheap rejection screen.
- Attractive 1000-game smokes like `+7` or `+14.9` were statistical mirages
  until longer runs contradicted them.
- Prefer tight smoke bounds such as `elo0=0`, `elo1=5` where the runner
  supports it.
- `+3..+6 Elo` with wide CI is not enough to extend.
- A `+10 Elo` smoke is direction only, not proof.
- Promote only after a longer screen confirms the signal.

## Architecture Sequence

Use this sequence for the next serious attempt:

1. Freeze the current reference net, validation commands, and failure-suite
   input.
2. Implement exactly one branch: proper king-bucket refinement.
3. Add known-FEN feature activation checks:
   `tools/validate/<branch>_features.py`.
   Required cases:
   - both kings in each bucket.
   - castling before and after.
   - mirrored positions.
   - king near a bucket boundary.
   - quiet non-king move should not change the king bucket.
   - king move across a boundary should change only the expected bucket and
     trigger only the expected accumulator refresh.
4. Add export/load/roundtrip checks:
   `tools/validate/roundtrip.py`.
5. Benchmark NPS before training; do not continue if the branch costs more
   than about `3-5%` NPS without optimization.
6. Train one candidate with `build.py`.
7. Run static validation plus failure-suite/move-choice gates.
8. Start SPRT only if gates are clean.

If this architecture branch fails gates or SPRT:

- Do not launch another bulk candidate immediately.
- First inspect whether the failure came from implementation, NPS cost, sparse
  buckets, bad bucket geometry, quantization/export mismatch, or true lack of
  signal.
- If that analysis still points to true lack of signal, reassess base net,
  architecture family, and teacher source before widening the net.

## Historical Notes

Important failed signals:

- d18 conservative `huber_cp1000_lr5e7_e4` looked promising at 1000 games
  (`+7.0 +/- 15.1`) but collapsed in the add-on run (`-1.7 +/- 9.8` at
  2302/3000).
- cp800 neighbor `cp800_lr7e7_e8` reached `+14.9 +/- 14.7` in smoke, then a
  follow-up restarted from zero and quickly went negative.
- Fresh d12 + 20% Lichess MPE reached `+8.0 +/- 15.2` in smoke but failed the
  4000-game screen at `-0.3 +/- 7.6`.
- Mixed-depth self-play was genuinely novel by exact-FEN overlap, but still
  produced no Elo.

Legacy/manual runs:

- Several historical runs were produced by manual scripts before the current
  `build.py` pipeline.
- Treat those runs as historical data.
- New candidates should use `build.py` so `config.json`, `manifest.json`,
  `status.json`, `events.jsonl`, resumability, and event notifications are
  present.

## Do Not Do

- Do not run another matrix of tiny LR/objective variants on the same data.
- Do not bulk-label d20 because it sounds stronger.
- Do not promote from MAE/sign alone.
- Do not extend a noisy positive smoke without clean move-choice gates.
- Do not return to fresh self-play as the main lever unless paired with a new
  architecture, stronger teacher source, or a gate showing it solves a concrete
  failure mode.
