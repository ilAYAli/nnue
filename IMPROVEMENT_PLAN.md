# Enyo NNUE Improvement Plan

`README.md` documents how to create a candidate. This file records the current
strategy for producing a stronger net.

Goal: add new signal. Do not keep rerunning the same architecture on the same
kind of Stockfish-labeled Enyo self-play.

## Current State

No trained Enyo net is currently a keeper.

Tooling correction:

- `train_child_ranking.py` now trains child margins in parent POV. The first
  target-only run exposed that the previous loss used child side-to-move POV
  with the wrong sign: loss went down while `.pt`/`.nn` gates got worse.
- `child_rank_engine_gate.py` uses the engine eval path. Current reference
  binaries do not expose `eval2`, so the gate falls back to `evalnet`. A bad
  fallback to plain `eval` made old engine-gate results invalid because it did
  not evaluate the requested child FEN.
- `nnue_event_ntfy.sh` now sends long-run `done` and `fail` events to
  `AI_stdin` by default. Phase spam stays on the normal `nnue` topic.

Latest child-ranking result:

- `child-ranking-a4b4-a4d4-targetonly-lr3e5-e48`: failed one-pair model gate
  after moving in the right direction (`.pt` margin `-269cp`, `.nn` margin
  `-271cp`).
- `child-ranking-a4b4-a4d4-targetonly-lr1e4-e160`: passed the one-pair
  capability proof. `.pt` top1 `1/1` margin `+189cp`, `.nn` top1 `1/1` margin
  `+172cp`, engine gate top1 `1/1`.
- `child-ranking-a4b4-neighbors-targetonly-lr1e4-e160`: passed the original
  seven-neighbor group. `.pt` top1 `1/1` margin `+141cp`, `.nn` top1 `1/1`
  margin `+113cp`, engine gate top1 `1/1`.
- `child-ranking-fourgroup-targetonly-lr1e4-e160`: failed at `3/4`; the
  `b6d7` group exposed a quantization-margin miss.
- `child-ranking-b6d7-neighbors-targetonly-lr1e4-e320`: passed the isolated
  hard group after more epochs. `.nn` margin was only `+0.5cp`, so this is
  barely export-visible.
- `child-ranking-fourgroup-targetonly-lr1e4-e320`: passed `.pt`, `.nn`, and
  engine gates at `4/4`, but broad drift was large (`broad_excess` about
  `313cp`).
- `child-ranking-fourgroup-preserve002-lr1e4-e320`: broad drift improved
  (`broad_excess` about `54cp`) but the hard `b6d7` group failed again. A
  `0.02` broad leash is too strong for this rung.
- `child-ranking-fourgroup-preserve001-lr1e4-e320`: passed `.pt`, `.nn`, and
  engine gates at `4/4` with `broad_excess` about `63cp`. This is the first
  useful child-ranking preserve setting for the small ladder.
- `child-ranking-lossv5-16-preserve001-lr1e4-e320`: passed the next rung.
  Corrected model and engine gates were `13/16`; misses were broad/quiet rows.
  Final training broad excess was about `54cp`.
- `child-ranking-lossv5-64-preserve001-lr1e4-e320`: failed the model and
  corrected engine gates at `33/64`. Diagnosis: the random 64-group sample
  contained too many tiny-gap rows and too many neighbors per group, so hard
  top1 was partly noise.
- `child-ranking-lossv5-signal64-preserve001-lr1e4-e320`: still failed.
  Model gate: `.pt` `42/64`, `.nn` `36/64`; corrected exported-engine gate:
  `36/64`. This means cleaner targets helped but not enough under the `0.01`
  broad leash.
- `child-ranking-lossv5-signal64-targetonly-lr1e4-e320`: also failed. Model
  gate: `.pt` `43/64`, `.nn` `37/64`; corrected exported-engine gate:
  `37/64`. Removing broad preservation was not enough. Category split showed
  broad rows are bad primary ranking targets: `broad_other` hit `3/15` and
  `quiet_broad` hit `8/15`, while conversion/pawn-race/queen-rook mostly
  learned and forcing was partial.
- `child-ranking-lossv5-primary34-targetonly-lr1e4-e320`: failed the focused
  primary target-only gate. Model gate: `.pt` `29/34`, `.nn` `27/34`.
  Conversion, pawn-race, and queen/rook rows all learned; every miss was a
  forcing row. Training pair accuracy was high (`728/748` on the final epoch),
  so the pairwise objective is optimizing easy best-vs-neighbor pairs while the
  group top1 gate still fails.
- `child-ranking-lossv5-primary34-listwise-targetonly-lr1e4-e320`: passed the
  focused primary capability gate. Model gate: `.pt` `33/34`, `.nn` `32/34`;
  corrected exported-engine gate: `32/34`. The remaining misses were forcing
  rows (`a8a7` and `h2h4`). Broad drift was still large (`broad_excess` about
  `256cp`), so this is only a capability proof.

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

Primary lane:

- Train move ranking from child-move groups, not only scalar position eval.
- Each target group must contain one parent FEN, the oracle best move, the
  engine/logged bad move, and a few engine-plausible neighbors.
- Neighbor moves must be moves the engine actually considered or selected, not
  random legal moves.
- Oracle settings and `max_gap_cp` belong in the stored target data so target
  semantics do not change silently between runs.
- Broad preservation is a deadzone/leash, not a normal competing scalar
  objective.
- New candidates must use `./build.py create`; no manual training pipelines.

Secondary lanes:

- Architecture/features are paused until child-ranking can prove or disprove
  exported move-ranking learning on small groups.
- Stronger teacher data is useful only for high-value child groups first:
  disagreement, PV-instability, failure-suite, and high-loss move-choice rows.
- Static MAE/sign remains a rejection filter only.

## Next Concrete Experiment

Run the next child-ranking ladder rung:

1. Listwise diagnostic on the focused 34-group primary set with weak broad
   preservation.
   - Use `targets/child-ranking/losslogs_v5_signal34_primary.jsonl`.
   - It excludes `broad_other` and `quiet_broad` from primary ranking targets.
   - Set `child_loss=listwise` so the loss optimizes group top1 directly.
   - Set `broad_preserve_weight=0.005`.
   - Require `.pt` and `.nn` model gates at least `30/34`.
   - Require corrected engine gate at least `30/34`.
   - Treat `broad_excess <= 80cp` as the first useful drift target for this
     rung. If it passes the move gates but broad drift is still high, increase
     preservation before scaling.
2. If the weak-preserve rung passes:
   - run the corrected engine gate;
   - inspect broad drift from the training log;
   - then scale to the next category-balanced child set.
3. If the weak-preserve rung fails:
   - diagnose whether the two forcing misses reappeared or preservation blocked
     many more groups;
   - do not launch another same-shape run without changing either target
     composition or the preservation weight.

## Candidate Workflow

Normal candidate creation:

```sh
./build.py create -c build.json
```

Current `build.json` intent:

- candidate name: `child-ranking-lossv5-primary34-listwise-preserve005-lr1e4-e320`
- backend: `child-ranking`
- target format: child-move groups with stored capped gaps
- broad-preserve data: existing packed broad data via `pack_dir`
- self-play depth: `12`
- self-play seed: `2026052101`
- skipped opening plies: `8`
- score depth: `16`
- objective: listwise ranking loss plus weak broad deadzone preservation for
  the focused primary 34-group diagnostic
- current ladder target: `targets/child-ranking/losslogs_v5_signal34_primary.jsonl`
  - 34 groups, 235 positive-gap training pairs, selected from loss logs.
  - category balance: forcing `23`, queen/rook endgame `5`, conversion `5`,
    pawn race `1`.
- main knobs:
  - `ranking_weight`
  - `broad_preserve_weight`
  - `broad_deadzone_cp`
  - `rank_margin_cp`
  - `rank_temperature_cp`
  - `min_groups`
  - `min_pairs`

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
- Current status: gate logic exists, but the committed baseline suite/status is
  not yet recorded.
- Blocker before architecture training starts: record the suite path, position
  count, current-reference baseline numbers, and command used to produce them.
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

Paused while child-ranking is still producing useful exported learning signal.

Resume architecture work only if:

- the 16-group child-ranking rung cannot reach `13/16` after miss diagnosis; or
- a larger child-ranking set passes local gates but fails early game smoke.

When resumed, use the normal architecture checklist:

1. Freeze the current reference net, validation commands, and failure-suite
   input.
2. Implement exactly one separability change.
3. Add known-FEN activation checks.
4. Add export/load/roundtrip checks.
5. Benchmark NPS before training; stop if the branch costs more than about
   `3-5%` NPS without optimization.
6. Train through `build.py`, not a manual script.

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
