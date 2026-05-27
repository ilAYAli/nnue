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

Run the child-ranking ladder:

1. One target group.
   - Exported `.nn` must rank the best child above the bad/neighbor children.
   - `.pt` and `.nn` gates must agree.
   - Broad deadzone excess must remain small.
2. Four target groups.
   - All groups should pass exported child ranking.
   - No broad-preserve collapse.
3. Sixteen target groups.
   - At least `13/16` groups should pass exported child ranking.
   - Worst margin and broad-preserve rows must remain acceptable.

Only after the ladder passes:

- build a larger loss-log/failure-suite child target set;
- run replay/failure-suite gates;
- run a 200-300 game smoke before any full SPRT.

## Candidate Workflow

Normal candidate creation:

```sh
./build.py -c build.json
```

Current `build.json` intent:

- candidate name: `child-ranking-a4b4-a4d4-targetonly-lr3e5-e48`
- backend: `child-ranking`
- target format: child-move groups with stored capped gaps
- broad-preserve data: existing packed broad data via `pack_dir`
- self-play depth: `12`
- self-play seed: `2026052101`
- skipped opening plies: `8`
- score depth: `16`
- objective: target-only ranking loss for the one-pair capability diagnostic
- first ladder target: `targets/child-ranking/one_pair.jsonl`
  - current contents are intentionally reduced to `a4b4` vs `a4d4`
    only; re-add neighbors only after this exported one-pair gate passes.
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

Use this sequence for the next serious attempt:

1. Freeze the current reference net, validation commands, and failure-suite
   input.
2. Implement exactly one branch: learned material/phase head input.
3. Add known-FEN feature activation checks.
4. Add export/load/roundtrip checks.
5. Benchmark NPS before training; do not continue if the branch costs more
   than about `3-5%` NPS without optimization.
6. Train one candidate with `build.py`.
7. Run static validation plus failure-suite/move-choice gates.
8. Start SPRT only if gates are clean.

If this architecture branch fails gates or SPRT:

- Try at most one more independent small architecture branch before reassessing.
- The next best candidate is king-bucket refinement with full trainer/engine
  support, not a folded conversion.
- If two independent architecture branches fail, stop spending bulk GPU/search
  time and reassess base net, architecture family, and teacher source.

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
