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
- `child-ranking-lossv5-primary34-listwise-preserve005-lr1e4-e320`: passed
  with weak broad preservation. Model gate: `.pt` `33/34`, `.nn` `31/34`;
  corrected exported-engine gate: `31/34`. Broad drift stayed controlled
  (`broad_excess` about `53cp`). This is the first rung where the listwise
  objective learns and broad preservation does not immediately block it.
- `child-ranking-lossv5-primary80-listwise-preserve005-lr1e4-e320`: failed
  after scaling too broadly. Model gate: `.pt` `55/80`, `.nn` `47/80`;
  corrected exported-engine gate also `47/80`. Broad drift stayed acceptable
  (`broad_excess` about `61cp`), so preservation was not the blocker. The
  target set mixed in 16 tiny-gap rows (`<30cp`) and too many hard forcing rows
  at once.
- `child-ranking-lossv5-primary64-g30-listwise-preserve005-lr1e4-e480`:
  passed the filtered high-signal rung. Model gate: `.pt` `61/64`, `.nn`
  `55/64`; corrected exported-engine gate: `56/64`. Broad drift stayed
  controlled (`broad_excess` about `56cp`). Cross-checks on the completed net:
  engine gate `28/34` on the old focused set and `60/80` on the full primary80
  set. The full-set misses are dominated by tiny-gap rows, so do not chase
  gap-1 to gap-8 rows as primary targets.
- `child-ranking-lossv5-primary64-g30-listwise-targetonly-lr1e4-e480`: failed
  the model gate. Float `.pt` reached `62/64`, but exported `.nn` stayed at
  `55/64`, essentially identical to the weak-preserve run, while broad drift
  grew to about `208cp`. This closes the "preservation is blocking the
  remaining rows" hypothesis for this target set. The active blocker is
  export-visible movement / quantization.
- `child-ranking-lossv5-primary64-g30-listwise-qfwd-preserve005-lr1e4-e480`:
  passed and fixed the export gap. Export-quantized `.pt`, exported `.nn`, and
  corrected engine gate all reached `62/64`; broad drift stayed controlled
  (`broad_excess` about `59cp`). Cross-checks: old focused set `30/34`, full
  primary80 `66/80`, and `.pt == .nn` on primary80. The two trained-set misses
  had margins only around `-2cp` and `-1cp`, so the next check should push
  margin, not change target source.
- `child-ranking-lossv5-primary64-g30-listwise-qfwd-preserve005-t15-lr1e4-e800`:
  passed the sharper margin check. Export-quantized `.pt` and exported `.nn`
  both reached `64/64`; corrected engine gate also reached `64/64` with worst
  margin `+28cp`. Broad drift stayed controlled (`broad_excess` about `51cp`).
  Cross-checks: old focused set `32/34`, full primary80 `68/80`. This proves
  export-aware child ranking can fully learn the filtered 64-group set; the
  next useful test is scale, not another 64-set margin tweak.
- `child-ranking-lossv5-primary80-listwise-qfwd-preserve005-t15-lr1e4-e800`:
  passed the primary80 scale check. Export-quantized `.pt`, exported `.nn`, and
  corrected engine gate all reached `73/80`; final broad drift stayed
  controlled (`broad_excess` about `61cp`). Primary80 baseline was only `41/80`.
  Non-training cross-checks also improved versus baseline: signal64 `29/64` to
  `42/64`, random losslogs64 `27/64` to `32/64`, and random worst regression
  improved from `-1325cp` to `-210cp`. This is enough local signal for an early
  game smoke, but not enough to call it a keeper.
- The early 256-game smoke for that net was aborted as a clear reject at
  `86/256`: about `-772 Elo`, `0.0%` LOS, `2.3%` draws. The engine binary and
  net loading were not the issue. The important diagnosis is that the old
  child-ranking broad leash was label-anchored, so it improved Stockfish-label
  MAE while failing to preserve the reference net's search surface. Static
  validation reflected this: the candidate had much better MAE but worse sign
  than the reference, and played catastrophically.
- `child-ranking-lossv5-primary80-listwise-qfwd-refpreserve02-t15-lr1e4-e800`:
  passed local gates with reference-anchored broad preservation: `.pt`, `.nn`,
  and engine all reached `71/80`. This fixed the label-MAE collapse but not
  game safety. On 100k broad rows it had `mae=169.45`, close to reference
  `169.91`, but sign was still much worse: `86.70%` versus reference `90.52%`.
  The near-zero bucket was the main damage (`69.39%` versus `77.98%`). A
  64-game smoke was a hard reject: `-325.9 +/- 107.6`, `0.0%` LOS, `17.2%`
  draws. This shows a `40cp` reference deadzone still permits search-destroying
  sign/order drift.
- `child-ranking-lossv5-primary80-listwise-qfwd-refpreserve05-dz0-t15-lr1e4-e800`:
  was the strict reference-preserve safety check. It passed only the relaxed
  local gate: `.pt/.nn` `67/80`, engine `68/80`, worst engine margin `-82cp`.
  Static broad behavior improved versus the previous reference-anchored run but
  was still materially worse than reference: sign `88.62%` versus `90.52%`;
  near-zero bucket `73.40%` versus `77.98%`. No smoke was run. This closes
  scalar child-ranking as a near-term promotion lane: preserving the scalar
  reference eval tightly enough blocks target learning, while enough target
  learning damages broad search behavior.
- `policy-ranker-primary80-v1-lr1e3-e2000`: first sidecar diagnostic. On the
  primary80 gate, base was `41/80`; policy-only reached `69/80`. Gated
  threshold `16` gave `17` overrides, all good, `0` bad. However, transfer was
  poor on random `losslogs_v5_64`: at threshold `16`, only `2` good overrides
  and `8` bad. This proves the sidecar can learn useful corrections, but the
  primary-only target mix is unsafe; the next run must include random/broad
  "do not override" groups.
- `policy-ranker-mix144-h64-lr5e4-e2500`: trained on primary80 plus random
  `losslogs_v5_64`. The combined gate found one clean threshold:
  threshold `48` had `11` good overrides, `0` bad, selected top1 `79/144`.
  Cross-target transfer was still weak. On the full 773-group losslogs set,
  both the primary80-only and mix144 models produced almost no safe action
  (`2` and `1` zero-bad overrides respectively). This makes the small-rung
  policy models diagnostic only; the next run must train on the full loss-log
  corpus.
- `policy-ranker-full773-h64-lr2e4-e3000`: trained directly on the full
  773-group losslogs corpus. Policy-only reached `428/773`, but validation
  stayed flat around `29-38/193` while train rose to about `400/580`. The final
  gate failed: threshold `32` improved to `237/773` but had `7` bad overrides;
  the only zero-bad threshold with action was threshold `96`, with just `3`
  good overrides and selected top1 `204/773`, below the required `220/773`.
  This closes the compact policy feature set. It can memorize corrections, but
  does not generalize.
- `policy-ranker-full773-board-h128-d15-lr2e4-e3000`: added parent and child
  board planes. The full-corpus gate passed: threshold `32` selected `245/773`
  versus base `201/773`, with `48` overrides, `47` good, and `0` bad. A
  train/validation split audit showed this was still mostly memorization:
  train had `46` good zero-bad overrides at threshold `32`, while validation
  had only `1` good zero-bad override and no top1 lift. The policy gate is now
  being made split-aware so a full-corpus pass cannot hide train-only action.
  Next policy runs must show same-threshold held-out overrides, not just full
  target-set improvement.

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

Move the child-ranking signal out of the scalar eval net:

1. Keep the current reference `.nn` unchanged.
2. Train a separate move-ranking/correction model on the same child groups.
3. Use it only as a gated move-order/tie-break signal in validation, not as an
   eval replacement.
4. Require broad game-safety before considering engine integration.

Interpretation:

- First diagnostic must answer whether the separate ranker can improve the
  primary80 selected moves while leaving the base eval untouched.
- If that works locally, test game action rate and harm rate with an offline
  gate before touching search.
- If it cannot produce clean high-confidence corrections, move to architecture
  or feature work rather than more scalar fine-tuning.

## Candidate Workflow

Normal candidate creation:

```sh
./build.py create -c build.json
```

Current `build.json` intent:

- candidate name: `policy-ranker-lossv5-v3x6-lowmat-board-h64-d25-lr3e4-e120`
- backend: `policy-ranking`
- target format: child-move groups with stored capped gaps
- base net: current reference `.nn`; scalar eval is unchanged
- self-play depth: `12`
- self-play seed: `2026052101`
- skipped opening plies: `8`
- score depth: `16`
- objective: separate move-ranking sidecar, not scalar eval fine-tuning
- current ladder target:
  - `targets/child-ranking/losslogs_v5_full773.jsonl`.
  - plus the generated v3/low-material policy mix from
    `nnue_native_hidden/runs/policy-mix-v3x6-lowmat5k-20260527`.
  - gate is deliberately stricter than the previous full-corpus gate: require
    a single zero-bad threshold that also produces at least 100 held-out good
    overrides and 100 held-out overrides.
- main knobs:
  - `policy_hidden`
  - `policy_feature_set`
  - `policy_dropout`
  - `policy_val_fraction`
  - `policy_target_temperature_cp`
  - `policy_thresholds`
  - `policy_gate_min_top1`
  - `policy_gate_max_bad`
  - `rank_temperature_cp`
  - `min_groups`

Current hypothesis:

- Board geometry is necessary but not sufficient. On 773 groups, it mostly
  memorized train rows. The next test is whether the same board-aware sidecar
  can produce held-out safe action when trained on a much broader real-game /
  low-material policy corpus. If held-out action remains flat, pause simple
  sidecar MLP work and move to either a larger policy corpus with better
  feature batching or a different representation.

Rules:

- Commit `build.json` changes so the current intended run is reviewable.
- Always record `--selfplay-seed`.
- Treat `--skip-plies` as an opening-distribution knob.
- Use `--select-metric` and `--patience`; do not blindly export the final epoch.
- Use `--trainable float-head` or `--trainable output` only for quick probes.
  Keeper attempts normally train all weights.
- For `policy-ranking`, there is no exported `.nn` keeper yet. The first
  success criterion is an offline high-confidence override gate with useful
  action rate and low harm; engine integration is a separate later change.
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
