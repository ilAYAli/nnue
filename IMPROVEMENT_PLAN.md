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
- proper 32-king-bucket v1:
  - trained on the existing d12/self-play Stockfish-d16 labels through
    `build.py -c build.json`.
  - Static validation moved slightly in the right direction, but sign was flat:
    candidate `mae=135.262`, `sign=92.18%`; expanded reference `mae=135.948`,
    `sign=92.20%`.
  - Failure-suite gate was aggregate-positive but tail-negative:
    `candidate_better=68`, `reference_better=54`, `sum_diff_cp=+1491`,
    `worst_regression_cp=-448`.
  - Expanded-legacy sanity check was clean: the legacy 16-bucket net expanded
    to 32 buckets produced zero replay-gate differences against the legacy net
    over the same 913 positions. That points to the training result, not the
    feature-map conversion, as the source of the tail regressions.
  - Decision: no SPRT. The tail regression violates the gate.
- proper 32-king-bucket input-only diagnostic:
  - trained only input feature rows plus accumulator bias, keeping L1/L2/output
    fixed.
  - Static validation was exactly equal to the expanded reference:
    candidate `mae=136.060`, `sign=92.20%`; reference `mae=136.060`,
    `sign=92.20%`.
  - Exported tensor diff was zero for every weight and bias. The small update
    did not survive quantization/export, so the candidate is a no-op.
  - Decision: no failure-suite, no SPRT. Add `validate.py net-diff` to catch
    this class of no-op before replay gates.
- thread voting/arbitration search experiments: clearly negative in early SPRT.
- Bullet/Reckless-like backend spike:
  - `./build.py -c build.json` successfully converted 100k Enyo-labeled rows
    to Bullet text, BulletFormat, and trained the spike trainer on pwa-5090.
  - The resulting checkpoint layout is documented in
    `tools/bullet/README.md` and can be inspected with
    `tools/bullet/inspect_checkpoint.py`.
  - Enyo commit `0252200` adds a correctness-first loader/evaluator for this
    checkpoint layout. `setoption name nnue_file value <quantised.bin>` can now
    route search through the Bullet spike evaluator.
  - The first Enyo evaluator path is intentionally from-scratch per eval, so it
    is for correctness/architecture experiments, not a final speed path.
  - Local startpos `go nodes 100000` smoke: normal evaluator reached roughly
    `400k-900k` noisy NPS; Bullet from-scratch evaluator reached roughly
    `160k` NPS.
  - Enyo now supports incremental Bullet accumulators and both the original
    1024-hidden smoke checkpoint and a smaller 768-hidden checkpoint. Parity
    checks pass for both sizes.
  - pwa-5090 `go nodes 1000000`, Threads=1:
    - baseline Enyo: final reported line around `1.69M` NPS.
    - Bullet 1024: final reported line around `0.20M` NPS.
    - Bullet 768: final reported line around `0.44M` NPS.
  - pwa-5090 direct evaluator benchmark, `evalnet bench 500000`:
    - normal Enyo: `9.47M` eval/s.
    - Bullet 1024: `1.35M` eval/s.
    - Bullet 768: `1.70M` eval/s.
    - Bullet 768 is about `5.6x` slower than normal Enyo direct eval.
  - Local direct evaluator benchmark:
    - normal Enyo, `evalnet bench 500000`: `3.12M` eval/s.
    - Bullet 1024: `0.90M` eval/s.
    - Bullet 768, `evalnet bench 500000`: `0.98M` eval/s.
  - Decision: do not run Bullet SPRT yet. The 768 checkpoint is faster than
    1024, but the direct evaluator is still several times slower than the
    current Enyo evaluator. The remaining issue is evaluator/head cost, not
    checkpoint loading correctness.
- scratch/Kaiming `1e-5` preflight:
  - 10k train rows, 2k validation rows, Huber cp800, 10 epochs.
  - Gradient norms were nonzero for input, L1, L2, and output, so the training
    graph is alive.
  - Loss was effectively flat: train MAE stayed about `141.83`, validation MAE
    stayed about `141.43`, sign drifted from `50.14%` to `50.03%`.
  - Decision: `1e-5` is too conservative for scratch initialization. Continue
    with a higher-LR preflight before drawing conclusions about scratch.
- scratch/Kaiming `1e-3` preflight:
  - 10k train rows, 2k validation rows, Huber cp800, 10 epochs.
  - The preflight used `pack_limit=12000`, so it avoided repacking the full
    3M-row source.
  - Learning became visible but remained slow: train MSE moved from
    `42632.13` to `42627.18`; validation sign moved from `50.08%` to
    `53.32%`.
  - Decision: scratch training is alive, but `1e-3` is still too slow for the
    planned baseline reset. Continue with `1e-2` before scaling rows/epochs.
- scratch/Kaiming `1e-2` preflight:
  - 10k train rows, 2k validation rows, Huber cp800, 10 epochs.
  - Loss and sign moved clearly: train MAE `141.83 -> 127.44`,
    validation MAE `141.43 -> 136.25`, validation sign `50.08% -> 62.96%`.
  - Static validation over the 12k-row packed slice: MAE `130.893`,
    sign `72.67%`, correlation `0.398`, slope `0.091`, bias `+22cp`.
  - Decision: scratch path is viable enough to scale modestly, but the eval is
    heavily compressed and not remotely ready for SPRT.
- scratch/Kaiming 100k-row `1e-2` scale check:
  - 100k train rows, 20k validation rows, Huber cp800, 20 epochs.
  - Training MAE improved strongly: `141.33 -> 75.85`.
  - Validation peaked early and then overfit: best validation MAE around
    `129.12`; final selected sign around `75.14%`.
  - Static validation over the 120k-row packed slice: MAE `123.470`,
    sign `68.94%`, correlation `0.760`, slope `0.591`, bias `+85cp`.
  - Same-slice Berserk-derived reference: MAE `136.089`, sign `92.21%`,
    correlation `0.831`, slope `1.400`, bias `-28cp`.
  - Decision: scratch learning is real, but the net is still badly biased and
    much weaker by sign. Scale once more to 1M rows before deciding whether the
    scratch baseline deserves longer schedules or a different objective.
- scratch/Kaiming 1M-row `1e-2` scale check:
  - 1M train rows, 200k validation rows, Huber cp800, 20 epochs.
  - Float training looked viable: validation peaked around epoch 4-5 with MAE
    about `97-98` and sign about `84.7%`.
  - Exported `.nn` was much worse than the float `.pt` checkpoint on the same
    validation rows: float `.pt` MAE `98.357`, sign `84.71%`, bias `-6cp`;
    exported `.nn` MAE `168.067`, sign `66.15%`, bias `-146cp`.
  - Root cause: Kaiming-scale scratch weights are fractional in Enyo's raw
    integer export format. Export rounds away too much signal and shifts the
    eval badly.
  - Decision: stop scaling Kaiming. Test an export-scale-compatible scratch
    initializer or add export-aware/fake-quantized training before any larger
    scratch run.
- scratch `berserk-ish` 100k-row `1e-4` export-scale preflight:
  - Float `.pt` and exported `.nn` matched: both around MAE `143.16` and sign
    `50%` on the 20k validation slice.
  - Decision: this scale survives export, but LR `1e-4` is far too low for
    scratch learning. Increase LR before scaling rows.
- scratch `berserk-ish` 100k-row `1e-2` export-scale preflight:
  - Export matched float: float `.pt` MAE `141.041`, sign `63.04%`; exported
    `.nn` MAE `141.056`, sign `63.02%`.
  - Learning is much slower than Kaiming, but the result survives export.
  - Decision: implement export-aware quantized-forward training so Kaiming can
    learn while optimizing the rounded int16/int8 path Enyo actually loads.

Conclusion:

- The current architecture/training regime appears locally saturated.
- Further gains from relabeling or self-play refresh alone are expected to be
  low.
- Static MAE/sign is now only a rejection filter.
- Novel Enyo self-play alone was not enough.
- Do not launch another same-architecture Stockfish-d16-labeled Enyo self-play
  candidate unless a move-choice/failure-suite gate gives a concrete reason.
- Rejected-candidate failure analysis is recorded in
  `assets/failure_suite/rejected_candidate_analysis_20260521.md`.
  Several independent candidates regress the same tail positions, so the next
  useful signal is move-choice aware, not another scalar-eval bulk run.
- A repeated-tail target set with FENs is recorded in
  `assets/failure_suite/repeated_tail_targets_20260521.csv`. It currently
  contains 13 positions that regressed by at least 100cp in at least two
  rejected candidates.
- Initial taxonomy is recorded in
  `assets/failure_suite/repeated_tail_taxonomy_20260521.md`. The repeated
  failures are concentrated in low-material or queen-heavy
  conversion/defensive move choice.
- Oracle legal-move scores for those targets are recorded in
  `assets/failure_suite/repeated_tail_move_scores_20260521.csv`.
- Reference-engine move-choice baseline is recorded in
  `assets/failure_suite/reference_move_choice_gate_20260521.csv`:
  top-1 `3/13`, top-3 `6/13`, `sum_gap_cp=32393`,
  `worst_gap_cp=31311`.

## Current Strategy

Priority order:

1. Enyo-owned baseline net.
   - If the project goal is to remove the Berserk-derived net, train a scratch
     Enyo net as its own baseline instead of fine-tuning Berserk forever.
   - This is a provenance goal first, not an immediate keeper claim.
   - Use `init_net: null` and `init: "kaiming"` in `build.json`.
   - Reuse the best-understood labeled source first to avoid confounding:
     current signed-balanced d12 self-play plus Stockfish-d16 labels.
   - Gate it as a new baseline candidate: static metrics, net-diff,
     repeated-tail move-choice gate, failure suite, then SPRT only if it is not
     obviously far weaker.
   - First run is a 10k-row preflight: prove loss decreases and gradients reach
     the sparse input, L1, L2, and output layers before any multi-day run.
   - Run an identical current-reference-initialized control if scratch looks
     promising, so the effect of initialization is measured instead of guessed.
   - If scratch is far weaker, keep it as a training base and continue with
     larger/cleaner data or move-choice training before considering promotion.

2. Architecture/features.
   - This is the primary lane.
   - First branch, learned material/phase head input, failed the pre-SPRT
     gates and should not be SPRT-tested.
   - Proper 32-bucket king refinement has failed the pre-SPRT gates; the
     input-only diagnostic was an export no-op.
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
   - Material/phase and 32-bucket king refinement have now both failed. Widening
     is allowed only as a fallback-of-last-resort after failure analysis, not as
     the next default move.
   - A Bullet/Reckless-like training backend and correctness-first Enyo loader
     now work. The next useful engine-side step is speed: incremental Bullet
     accumulators, or an intentionally Enyo-shaped Bullet trainer if the goal is
     faster training rather than a richer architecture.

3. Stronger or different teacher data.
   - Treat Stockfish d16 as the bulk baseline, not the ceiling.
   - Test d18/d20 only on high-value slices first: disagreement,
     PV-instability, failure-suite, and high-loss move-choice rows.
   - Do not spend a full bulk d20 label run unless a small slice improves
     move-choice gates, not just MAE.
   - External/prepared datasets are acceptable if converted once into the Enyo
     row format and stored with provenance under `runs/` or `assets/`.

4. Targeted move-choice data.
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

5. Tooling.
   - Tooling work is justified only when it directly supports the lanes above.
   - New candidates must use `./build.py create`.
   - The reviewed active recipe lives in `build.json` and should be updated in
     the same commit as the experiment decision.
   - Manual step-by-step pipelines are historical/legacy only.
   - Planned recipes should be concrete `build.py create` commands, not prose.

## Next Concrete Experiment

Run the scratch quantized-forward check in `build.json`.

Next branch:

- Enyo-owned scratch quantized-forward preflight.
- Active recipe: `./build.py -c build.json`.

Reason:

- the Bullet spike proved conversion/training works and Enyo can now load the
  checkpoints, but direct eval is still several times slower than current Enyo
  eval and is too slow for serious SPRT.
- material/phase and proper 32-bucket king refinement both failed pre-SPRT
  gates.
- the input-only diagnostic proved that small input-only updates can export as a
  no-op after quantization.
- another same-data architecture tweak is not justified by the current evidence.
- scratch training is justified only as a provenance/baseline reset, not because
  it is expected to immediately beat the Berserk-derived reference.

Immediate scratch decision:

- Run the 100k-row Kaiming scratch preflight from `build.json` with
  `forward: "quantized"`.
- Compare float `.pt` validation against exported `.nn` validation.
- If quantized-forward training learns and export matches, scale rows.

Deferred Bullet decision:

- Either optimize the Bullet spike layout further in Enyo:
  10 input king buckets, 8 material output buckets, pairwise-mul hidden,
  bucketed dense head, incremental accumulator updates, and a faster head path.
- Or configure Bullet to train exactly Enyo's current `.nn` layout, which gives
  faster training tooling but not a Reckless-like architecture test.
- Do not confuse these two goals.

Anti-confounding rule:

- Do not change architecture and data source in the same first candidate.
- Reuse the best-understood training source for the first architecture test:
  the current signed-balanced d12 self-play plus Stockfish-d16 labels.
- Use `build.py --labeled-jsonl` so the first architecture test repacks the
  existing labels with the new feature map instead of generating fresh
  self-play or relabeling.
- Because this moves the recipe through the new `build.py` pack/train path,
  run the pack/static/roundtrip sanity checks before training starts.

Immediate action:

- Stop bulk NNUE training.
- Use the scored target set to design the next move-choice gate: top-1/top-3
  overlap, child-score gap, and repeated-tail regression count.
- Run `tools/validate/move_choice_gate.py` on the reference and future
  candidates before SPRT.
- The current reference baseline is weak on this tiny set, so first use the
  gate as a regression detector and diagnostic. Do not require a candidate to
  solve every target before SPRT unless the scoring is refined into a stable
  validation suite.
- Add no-op export checks (`validate.py net-diff`) before static/replay gates.
- Choose the next branch only after the analysis points to a concrete weakness.

## Candidate Workflow

Normal candidate creation:

```sh
./build.py -c build.json
```

Current `build.json` intent:

- candidate name: `scratch-kaiming-100k-quant-lr1e2-e10`
- selected branch: scratch Enyo-owned quantized-forward preflight
- self-play depth: `12`
- self-play seed: `2026052111`
- skipped opening plies: `8`
- labeled input: existing imported `fresh_d12self18h64_d16_labels` JSONL
- label provenance: Stockfish depth `16`; `build.py` skips scoring because
  `labeled_jsonl` is set.
- initializer: scratch `kaiming`; no Berserk init net
- forward path: export-aware quantized int16/int8 forward
- objective: Huber, clamp `800`, beta `200`, lr `1e-2`, epochs `10`
- trainable weights: `all`
- row limit: `100k` train rows plus `20k` validation rows
- checkpoint selection: `sign`, patience disabled

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
