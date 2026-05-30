# Enyo NNUE Improvement Plan

`README.md` documents how to create a candidate. This file records the current
strategy for producing a stronger net.

Goal: add new signal. Do not keep rerunning the same architecture on the same
kind of Stockfish-labeled Enyo self-play.

## Current State

No trained Enyo net is currently a keeper.

2026-05-30 validation update:

- Added `validate.py engine-static`, which evaluates scored JSONL rows through
  Enyo's exported-net runtime path (`evalnet`) instead of the Python `.nn`
  loader.
- Python-only static metrics are no longer authoritative for promotion
  decisions. They can still catch obvious failures, but exported candidates
  must be checked through the engine path.
- On the first 1000 rows of `runs/imported/latest_20m/score/labeled.jsonl`:
  - Berserk: `mae=133.8`, `sign=92.0%`, `0-50cp sign=81.9%`.
  - native d12 20m: `mae=92.4`, `sign=86.9%`, `0-50cp sign=74.7%`.
  - native d16 fine-tune 20m: `mae=96.5`, `sign=86.6%`,
    `0-50cp sign=75.8%`.
- This explains why lower MAE did not translate to game strength: the native
  nets are less reliable in near-zero sign/ranking decisions, which dominate
  practical move choice and game outcomes.
- Promotion candidates now need engine-side static/sign checks plus early game
  smokes. Static MAE alone is a rejection filter only.

2026-05-30 32-bucket runtime preflight update:

- Tooling now reads and writes both 16-input-bucket and 32-input-bucket Enyo
  `.nn` files.
- Enyo runtime now detects 16/32-bucket `.nn` file sizes, switches the active
  feature-index table, and accepts 32-bucket nets through `evalnet`.
- Local engine validation passed: full `build/test`, focused
  `network_model.*:network_audit.*`, and a Python-generated 32-bucket zero net
  loaded through Enyo and returned `evalnet 0 cp` on `startpos`.
- pwa clean-main preflight later passed; see the 32-bucket training update below.


2026-05-30 32-bucket training update:

- pwa clean-main preflight passed after the runtime/export work:
  - init-only 32-bucket Bullet export produced a `50368836` byte `.nn`.
  - Enyo `4322021` loaded the file through `evalnet` and engine-static eval ran.
  - startpos node-search NPS was about `0.3%` slower than the 16-bucket zero net,
    inside the `3-5%` guard.
- First real 32-bucket scratch run:
  `native-kb32-d12-20m-scratch-max800-scale300-wdl25-sb4096-20260530`.
  - trained from the known Enyo d12 20M Bullet data, wdl25, scale300,
    no input factorizer.
  - Bullet phase completed 4096 superbatches in `12m20s`, around
    `1.4-1.5M` positions/sec.
  - Python/static validation on 100k rows: `mae=121.38`, `sign=81.68%`,
    `0-50cp sign=67.21%`.
  - Engine-static validation on 1000 rows: `mae=108.69`, `sign=83.00%`,
    `0-50cp sign=66.89%`.
  - The first smoke accidentally used the old `43abd4b` reference binary, which
    cannot be trusted for 32-bucket nets; discard that smoke.
  - Corrected smoke with Enyo `4322021` still scored `0` through 34 games
    against Berserk and was stopped.
- Conclusion: 32 input buckets are runtime-viable but this first scratch data
  recipe is not a keeper. Static metrics again failed to predict game strength.
  Do not launch another kb32 training run until the game/static gap is isolated.

2026-05-30 native 100M continuation update:

- Tested `native-bullet-test80-100m-d12init-lr5e7-sb4096-kb16-20260530`.
  - Continued the best native d12 Bullet checkpoint on the existing 100M
    test80 Bullet data with conservative LR (`5e-7 -> 1e-7`).
  - Bullet training completed 4096 superbatches in `8m50s`, around
    `2.0M` positions/sec.
  - Python/static validation on 100k rows: `mae=139.51`, `sign=80.86%`,
    `0-50cp sign=67.03%`.
  - Engine-static validation on 1000 rows: `mae=133.48`, `sign=84.27%`,
    `0-50cp sign=73.04%`.
  - Short Berserk smoke was stopped after 54 games at `0-52-2`,
    about `-690 Elo`, `LOS 0.0%`.
- Conclusion: the current same-architecture Enyo/self-play Bullet recipe is
  rejected. Do not spend more runs on LR/superbatch variants of this source
  without new signal.

2026-05-30 LC0 import preflight:

- Added minimal LC0 V6 tooling on a clean feature branch, without importing the
  old contaminated research branch.
- Sample decode from `training-run1--20210605-0516`:
  - `1000/1000` rows decoded.
  - played and best moves were `100%` legal.
  - top-policy moves were `7509/8000` legal (`93.86%`).
- Policy-logit child conversion smoke wrote `200` groups.
- Oracle-scored child conversion smoke wrote `20` groups through Enyo as the
  UCI scorer. This validates the format path only; keeper data still needs a
  chosen oracle and documented scoring settings.

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
- 32 input-bucket scratch on the known Enyo d12 20M recipe: runtime/NPS passed,
  static looked plausible, but corrected game smoke scored `0/34` versus
  Berserk and was stopped.
- conservative 16-bucket 100M continuation from native d12 init: trained fast,
  but engine-static near-zero sign remained weak and Berserk smoke scored
  `0-52-2` through 54 games before stopping.
- thread voting/arbitration search experiments: clearly negative in early SPRT.

Conclusion:

- The current architecture/training regime appears locally saturated.
- Further gains from relabeling or self-play refresh alone are expected to be
  low.
- Static MAE/sign is now only a rejection filter.
- Novel Enyo self-play alone was not enough.
- Do not launch another same-architecture Stockfish-labeled Enyo/self-play
  candidate unless a move-choice/failure-suite or external-data gate gives a
  concrete reason.

## Current Strategy

Priority order:

1. Stronger or different teacher data.
   - Treat Stockfish d16 as the bulk baseline, not the ceiling.
   - Test d18/d20 only on high-value slices first: disagreement,
     PV-instability, failure-suite, and high-loss move-choice rows.
   - Do not spend a full bulk d20 label run unless a small slice improves
     move-choice gates, not just MAE.
   - External/prepared datasets are acceptable if converted once into the Enyo
     row format and stored with provenance under `runs/` or `assets/`.
   - Current branch: minimal LC0 V6 import and oracle child-target generation.
     This is a data-source preflight, not a training result yet.

2. Targeted move-choice data.
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

3. Architecture/features.
   - 32 input king buckets are runtime-viable but not yet useful with the old
     data recipe.
   - Do not widen or add buckets again until the data/source problem has a
     positive gate, or a small representation probe proves unique movement on
     hard rows without game collapse.

4. Tooling.
   - Tooling work is justified only when it directly supports the lanes above.
   - New candidates must use `./build.py create`.
   - The reviewed active recipe lives in `build.json` and should be updated in
     the same commit as the experiment decision.
   - Manual step-by-step pipelines are historical/legacy only.
   - Planned recipes should be concrete `build.py create` commands, not prose.

## Next Concrete Experiment

Do not launch another native self-play Bullet training run yet.

The next experiment is an external-data preflight:

1. Keep the minimal LC0 import tooling isolated and reviewable.
2. Generate a larger LC0 sample JSONL and record legality/source summaries.
3. Score a small LC0 policy-selected child-target set with one documented
   oracle configuration.
4. Train only a small capability proof after the target set passes:
   - legal move coverage is high enough;
   - target rows have explicit provenance and ODbL license tags;
   - oracle settings are fixed and written to the target JSONL;
   - engine-static and a tiny move-choice gate are defined before training.
5. If LC0-derived targets cannot improve move-choice gates without collapse,
   stop this lane before a full candidate run.

Pass criteria for continuing a lane:

- a smoke must score at least non-catastrophically against Berserk;
- engine-static sign must improve in the `0-50cp` bucket;
- no SPRT should run on a net that cannot draw or score in the early smoke.

## Candidate Workflow

Normal candidate creation:

```sh
./build.py -c build.json
```

Current `build.json` intent:

- candidate name: `lc0-minimal-import-preflight-20260530`
- selected branch: no training; LC0 conversion/tooling preflight
- backend: dry-run placeholder only
- training source: none until LC0 JSONL and oracle child-target summaries are
  recorded and reviewed

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
