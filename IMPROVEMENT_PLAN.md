# Enyo NNUE Improvement Plan

`README.md` documents how to create a candidate. This file records the current
strategy for producing a stronger net.

Goal: add new signal. Do not keep rerunning the same architecture on the same
kind of Stockfish-labeled Enyo self-play.

## Current State

No trained Enyo-owned net is proven stronger than Berserk yet. The previous
near-replacement candidates (`d16-continue-latest20m-huber-sign-*` and RC2)
are now rejected for the clean Enyo-owned lane: the provenance gate traces
their init chain back to `berserk-d43206fe90e4.nn`.

Versioning rule for clean Enyo-native nets:

- Use `native-MAJOR.MINOR.PATCH[-rcN]`.
- Bump `MAJOR` only for a materially different lane: architecture, feature
  layout, oracle/data family, objective, export format, or runtime semantics.
- Bump `MINOR` for a measurable same-lane candidate from continued training,
  a new self-play iteration, or a non-trivial hyperparameter/data-dose change.
- Bump `PATCH` only for same-lineage metadata/export/calibration fixes or tiny
  corrective retrains.
- Use `rcN` for validation candidates before promotion.
- Use `native` in current status and candidate names. Historical run names may
  still contain `owned`; do not rename existing artifacts.

Current clean-native lineage names:

- `native-1.0.0`: clean Enyo-native baseline from the current lane.
- `native-1.1.0`: v2-cont continuation that beat `native-1.0.0` by
  `+99.0 +/- 40.3 Elo` over 256 games.
- `native-1.2.0-rc1`: fresh self-play continuation from `native-1.1.0`.
  It is still far below Berserk, but the native-baseline control was positive
  (`+46.4 +/- 53.7` over 128 games), so it is the current best clean-native
  starting point.
- `native-1.3.0-rc1`: next native-only self-play iteration, generated from
  `native-1.2.0-rc1` and continued from its Bullet weights.

Current build intent:

- LC0 scalar/root-q probes are rejected for now. Do not use LC0 again unless it
  is a materially different experiment with a clear chance to shorten the path
  to a stronger net.
- Current run intent: `native-1.3.0-rc1-v4selfplay-sf-d12-lr2e6-sb512-20260601`.
  Generate self-play from the best clean-native v4/native-1.2.0-rc1 net, label
  with Stockfish d12, continue from v4 Bullet weights at a lower dose, then gate
  versus v4 before any Berserk test.
- The random-init bootstrap on the v2-generated corpus is rejected:
  `native-d16-owned-bootstrap-v3-20k-sf-d12-20260601` passed provenance and
  engine-static validation, but the Berserk smoke hard-rejected at
  `-961 Elo`, LLR `-2.95/2.94`, and only `0.8%` draws.
- The clean-owned v2 continuation on that corpus produced mixed but useful
  results:
  - versus Berserk it is still a hard reject: `-771.8 Elo`, LLR
    `-2.95/2.94`, and only `0.8%` draws in the 256-game smoke;
  - versus the previous clean-owned v2 net it is a clear step forward:
    `+99.0 +/- 40.3 Elo` over 256 games, LOS `100%`, draw `16.8%`.
- Next test: generate self-play from the improved clean-owned v2-cont net,
  label with Stockfish d12, and continue from v2-cont at low dose. This tests
  whether owned self-play iteration continues to improve the clean-owned
  baseline.
- Generate positions from Enyo self-play/replay only. Self-play generated with
  Berserk, `default.net`, or an empty NNUE fallback is contaminated and rejected.
- Allow Stockfish only as a fixed oracle labeler, not as a position source.
- Require `net_provenance.py --require-clean-enyo-owned` before static
  validation or SPRT.
- First promotion threshold is "not worse than Berserk", not merely "close".
- Do not rerun a random-init candidate on the v2-generated corpus unless the
  architecture or label objective changes. Current progress is in continuation
  from the best clean-owned baseline, not scratch replacement strength.

2026-05-31 ownership correction:

- `default.net` is also a borrowed-weight source. It cannot be used to generate
  clean Enyo-owned self-play positions.
- Enyo `cebcd78` exposes `use_nnue=false`, so the next clean source run should
  generate positions with HCE/no-NNUE self-play and no `nnue_file`.

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

2026-05-31 LC0 pairwise scalar probes:

- Converted `1000` LC0/Berserk-oracle child groups into `2951` pair rows.
- Broad-preserved full pairwise run
  `pairwise-lc0oracle1k-native-d12init-w5-lr2e5-e80-20260530` was rejected:
  final pair correctness was only `64.9%`, with exported engine-static
  `mae=112.63`, `sign=82.15%`, and `0-50cp sign=62.46%`.
- Full target-only run
  `pairwise-lc0oracle1k-full-targetonly-lr2e3-e500-20260530` showed the pair
  signal is learnable when broad preservation is removed:
  - final pair correctness `89.4%`;
  - pair MAE `42.70`;
  - predicted margin `132.21` versus target margin `159.91`.
- The same target-only net collapsed broad behavior:
  - Python/static 100k: `mae=184.90`, `sign=65.45%`,
    `0-50cp sign=52.70%`;
  - engine-static 1000: `mae=164.30`, `sign=66.63%`,
    `0-50cp sign=51.19%`.
- Recovery/preservation attempts were rejected:
  - `pairwise-lc0oracle1k-recover-pw10-lr1e5-e160-20260530` was stopped at
    epoch 49 with `broad_mae=164.31`, `pair_correct=76.3%`, and predicted
    margin `95.17`.
  - `pairwise-lc0oracle1k-diverse64-preserve-pw50-lr1e5-e200-20260530` was
    stopped at epoch 18 with broad MAE already drifting from `110.24` to
    `121.19` while pair correctness stayed around `70-75%`.
- Conclusion: LC0-derived pairwise supervision is real, but pushing it through
  the same scalar eval path conflicts with broad preservation. Do not run
  another scalar pairwise-plus-broad blend unless the representation changes or
  the pairwise signal is moved to a separate policy/ranking path.

2026-05-31 separable move-policy proof:

- Added a fixed move-choice gate builder and engine-side move-gate validator.
  Same-net Berserk sanity on the 80-case smoke gate passed with identical
  baseline/candidate scores (`57/80`, no regressions).
- Added a tiny sidecar move-policy proof that scores `best` versus `played`
  moves without touching the scalar `.nn` eval path.
- Mixed LC0+loss-log cases are not useful enough:
  - compact features overfit train at `100%` but held out only `348/498`
    (`69.9%`);
  - board features overfit train at `100%` but held out `344/498` (`69.1%`).
- Source split isolated the issue:
  - LC0-only board sidecar held out `153/248` (`61.7%`);
  - loss-log-only board sidecar held out `211/250` (`84.4%`) on the capped
    1000-case set;
  - full loss-log gate (`2487` cases) held out `508/622` (`81.7%`) and seed
    checks landed at `81.0%`, `83.8%`, and `83.8%`.
- Added a no-override guard builder/evaluator using replay rows where Enyo's
  current move is within `10cp` of oracle.
- Training with one guard-negative pair per guard row produced the first clean
  offline sidecar threshold:
  - held-out mistake gate at threshold `18`: `101/622` selected, all correct;
  - held-out guard gate at threshold `18`: `0` harmful overrides and `1`
    neutral override out of `596` guards;
  - threshold `16` has more action (`158/622`) but still has `2/596` harmful
    guard overrides;
  - x2 guard negatives were worse and still had harmful guard overrides at high
    thresholds.
- Conclusion: LC0 is currently a poor source for this sidecar gate. The
  Enyo loss-log replay rows carry a real move-choice signal, and the sidecar can
  be calibrated to a clean offline threshold. This is still not a replacement
  `.nn`; the next step is a no-op/runtime preflight for an engine policy
  tie-break path, followed by real-game action-rate and SPRT gates.
- Exported sidecar JSON was verified against the same held-out sets and matched
  Python-side decisions at thresholds `16`, `18`, `20`, and `32`.

2026-05-31 d16 RC provenance correction:

- The d16 huber/sign continuation is no longer considered Enyo-owned:
  `d16-continue-latest20m-huber-sign-nocompile-lr2e7-e10-20260531`.
  - Architecture: 16 king/input buckets, 12288 features, hidden width 1024,
    L2 size 16.
  - Confirm run versus Berserk was interrupted at 2000 games by choice, not by
    a hard reject: `-5.4 +/- 12.0 Elo`, `LOS 18.9%`, draw rate `38.0%`.
  - The result is useful as a strength reference, but the provenance chain
    traces through borrowed weights and it must not be promoted as an
    Enyo-owned net.
- Loss analysis from the 2000-game confirm found candidate-specific excess
  losses concentrated in late/endgame conversion/search positions:
  - candidate as White: `585W/388D/27L`;
  - candidate as Black: `19W/373D/608L`;
  - paired opening set is strongly White-favored for both engines, so this is
    not a candidate-only black-side collapse.
- Candidate-specific search gates separated Berserk from the d16 candidate:
  - root excess gate: Berserk `36/38`, candidate `22/38`;
  - root miss subset: `14` positions;
  - search-descendant gate: Berserk `46/46`, candidate `20/46`.
- Scalar pairwise repair was tested and rejected:
  - search-descendant target-only repair improved descendant gate only
    `20/46 -> 24/46` and root gate `0/14 -> 4/14`;
  - targeted overfit on remaining root misses improved root gate to `6/14` but
    regressed descendant gate to `22/46`;
  - conclusion: scalar pairwise pressure can force some local margins, but does
    not produce stable search repair. Do not promote the pairwise repair nets.

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
- LC0/Berserk-oracle pairwise supervision through the scalar eval path:
  target-only training can learn the pairs, but broad preservation either blocks
  the movement or collapses when removed.
- LC0/Berserk-oracle sidecar policy on the current fixed gate: held-out
  accuracy is only about `62%`, and mixing it with loss-log rows drags the
  useful loss-log signal down to about `69%`.
- scalar pairwise repair of the d16 RC's candidate-specific search misses:
  local pair metrics moved, but root/search-descendant engine gates did not
  stabilize and the targeted overfit regressed the broader descendant gate.
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
   - LC0 V6 import and oracle child-target generation now work as a data-source
     path. The current result says not to push LC0 pairwise signal through the
     scalar eval head without a separability change. The first sidecar proof
     also says to exclude LC0 from the current move-policy gate until its labels
     are audited or regenerated with a better oracle setup.

2. Targeted move-choice data.
   - Expand the fixed failure-suite and disagreement/PV-instability samplers.
   - Train at most one isolated candidate from this signal at a time.
   - Tail regressions can veto a candidate even when aggregate sum diff is
     positive.
   - Longer-term goal: optimize search decision quality, not only scalar
     evaluation accuracy.
   - Current positive source: Enyo loss-log replay rows. Current rejected source
     for this gate: LC0/Berserk-oracle rows.
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

The current experiment is clean-owned v4 self-play iteration:

1. Generate 20k games of self-play using the improved clean-owned v2-cont net:
   `native-d16-owned-v2cont-v3data-lr3e6-sb512-20260601/model.nn`.
2. Extract/sample positions with `signed-balanced-v1`.
3. Label with Stockfish d12.
4. Initialize Bullet from the clean-owned v2-cont checkpoint weights:
   `native-d16-owned-v2cont-v3data-lr3e6-sb512-20260601-512`.
5. Train one low-dose continuation (`lr=3e-6 -> 1e-6`, `512` superbatches).
6. Run provenance and engine-static validation.
7. First game gate is against the previous clean-owned v2-cont baseline. Run
   Berserk smoke only if it improves the owned baseline.

Pass criteria for continuing a lane:

- a smoke against the previous clean-owned baseline should be positive;
- engine-static sign should not collapse in the `0-50cp` bucket;
- no Berserk SPRT should run until the owned-baseline gate is positive.

## Candidate Workflow

Normal candidate creation:

```sh
./build.py -c build.json
```

Current `build.json` intent:

- candidate name:
  `native-d16-owned-v4-selfplay-v2cont-sf-d12-lr3e6-sb512-20260601`
- selected branch: generate self-play with clean-owned v2-cont, label with
  Stockfish d12, and continue the clean-owned v2-cont checkpoint
- backend: Bullet
- initialization: clean-owned v2-cont Bullet checkpoint weights, with
  `require_clean_enyo_owned=true`
- current cap: 20k self-play games for the iteration proof
- rejected near-RC nets: `d16-continue-latest20m-huber-sign-*`, RC2, and all
  pairwise repair nets

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
