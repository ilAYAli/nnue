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
- `native-1.5.0-rc1`: best current clean-native continuation baseline.
- `native-1.5.1-rc1`: rejected. The 60k-full self-play continuation lost to
  `native-1.5.0-rc1` in smoke at `-16.3 +/- 36.4 Elo`, LOS `18.9%`.
- `native-1.6.0-rc1`: rejected. The final checkpoint overshot and lost to
  `native-1.5.0-rc1` in smoke at `-19.0 +/- 37.4 Elo`. Checkpoint 64 looked
  promising in the 1000-game screen at `+7.6 +/- 18.7 Elo`, LOS `78.9%`, but
  failed the longer 4000-game confirm against `native-1.5.0-rc1`; the match
  stopped at 3127 games with `-8.9 +/- 10.6 Elo`, LLR `-2.95/2.94`, LOS
  `4.9%`.
- `native-1.6.1-rc1`: rejected. This low-dose continuation from
  `native-1.6.0` checkpoint 64 improved engine-static only marginally, then
  failed game validation. Checkpoint 96 looked good in 128 games
  (`+19.0 +/- 51.7`) but contradicted itself in the 1000-game confirm and was
  stopped at 354 games around `-23.6 +/- 32.0`, LOS `7.3%`.
- `native-2.0.0-rc1`: rejected. The material-count output-bucket head-only
  adaptation lost to `native-1.5.0-rc1` in smoke at `-16.3 +/- 36.2 Elo`,
  LOS `18.8%`.
- `native-2.0.0-rc2`: rejected. The material-count output-bucket all-layer
  adaptation was approximately neutral-negative versus `native-1.5.0-rc1` in
  smoke at `-5.4 +/- 37.4 Elo`, LOS `38.8%`.
- `native-2.1.0-rc1`: rejected before games. The material/phase dense
  output-head probe used `lr=3e-6` and was effectively a no-op; the output-head
  weights moved only to about `0.0005`, and engine-static matched
  `native-1.5.0-rc1`.
- `native-2.1.0-rc2`: rejected before games. Calibrating the same output-only
  material/phase head at `lr=0.1` improved engine-static MAE on 2000 rows
  (`127.06 -> 126.03`) and bias (`-5.26 -> +0.50`), but failed the
  move-choice gate against `native-1.5.0-rc1`: baseline preferred the best move
  on `1139/2487`, candidate `1138/2487`, fixed `11`, regressed `12`,
  `delta_avg_margin=-2.0cp`, `delta_loss_weighted_margin=-3.0cp`. Do not run
  SPRT for this net.
- `native-1.7.0-rc1`: inconclusive. The unfiltered low-dose replay-pair mix
  was weak-positive versus `native-1.5.0-rc1` in smoke at
  `+6.8 +/- 36.5 Elo`, LOS `64.3%`, but not strong enough for confirmation.
  Its mixed engine-static gate also included replay mate-sentinel rows, so the
  next retry filters source scores before sampling.
- `native-1.7.1-rc1`: rejected. Filtering the replay-pair mix to
  `abs(score) <= 800` before sampling lost to `native-1.5.0-rc1` in smoke at
  `-17.7 +/- 37.5 Elo`, LOS `17.7%`.
- `native-1.7.2-rc1`: rejected. Cutting the unfiltered replay-pair dose to
  `3000` rows still lost to `native-1.5.0-rc1` in smoke at
  `-9.5 +/- 36.9 Elo`, LOS `30.6%`.
- `native-1.7.3-rc1`: rejected. A low-dose scalar continuation using repeated
  SF14-labeled child rows from harmful sidecar root-trigger pairs lost to
  `native-1.5.0-rc1` in smoke at `-23.1 +/- 36.6 Elo`, LOS `10.6%`.
- `native-1.12.0-rc1`: rejected. Full `native-1.5.0` broad preservation plus
  replay child-pair supervision passed provenance and the replay move gate, but
  lost the 256-game distributed smoke versus `native-1.5.0-rc1`:
  `89-104-63`, score `0.4707`, about `-20.38 +/- 37.07 Elo`, LOS `14.0%`.
  A later checkpoint-5 retest finished neutral over 300 distributed games
  (`112-112-76`, score `0.5000`, about `0.0 +/- 34.0 Elo`, LOS `50%`), so
  there is still no reason to extend this scalar replay-pair setup.

Previous build intent:

- `build.json` is `native-1.13.0-rc1-broadmix-existing-sf-d16-lr5e7-sb512-20260606`.
- Hypothesis: a balanced mix of existing clean Enyo-native SF16-labeled broad
  datasets can add useful data diversity without replay-pair overfit or another
  architecture change.
- Sources:
  - `1,000,000` rows from `native-1.5.0-rc1`;
  - `1,000,000` rows from `native-1.8.0-rc1`;
  - `1,000,000` rows from `native-1.9.1-rc1`;
  - up to `480,024` rescued rows from `native-1.10.0 score_rescore96`.
- Every source is filtered by `abs(score) <= 800` before sampling. The run
  starts from `native-1.5.0-rc1` and keeps the proven `16 x 12` scalar
  architecture.
- Result: neutral-positive, not a clear replacement yet. The run passed
  source mixing, provenance, and engine-static. A 300-game distributed smoke
  versus `native-1.5.0-rc1` finished `114-109-76`, about `+4.6 Elo`. The
  corrected 1000-game distributed confirm finished `375-361-264`, score
  `0.5070`, about `+4.86 +/- 18.48 Elo`, LOS `69.7%`.
- Conclusion: broad existing-data mixing is the first recent same-lane setup
  that did not fail game validation, but the signal is still too weak for a
  release replacement. Keep the `16 x 12` scalar architecture for the next
  same-lane follow-up and change only one training/data knob.
- Follow-up: `native-1.13.1-rc1` tested the slightly longer/lower-rate
  continuation from `native-1.13.0-rc1` on the same broad mix. It improved the
  validation MAE but failed the distributed 300-game smoke, so do not continue
  this exact Huber/MAE optimization path.

Rejected follow-up:

- `build.json` is `native-1.13.1-rc1-broadmix-cont-sf-d16-lr2e7-e16-sb512-20260606`.
- Hypothesis: continuing the neutral-positive `native-1.13.0-rc1` net at a
  lower learning rate on the exact same broad mixed data can turn a weak
  `+5 Elo` signal into a clearer candidate without changing architecture or
  data.
- Data is fixed to the already mixed
  `native-1.13.0-rc1/score/mixed.jsonl`; do not resample the source mix for
  this run.
- Only the continuation schedule changes: init from `native-1.13.0-rc1`,
  `lr=2e-7`, `epochs=16`, `patience=4`, same `16 x 12` scalar architecture,
  same Huber/WDL objective.
- Result: rejected by smoke. The run passed provenance and engine-static.
  Engine-static on the fixed mixed rows was nearly flat versus `native-1.13.0`
  (`mae=130.894`, `sign=83.33%`, `corr=0.813`), but the distributed 300-game
  smoke versus `native-1.5.0-rc1` finished `106-121-73`, score `0.4750`, about
  `-17.4 Elo`.
- Conclusion: lower validation MAE did not transfer to games. Do not run a
  1000-game confirm for `native-1.13.1-rc1`.

Rejected build intent:

- `build.json` is `native-4.0.0-rc1-broadmix-mpe25-sf-d16-lr1e4-e24-sb512-20260606`.
- This is a major-lane probe because the objective changes from Huber CP loss
  to MPE25 probability loss.
- Hypothesis: MPE25 may align the scalar net with win-probability/game outcome
  better than Huber CP MAE on the same broad native data. Keep data,
  architecture, and init fixed; change only the objective.
- Data is fixed to the already mixed
  `native-1.13.0-rc1/score/mixed.jsonl`; do not resample the source mix for
  this run.
- Init is fixed to `native-1.13.0-rc1`. Architecture remains `16 x 12` scalar.
  Training uses `objective=mpe25`, `lr=1e-4`, `epochs=24`, `patience=5`, and
  `select_metric=loss`.
- Result: rejected before games. Provenance passed, but engine-static showed a
  clear CP-scale collapse: `mae=152.193`, `sign=83.24%`, `corr=0.811`,
  `slope=0.405`. Do not run SPRT for `native-4.0.0-rc1`.

Rejected build intent:

- `build.json` is `native-4.0.1-rc1-mpe25-outputcal-huber-lr3e4-e24-sb512-20260606`.
- This is a patch-level corrective calibration for the `native-4.0.0` MPE25
  lane, not a new data or architecture run.
- Hypothesis: the MPE25 net preserved sign but compressed CP scale. Training
  only the output layer with Huber may restore scale while preserving any
  useful MPE-shaped hidden representation.
- Data is fixed to the already mixed
  `native-1.13.0-rc1/score/mixed.jsonl`. Init is fixed to the rejected
  `native-4.0.0-rc1` net. Architecture remains `16 x 12` scalar.
- Result: rejected before games. Provenance passed, but output-only Huber
  calibration did not repair the MPE25 scale compression. Engine-static was
  still `mae=150.205`, `sign=83.02%`, `corr=0.811`, `slope=0.423`.
- Conclusion: close the MPE25 objective branch for now. Both the direct MPE25
  run and output-only scale repair failed the static gate before SPRT.

Rejected build intent:

- `build.json` was `native-5.0.0-rc1-test80-sfbinpack-lowdose-lr2e7-sb4096-20260606`.
- This was a major-lane preflight because the data family changed from
  Enyo-generated Stockfish-labeled positions to public Stockfish/binpack
  positions. It kept the proven Enyo-native `16 x 12` scalar architecture and
  started from `native-1.5.0-rc1`.
- Hypothesis: a small public test80 dose might add broad coverage that the
  self-play/replay mixes lack, without the cost of a new distributed self-play
  and Stockfish-labeling cycle.
- Data: `20,000,000` rows from
  `test80-2024-01-jan-2tb7p.min-v2.v6.binpack`, converted through Bullet
  `sfbinpack`, filtered to `min_ply=16`, quiet positions only, and
  `abs(cp) <= 800`.
- Training: Bullet, `lr=2e-7`, `final_lr=5e-8`, `wdl=0.25`,
  `batch_size=4096`, `superbatches=512`, `save_rate=128`, all weights
  trainable, `eval_scale=300`.
- Result: rejected before games. The run was operationally useful because it
  proved the public `sfbinpack` conversion/training path is very fast
  (`20,000,000` converted rows in about `4.9s`, `512` superbatches in about
  `68s`), but the trained net failed transfer badly on native validation rows.
  Engine-static on the native-1.13 mixed gate regressed from `native-1.13.0`
  `mae=130.797`, `sign=83.35%`, `slope=0.599` to `mae=165.199`,
  `sign=80.62%`, `slope=0.320`. This is worse than the already rejected
  `native-4.0.1` MPE repair (`mae=150.205`, `sign=83.02%`, `slope=0.423`).
- Checkpoint screen: checkpoint 0 matched the native-1.5 init
  (`mae=129.814`, `sign=83.35%`, `slope=0.624`), but checkpoint 128 was
  already damaged (`mae=149.037`, `sign=81.33%`, `slope=0.434`) and later
  checkpoints degraded further. Do not SPRT this lane. Do not scale public
  test80 all-layer low-dose unless the trainable scope or objective changes.

Current build intent:

- `build.json` is `native-1.14.0-rc1-broadmix-instability-lowdose-sf-d16-lr2e7-sb512-20260606`.
- Hypothesis: the neutral-positive `native-1.13.0-rc1` broad mix is the best
  recent same-lane baseline, but it may lack hard tactical/static correction
  rows. A small weighted dose of clean Enyo d10/d18 instability rows may patch
  unstable cases while preserving broad native behavior.
- This is not an architecture, objective, or public-data change. Architecture
  remains `16 x 12` scalar, objective remains Huber/WDL, and Stockfish is only
  an oracle labeler.
- Init is fixed to `native-1.13.0-rc1`. Data is about `3,000,000` rows sampled
  from the existing native-1.13 broad mix plus four repeated passes over the
  `21,716` clean d10/d18 instability rows, for about `87k` hard rows.
- Static gates before games:
  - provenance must remain clean Enyo-owned;
  - engine-static on the mixed rows must stay close to `native-1.13.0` and must
    not show the `native-4.0.x` / `native-5.0.0` CP-scale collapse;
  - if static is weak or negative, reject before SPRT.
- If static passes, run a 200-300 game distributed Crucible smoke versus
  `native-1.5.0-rc1`. Extend only if the smoke is neutral-positive.



2026-06-06 native 1.14.0 hardmix update:

- `native-1.14.0-rc1` is rejected as a parent improvement. It passed source
  mixing, clean provenance, and engine-static without scale collapse. It also
  beat the older `native-1.5.0-rc1` baseline in distributed game tests:
  300-game smoke `110-100-90`, `+11.59 +/- 32.96 Elo`, LOS `75.5%`; 1000-game
  confirm `387-351-262`, `+12.51 +/- 18.52 Elo`, LOS `90.8%`.
- The direct parent gate versus `native-1.13.0-rc1` failed: `352-380-268`,
  score `0.4860`, about `-9.73 +/- 18.44 Elo`, LOS `15.0%`.
- Conclusion: the low-dose d10/d18 instability rows did not improve the current
  best broadmix lineage. Do not extend this exact hardmix dose. Keep
  `native-1.13.0-rc1` as the same-lane reference for now.

Rejected build intent:

- `build.json` was `native-5.1.0-rc1-test80-floathead-v13init-lr5e7-sb256-20260606`.
- This stayed in the public test80/binpack data family, but froze input/L1 and
  trained only the float-head layers from `native-1.13.0-rc1`.
- Result: rejected before games. Provenance correctly reported public external
  data and no borrowed Berserk/default init, but static transfer to native rows
  collapsed anyway. Final Bullet static on native-1.13 mixed rows was
  `mae=164.336`, `sign=79.50%`, `slope=0.340`; engine-static was
  `mae=168.231`, `sign=80.57%`, `slope=0.300`.
- Checkpoint screen on 100k native rows showed the damage was already present
  at checkpoint 64: ck0 `mae=128.714`, `sign=82.33%`, `slope=0.679`;
  ck64 `mae=152.815`, `sign=80.34%`, `slope=0.419`; ck128
  `mae=161.269`, `sign=79.89%`, `slope=0.360`; ck192
  `mae=163.396`, `sign=79.68%`, `slope=0.346`; ck256
  `mae=164.336`, `sign=79.50%`, `slope=0.340`.
- Conclusion: close the public test80 scalar-eval lane for now. Freezing the
  lower representation did not prevent CP-scale/sign transfer damage, so do
  not spend SPRT or more public-test80 scalar variants unless the representation
  or objective changes materially.

Rejected build intent:

- `build.json` was `native-1.15.0-rc1-expanded-broad-existing-sf-d16-lr15e8-sb512-20260606`.
- Hypothesis: `native-1.13.0-rc1` was still the best same-lane reference, and
  a broader clean-native existing-data mix might improve game strength by adding
  ordinary position coverage/diversity without hard-example overfit or public-data
  transfer damage.
- This changed only existing-label source scale/distribution. Architecture stayed
  `16 x 12` scalar, objective stayed Huber/WDL, and init stayed fixed to
  `native-1.13.0-rc1`.
- Result: neutral, not a parent improvement. Training, provenance, and
  engine-static completed. A 300-game distributed smoke versus `native-1.13.0`
  finished `112-110-78`, score `0.5033`, about `+2.32 +/- 33.88 Elo`, LOS
  `55.3%`. The 1000-game distributed confirm finished `378-376-246`, score
  `0.5010`, about `+0.69 +/- 18.71 Elo`, LOS `52.9%`.
- Conclusion: existing-label broad resampling has flattened out. Keep
  `native-1.13.0-rc1` as the same-lane reference and change the data freshness
  instead of promoting `native-1.15.0-rc1`.

Current build intent:

- `build.json` is `native-1.16.0-rc1-fresh40k-v13self-sf-d16-lr3e7-sb512-20260606`.
- Hypothesis: fresh self-play from the current same-lane reference
  `native-1.13.0-rc1` may add useful position coverage that repeated training
  on existing labeled pools did not provide.
- This changes only data freshness. Architecture remains `16 x 12` scalar,
  objective remains Huber/WDL, Stockfish remains only an oracle labeler, and init
  is fixed to `native-1.13.0-rc1`.
- Source generation: `40,000` Enyo self-play games with the clean
  `native-1.13.0` net, depth `8`, seed `2026060616`, `skip_plies=8`, and
  `signed-balanced-v1` sampling. Self-play is distributed through Crucible as
  `80` tasks of `500` games.
- Labeling: Stockfish d16, `hash=128`, `max_abs_cp=1600`, distributed through
  Crucible as `96` scoring tasks.
- Training: PyTorch from the `native-1.13.0` `.nn`, `lr=3e-7`, `epochs=8`,
  `batch_size=8192`, `patience=2`, all weights trainable.
- Static gates before games:
  - self-play, extraction, and Stockfish d16 scoring must complete with sane row
    counts;
  - provenance must remain clean Enyo-owned;
  - engine-static on fresh labeled rows must stay close to `native-1.13.0` and
    must not show CP-scale collapse.
- If static passes, run a distributed 300-game Crucible smoke versus
  `native-1.13.0-rc1`. Extend only if the parent smoke is neutral-positive.

2026-06-07 update:

- `native-1.16.0-rc1` is the first fresh current-parent self-play candidate
  with a clean positive parent confirm. The original game validation was
  invalidated by worker failure, so it was retested with the healthy Crucible
  worker set.
- Clean distributed smoke versus `native-1.13.0-rc1`: `116-108-76`, score
  `0.5133`, about `+9.27 +/- 34.04 Elo`, LOS `70.3%`, with `6/6` tasks done
  and no failures or stale claims.
- Clean distributed 1000-game confirm versus `native-1.13.0-rc1`:
  `382-347-271`, score `0.5175`, about `+12.17 +/- 18.40 Elo`, LOS `90.3%`,
  with `20/20` tasks done and no failures or stale claims.
- Clean distributed 4000-game screen versus `native-1.13.0-rc1` confirmed the
  signal: `1546-1405-1049`, score `0.5176`, about `+12.25 +/- 9.25 Elo`,
  LOS `99.5%`, with `40/40` tasks done and no failures or stale claims.
- Interpretation: fresh current-parent self-play is live again. Treat
  `native-1.16.0-rc1` as the new same-lane parent for follow-up training and
  validation.

Current build intent:

- `build.json` is
  `native-1.16.1-rc1-sfstatic-direct50k-v16src-lr3e7-sb64-20260606`.
- Hypothesis: direct Stockfish static evaluation to Bullet `.data` can remove
  UCI labeling and JSONL conversion overhead while preserving the useful part
  of the current native workflow: Enyo-generated fresh positions and
  Stockfish as a fixed oracle labeler.
- This is a tooling/data-pipeline preflight, not a release candidate. It
  changes only the scoring/export path. Source positions, scalar `16 x 12`
  architecture, Huber/WDL objective, and `native-1.13.0` initialization remain
  fixed.
- Source: first `50,000` rows from the completed `native-1.16.0` fresh-position
  source. The new Enyo datagen tool evaluates each accepted row with Stockfish
  static NNUE, optionally also evaluates the Enyo net for diagnostics, and
  writes Bullet records directly.
- Stop before scaling unless:
  - direct datagen writes the expected Bullet record count;
  - Bullet training accepts the direct `.data` file;
  - provenance remains clean Enyo-owned;
  - measured throughput is clearly better than UCI scoring.
- If the chain passes, the next useful action is a larger direct-Bullet data
  run using the same source family, then game validation only after static
  gates remain sane.

2026-06-07 update:

- `native-1.16.1-rc1-sfstatic-direct50k-v16src-lr3e7-sb64-20260606`
  passed the direct Bullet data chain after fixing Bullet `.data` static
  validation. The 300-game distributed smoke versus `native-1.13.0-rc1`
  finished `106-105-89`, about `+1.16 +/- 33.03 Elo`, LOS `52.7%`.
- This is neutral, not a promotion result, but it validates the pipeline enough
  to scale the same hypothesis.
- Current `build.json` is
  `native-1.16.2-rc1-sfstatic-direct500k-v16src-lr3e7-sb128-20260607`.
  It changes only data scale and training duration: 500k direct static source
  rows, 8 static scoring shards, 128 Bullet superbatches, same `16 x 12`
  architecture, same Huber/WDL objective, and same `native-1.13.0` init.
- Gate: provenance and Bullet static must pass, then run a distributed 300-game
  Crucible smoke versus `native-1.13.0-rc1`. Extend only if neutral-positive.

2026-06-07 update 2:

- `native-1.16.2-rc1-sfstatic-direct500k-v16src-lr3e7-sb128-20260607`
  passed provenance and Bullet static validation on `449,990` rows. Static
  validation reported MAE `172.204`, sign `81.27%`, corr `0.879180`, and
  slope `1.359375`.
- The 300-game distributed smoke versus `native-1.13.0-rc1` finished
  `117-106-77`, about `+12.75 +/- 33.97 Elo`, LOS `76.9%`. The 1000-game
  confirm finished `370-359-271`, about `+3.82 +/- 18.40 Elo`, LOS `65.8%`.
- Interpretation: weak-positive, not a promotion result. This is enough to
  keep the direct Stockfish-static Bullet-data lane alive, but not enough to
  promote or claim clear improvement.
- Current `build.json` is
  `native-1.16.3-rc1-sfstatic-direct2m-v16src-lr3e7-sb256-20260607`.
  It changes only data scale and training exposure: target up to 2M direct
  static source rows, 256 Bullet superbatches, same `16 x 12` scalar
  architecture, same Huber/WDL objective, same source family, and same
  `native-1.13.0` init.
- Gate: provenance and Bullet static must pass, then run a distributed
  300-game Crucible smoke versus `native-1.13.0-rc1`. Extend only if the
  game signal is at least neutral-positive.

2026-06-07 update 3:

- `native-1.16.3-rc1-sfstatic-direct2m-v16src-lr3e7-sb256-20260607`
  passed provenance and Bullet static validation on `1,800,510` rows, but the
  static shape worsened: MAE `205.867`, sign `81.26%`, corr `0.880152`, and
  slope `1.510046`.
- The 300-game smoke versus `native-1.13.0-rc1` was positive
  (`119-106-75`, about `+15.06 +/- 34.13 Elo`, LOS `80.7%`), but the
  1000-game confirm failed (`349-368-283`, about `-6.60 +/- 18.25 Elo`,
  LOS `23.9%`).
- Interpretation: reject the 2M direct-static scale-up. Do not scale this exact
  setup further. The likely failure mode is too much direct static exposure or
  data distribution shift, not a lack of raw rows.
- Current `build.json` is
  `native-1.16.4-rc1-sfstatic-500k-v162cont-lr1e7-sb64-20260607`.
  It reuses the existing 1.16.2 `449,990` direct-static Bullet rows, starts
  from the weak-positive `native-1.16.2` model, lowers LR to `1e-7`, and limits
  exposure to `64` Bullet superbatches. No rescoring is performed.
- Gate: provenance and Bullet static must pass, then run a distributed smoke
  against `native-1.13.0-rc1`; if neutral-positive, compare directly against
  `native-1.16.2` to determine whether the continuation helped.

2026-06-07 update 4:

- `native-1.16.4-rc1-sfstatic-500k-v162cont-lr1e7-sb64-20260607`
  is rejected. It passed provenance and Bullet static validation, but the
  300-game distributed smoke versus `native-1.13.0-rc1` finished
  `106-109-85`, about `-3.47 +/- 33.34 Elo`, LOS `41.9%`.
- Direct-static exposure now has a consistent shape: `native-1.16.2` at
  500k rows was weak-positive, the 2M scale-up failed confirmation, and the
  low-dose continuation from `native-1.16.2` also failed. Do not continue the
  exact 1.16.2 net on the same direct-static rows.
- New checkpoint-screen hypothesis: less exposure may be the useful part of
  the 500k direct-static lane. Export and screen `native-1.16.2` checkpoint 64
  before generating new data. Its direct-static validation is better-shaped
  than the final 1.16.2 net on the same `449,990` rows: ck64 MAE `154.850`,
  slope `1.262523`; final 1.16.2 MAE `172.204`, slope `1.359375`; 1.16.4
  MAE `184.594`, slope `1.421814`.
- Gate: put the checkpoint export under a provenance-aware run directory,
  pass clean provenance and static validation, then run a distributed
  300-game smoke versus `native-1.13.0-rc1`. Extend only if the smoke is
  neutral-positive.
- Smoke result: checkpoint 64 passed the screen. The distributed 300-game
  smoke versus `native-1.13.0-rc1` finished `118-101-81`, score `0.5283`,
  about `+19.71 +/- 33.68 Elo`, LOS `87.5%`, with all Crucible tasks clean.
  The 1000-game confirm rejected the checkpoint: `348-364-288`, score
  `0.4920`, about `-5.56 +/- 18.18 Elo`, LOS `27.4%`. Close this checkpoint
  as a candidate.

Rejected build intent:

- `native-1.17.0-rc1-broadmix-freshdose-sf-d16-lr3e7-sb512-20260607` is
  rejected. It passed source mixing, training, clean provenance, and
  engine-static (`mae=130.669`, sign `83.41%`, corr `0.809017`, slope
  `0.595812`), but the distributed 300-game smoke versus `native-1.13.0-rc1`
  finished `103-114-83`, score `0.4817`, about `-12.75 +/- 33.51 Elo`,
  LOS `22.8%`.
- Conclusion: adding a small dose of `native-1.16.0` fresh UCI-labeled rows to
  the broad `native-1.13.0` mix did not transfer. Do not extend the fresh
  self-play/fresh-dose lane without a new gate that explains why it should
  help.

Current build intent:

- `build.json` is
  `native-1.18.0-rc1-v13replaypair-pw2-lr5e7-e4-sb8192-20260607`.
- Hypothesis: Enyo replay loss-pair rows contain a real move-choice signal,
  but prior scalar pairwise runs used the older `native-1.5.0` parent. Retest
  the signal from the current `native-1.13.0` parent with conservative
  `pair_weight=2`, low LR, and only four epochs.
- This changes objective family to pairwise replay supervision. Architecture
  and runtime net format remain scalar `16 x 12`; broad data and init are fixed
  to `native-1.13.0-rc1`.
- Gate result: clean provenance passed. Broad engine-static got slightly worse
  versus `native-1.13.0` (`mae 130.797 -> 132.029`, sign `83.35% ->
  83.30%`, slope `0.598541 -> 0.580537`), but the move gate was weak-positive:
  preferred-best `1135 -> 1137`, fixed `6`, regressed `4`,
  `delta_avg_margin=+0.7cp`, `delta_loss_weighted_margin=+0.6cp`.
- The distributed 300-game smoke versus `native-1.13.0-rc1` passed strongly for
  a smoke: `119-94-87`, score `0.5417`, about `+29.02 +/- 33.25 Elo`,
  LOS `95.7%`, with all Crucible tasks clean.
- The distributed 1000-game confirm rejected the final checkpoint:
  `358-375-267`, score `0.4915`, about `-5.91 +/- 18.45 Elo`, LOS `26.5%`.
  Crucible completed cleanly: `20/20` tasks, `fail=0`, `stale=0`.
- Conclusion: close the final `native-1.18.0` replay-pair candidate. The
  300-game smoke was a statistical mirage, and the weak move-gate gain was not
  enough to transfer to games.
- Next action: screen the exported `native-1.18.0` epoch checkpoints with the
  same broad engine-static and replay move-choice gates. Only launch games for a
  checkpoint if it shows a materially better broad-vs-ranking tradeoff than the
  rejected final checkpoint.
- Checkpoint screen result: epoch `0000` had the best broad-vs-ranking tradeoff
  on gates (`mae=131.157`, sign `83.30%`, slope `0.592659`, move gate
  `1138/2487`, fixed `3`, regressed `0`, `delta_avg_margin=+0.17cp`). Its
  distributed 300-game smoke still failed badly versus `native-1.13.0-rc1`:
  `99-120-81`, score `0.4650`, about `-24.36 +/- 33.70 Elo`, LOS `7.8%`.
- Conclusion: close scalar replay-pair checkpoint rescue. The replay ranking
  signal does not transfer through the current scalar net path strongly enough
  to justify more game tests.

Current build intent:

- `build.json` is
  `native-1.19.0-rc1-v162-outputcal-broad-lr1e4-e12-sb512-20260607`.
- Hypothesis: `native-1.16.2` had the only still-live direct-static signal
  (`+3.8 +/- 18.4 Elo` versus `native-1.13.0` after 1000 games) but its
  static shape was over-amplified. Train only the output layer on the broad
  `native-1.13.0` mixed rows to restore broad CP scale while preserving any
  useful hidden representation learned from direct-static labels.
- This changes only calibration scope/data. Architecture remains scalar
  `16 x 12`; no new self-play, Stockfish scoring, public data, replay-pair
  objective, or 32-bucket representation is introduced.
- Gate: clean provenance and broad engine-static must be at least as close to
  `native-1.13.0` as `native-1.16.2`; reject before games if MAE/sign/slope
  indicate another scale-collapse or no useful calibration. If static passes,
  run a distributed 300-game Crucible smoke versus `native-1.13.0-rc1`.
- Result: rejected. The run passed clean provenance and broad engine-static was
  only slightly better than `native-1.16.2` on the same 5000 broad rows
  (`mae 140.126 -> 139.697`, sign `82.93% -> 82.97%`, slope
  `0.842 -> 0.839`), but still much worse than `native-1.13.0`
  (`mae=130.797`, sign `83.35%`, slope `0.599`). The 300-game smoke was
  weak-positive (`110-105-85`, `+5.79 +/- 33.34 Elo`, LOS `63.3%`), but
  the 1000-game confirm failed: `361-396-243`, score `0.4825`, about
  `-12.17 +/- 18.75 Elo`, LOS `10.2%`.
- Conclusion: close output-only broad calibration from `native-1.16.2`. The
  slight broad-static repair did not preserve the weak-positive game signal.
  Do not extend or confirm more checkpoints from this exact output-calibration
  lane.


Closed build intent:

- Previous `build.json` was
  `native-1.10.1-rc1-rescued480k-sf-d16-lr3e7-sb128-20260605`. It keeps the
  proven `16 x 12` scalar architecture and starts from `native-1.5.0-rc1`.
- Hypothesis: the failed `native-1.10.0` orchestration still produced a valid
  broad clean-native label set. Reusing the completed `score_rescore96` output
  should test the larger-data lane without spending another 2-3 hours
  regenerating the same Stockfish d16 labels.
- Rescued dataset: `datasets/native-1.10-preflight-480k-sf-d16.json` records
  `480024` rows, SHA-256
  `aa596a0832d5885d3cbf32d17b18afc2f933a6d91efe99a8f6662420e74edcad`, and the
  source file
  `runs/native-1.10.0-preflight-500k-sf-d16-lr3e7-sb128-20260604/score_rescore96/labeled.jsonl`.
- Do not resume the old Crucible wrapper
  `score-native-1.10.0-preflight-500k-sf-d16-lr3e7-sb128-20260604`; it was a
  failed control job. Reuse only the completed `score_rescore96/labeled.jsonl`.
- Positions remain Enyo self-play/replay only. Self-play generated with Berserk,
  `default.net`, or an empty NNUE fallback is contaminated and rejected.
- Allow Stockfish only as a fixed oracle labeler, not as a position source.
- Require `net_provenance.py --require-clean-enyo-owned` before static
  validation or SPRT.
- Rescue gates:
  - rescued JSONL row count and SHA-256 must match the dataset manifest;
  - JSONL parsing must succeed on the full file;
  - provenance must remain clean Enyo-owned;
  - engine-static on the rescued rows must not collapse versus
    `native-1.5.0-rc1`;
  - a 200-300 game smoke versus `native-1.5.0-rc1` must be neutral-positive.
- If the rescue run passes, consider a larger same-lane data run. If it fails,
  do not start the 10M run; inspect whether the failure is data quality, label
  quality, or training overshoot.
- Result: rejected by the 256-game distributed smoke versus `native-1.5.0-rc1`.
  The rescue run passed dataset integrity, provenance, and engine-static: on the
  same 2000 rescued rows, candidate MAE/sign was `139.338` / `84.34%` versus
  baseline `141.079` / `83.89%`. The game smoke finished `94-100-62`, score
  `0.4883`, about `-8.14 +/- 37.13 Elo`, LOS `33.4%`. Do not scale this exact
  rescued-480k setup to 10M; static improvement did not transfer to games.
- First promotion threshold is "not worse than Berserk", not merely "close".
- Closed lanes remain closed unless their representation or objective changes:
  output buckets, scalar replay-pair mixing, sidecar root override, and the
  `16 x 12` to `32 x 11` HalfKAv2-style initialization recipe.

2026-06-04 native 3.0.0 HalfKAv2-style hypothesis:

- Exact Stockfish-style `HalfKAv2_hm` is not just "32 buckets". It combines
  horizontal mirroring with an 11-channel piece-square layout that merges the
  own-king and opponent-king channels into one `PS_KING` channel.
- Enyo already had horizontal mirroring and supported plain `32 x 12 x 64`
  inputs. The new representation target is `32 x 11 x 64 = 22528` input
  features per perspective.
- The first 32-bucket scratch run failed, so this is not another scratch
  retry. The controlled probe initializes from clean `native-1.5.0-rc1`
  `16 x 12` weights, maps them into `32 x 11`, and continues on the already
  generated native-1.9.1 SF18-labeled data. This isolates the representation
  change from new self-play/scoring data.
- Current `build.json` is
  `native-3.0.0-rc1-halfka-v2hm-n19data-lr5e7-sb256-20260604`.
- Result: runtime/tooling support passed, but the candidate is rejected before
  games. Enyo candidate `62c929b` loaded an init-only `32 x 11` net and
  reported 32 input buckets / 11 feature channels. Training completed, but
  engine-static on the same 2000 native-1.9.1 SF18-labeled rows regressed from
  `native-1.5.0-rc1` MAE `146.057`, sign `82.98%` to candidate MAE `190.988`,
  sign `82.92%`. No SPRT was launched.
- 2026-06-11 preservation retest: `native-1.31.2-rc1` mapped the stronger
  `native-1.23.0` into the HalfKAv2-hm `32 x 11` layout and preserved broad
  static reasonably well, but the 300-game smoke versus `native-1.23.0` lost
  `108-120-72`, about `-13.90 +/- 34.35 Elo`, LOS `21.4%`. Close this
  projection lane unless a materially different initialization/projection is
  designed first.
- Required gates before games:
  - C++ and Python feature-layout tests pass.
  - A converted init-only `32 x 11` `.nn` loads in Enyo and reports
    32 input buckets / 11 feature channels.
  - Engine-static broad metrics do not collapse versus `native-1.5.0-rc1`.
  - The move-choice gate is neutral-positive versus `native-1.5.0-rc1`.
- Stop criteria: reject before SPRT if engine-static or move-choice regresses.
  If the 256-game smoke is negative, close this HalfKAv2-style init lane before
  generating more data.
- Next useful action: do not rerun this `16 x 12` to `32 x 11` initialization
  recipe. Either keep the proven `16 x 12` architecture and improve objective /
  checkpoint selection, or only revisit `32 x 11` with a different projection
  that passes the engine-static gate before training.


2026-06-04 native 1.9.1 checkpoint-screen update:

- `native-1.9.1-rc1` is not a promotion candidate. Its corrected distributed
  4096-game confirm against `native-1.5.0-rc1` finished `1517-1571-1008`,
  about `-4.6 Elo`, after invalid pwa-wsl disconnect-only chunks were
  quarantined and rerun on healthy workers.
- The 256-game smoke was positive (`+17.7 +/- 36.5 Elo`), while the longer
  confirm was slightly negative. This points to noisy smoke plus possible
  checkpoint overshoot rather than a clear data-lane gain.
- Next hypothesis: screen the saved `native-1.9.1` Bullet checkpoints
  (`64,128,192,256,320,384`) before generating new positions. If an earlier
  checkpoint is clearly better than the final net, confirm that checkpoint;
  otherwise close the same-architecture `native-1.9` continuation lane and move
  to a representation/objective change.
- Use Crucible for checkpoint screens and require actual engine+net search
  probes on every worker before tasks start.
- Checkpoint screen result, 256 games each versus `native-1.5.0-rc1`, no
  fatal/disconnect markers: ck64 `-14.9 +/- 37.9`, ck128 `+5.4 +/- 36.2`,
  ck192 `0.0 +/- 38.0`, ck256 `+5.4 +/- 36.8`, ck320 `-25.8 +/- 36.8`,
  ck384 `-5.4 +/- 36.6`.
- Conclusion: no checkpoint is clearly better. Close the same-architecture
  `native-1.9` continuation lane rather than spending a long confirm on a
  weak `+5 Elo` screen signal.

2026-06-04 native 1.5 checkpoint-screen update:

- Screened saved `native-1.5.0-rc1` Bullet checkpoints against final
  `native-1.5.0-rc1` using distributed Crucible SPRT chunks.
- The only weak-positive 256-game screen was checkpoint 320:
  `90-87-79`, `+4.07 +/- 35.46 Elo`, LOS `58.9%`.
- The 2048-game confirm rejected that checkpoint as a replacement:
  `762-770-516`, score `0.4980`, about `-1.36 +/- 13.02 Elo`, LOS `41.9%`.
- Conclusion: saved-checkpoint selection did not find a stronger native 1.5
  replacement. Do not spend more time confirming native 1.5 checkpoints.

2026-06-04 notification routing correction:

- Distributed SPRT completion routing was wrong in two ways:
  - `run_crucible_sprt.py` defaulted to `NNUE_AI_STDOUT_ENABLE=0`, so aggregate
    SPRT completions woke `AI_stdin` but suppressed the user-facing
    `AI_stdout` summary.
  - `nnue_event_ntfy.sh` used Bash `:-` defaults, so
    `NNUE_NTFY_EVENTS=` did not disable the `nnue` topic; it fell back to the
    default `done,fail,test` list.
- Both bugs are fixed and covered by focused tests. The default distributed
  SPRT hook now sends aggregate conclusions to `AI_stdout`, wakes through
  `AI_stdin`, and can suppress the `nnue` topic without disabling
  `AI_stdout`.
- pwa-5090 verification after deploying `279c755` logged
  `ai_stdout_sent`, `notifai_ok`, `ai_stdin_ntfy_ok`, and no `nnue_sent`.
  The `AI_stdout` topic contained the diagnostic result at priority 4.

2026-06-04 replay-pair checkpoint confirmation:

- `replaypair-ck1` is rejected. The corrected distributed 4000-game confirm
  against `native-1.5.0-rc1` finished `1569-1657-774`, score `0.4890`,
  about `-7.6 Elo` with approximate `+/- 9.7` CI.
- The single failed SPRT chunk was a transient engine startup failure
  (`Engine didn't respond to uciok after startup`) and reran cleanly. Future
  distributed SPRT manifests must pass `--restart off` to the SPRT runner and
  probe both candidate and reference nets through an actual `go depth 1`
  search before starting tasks.
- Add NNUE-side distributed SPRT orchestration before the next confirm:
  generate the Crucible manifest, deploy it, retry only classified transient
  startup failures, aggregate the final W/L/D result, and notify the aggregate
  conclusion through `AI_stdout` plus a wake event through `AI_stdin`.

2026-06-03 Crucible deploy wrapper correction:

- `build.py` previously wrapped `crucible deploy ... | tee tmp` inside `if`.
  Because `tee` returned success, a failed deploy could be recorded as
  `score_crucible_deploy rc=0`, letting `score_crucible_merge` run while score
  tasks were still active.
- The wrapper now captures `${PIPESTATUS[0]}` before deciding success, so a
  nonzero deploy exits nonzero unless the explicit `--resume` recovery path is
  used. A focused build-config regression test covers this.
- The `native-1.8.0-rc1` score merge failure was therefore a premature merge
  attempt, not corrupted labeled data. Let active Crucible score tasks finish,
  then verify/merge and resume after `score_merge`.

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

No bulk scalar candidate is currently approved.

The old `native-1.20.0-rc1-v16self40k-sf-d16-lr3e7-sb512-20260607`
self-play refresh plan is obsolete. Later results closed the same-size scalar
lanes: public Stockfish/LC0 value mixes, hard-delta continuation, material-shape
output heads, output-scale tuning, replay-loss pairwise fitting, and depth-8
residual scalar fitting all failed game gates or pre-game move gates.

Next work must first establish a new non-scalar or search-aware gate:

- a search/eval-interface change that improves the frozen replay-loss or
  residual move-choice gate without broad static collapse;
- a separate policy/ranking signal with a proven safe runtime action region;
- or a cheaper representation feature than the rejected fresh per-eval threat
  path, with NPS within the agreed budget and exact legacy-net parity.

Do not start another long bulk label/train/SPRT candidate until one of those
gates gives a concrete reason.

2026-06-11 residual search-depth audit:

- Ran Enyo `9f3a006` with native `1.23.0` on the `175` high-confidence
  depth-8 residual cases, using the same net and increasing only root search
  depth.
- Teacher-best recovery improved with depth:
  - depth `8`: best `47/175`, replay-bad move `67/175`, other `61/175`;
  - depth `12`: best `67/175`, replay-bad move `56/175`, other `52/175`;
  - depth `14`: best `75/175`, replay-bad move `46/175`, other `54/175`;
  - depth `16`: best `84/175`, replay-bad move `41/175`, other `50/175`.
- Transitions from depth `8` to depth `16`: `25` replay-bad moves became
  teacher-best, `30` replay-bad moves stayed replay-bad, and `4` depth-8
  teacher-best moves regressed to replay-bad.
- Interpretation: the residual set is partly search-depth sensitive. It is
  not a pure scalar-eval representation target, and it is not fully solved by
  deeper search either. Prioritize search/eval-interface diagnostics over
  another scalar pairwise retrain.

2026-06-11 LMR-off residual audit:

- Initial `use_lmr=false` testing on Enyo `9f3a006` was invalid: the UCI option
  was parsed and exposed but not checked by the LMR search block. Enyo
  `3e1a20b` fixes the guard while keeping the default at `use_lmr=true`, so
  normal play remains row-identical to `9f3a006` on the `175` residual cases
  at depths `8`, `10`, and `12`.
- With fixed Enyo `3e1a20b` and `use_lmr=false`, the same residual gate changes:
  - depth `8`: best `48/175`, replay-bad move `58/175`, other `69/175`;
  - depth `10`: best `59/175`, replay-bad move `55/175`, other `61/175`;
  - depth `12`: best `74/175`, replay-bad move `49/175`, other `52/175`.
- Row transitions versus default at depth `12`: `12` replay-bad moves became
  teacher-best, `8` teacher-best moves regressed to replay-bad.
- Cost on this set is large: depth-12 average time rose from about `47.6ms` to
  `170.7ms` per case, about `3.6x` slower.
- Conclusion: LMR has a real but expensive effect on the residual gate once the
  option is wired. Do not ship global LMR-off without games and NPS analysis.
  The useful next search lane is targeted reduction tuning around these
  residual motifs, not another scalar NNUE pairwise retrain.
- Game smoke rejected global LMR-off decisively. The 300-game Crucible smoke
  `enyo-lmr-off-vs-default-smoke300-20260611` used the same Enyo `3e1a20b`
  binary and native `1.23.0` net on both sides, with only candidate
  `use_lmr=false` versus reference `use_lmr=true`. It finished
  `55-178-67/300`, about `-151.35 Elo`. Do not test or ship global LMR-off
  again; keep only the diagnostic insight that some residual failures are
  reduction-sensitive.

## Candidate Workflow

Normal candidate creation:

```sh
./build.py -c build.json
```

Current `build.json` intent:

- `native-1.16.0-rc1-fresh40k-v13self-sf-d16-lr3e7-sb512-20260606` is
  enabled. It tests fresh current-parent self-play while keeping architecture,
  objective, and init fixed.
- rejected near-RC nets: `d16-continue-latest20m-huber-sign-*`,
  `native-2.1.0-rc2`, `native-1.12.0-rc1`, and all scalar pairwise repair nets.

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
- Frozen baseline suite: `runs/move-policy-loss-full-gate-20260531/cases.jsonl`
  with `2487` positions.
- Current-reference self-baseline is recorded in
  `runs/move-gate-native123-self-baseline-20260611/`: native-1.23.0 versus
  itself gives `baseline_prefers_best=1141/2487`,
  `candidate_prefers_best=1141/2487`, `fixed=0`, `regressed=0`,
  `delta_avg_margin=0.0cp`, and `delta_loss_weighted_margin=0.0cp`.
- This gate is mandatory before spending SPRT on non-trivial native candidates.
  It should reject ugly tails even when static MAE improves.
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

Current material/phase head status:

- Enyo runtime support is merged in engine commit `ada4452`.
- NNUE Python tooling now supports material/phase output-head nets for the
  PyTorch backend: layout detection, load/write, zero-preserving expansion from
  legacy nets, train/export, and static validation.
- Verified gates:
  - Enyo full test binary: `79 passed`, `11 skipped` due unavailable
    tablebases.
  - NNUE tool tests: `71 passed`.
  - A tiny PyTorch train/export smoke produced a 16-input/1-output/2-head net
    of `25203020` bytes.
  - Enyo candidate `ada4452` loaded that net and `evalnet` returned a score.
  - Zero-expanded material/phase native 1.5 is eval-identical to legacy native
    1.5 on sampled FENs.
  - Candidate-engine single-thread depth-12 NPS preflight measured
    material-head/legacy mean ratio `0.9965`.
- Bullet does not yet train dense material/phase output-head features. Use
  `backend=pytorch` for the first architecture probe.
- `native-2.1.0-rc1` completed cleanly but was effectively a no-op:
  `lr=3e-6` only moved the new output-head weights to about `0.0005`, and
  engine-static metrics were identical to native 1.5 on the same 2000 rows.
- A short output-only LR calibration on the same packed data found:
  - `lr=1e-3`: held-out MAE `125.97`.
  - `lr=1e-2`: held-out MAE `125.22`.
  - `lr=1e-1`: held-out MAE `124.92`, and engine-static 2000-row MAE
    `125.915` versus native 1.5/rc1 `127.059`.
  `native-2.1.0-rc2` therefore uses `lr=0.1`.
- `native-2.1.0-rc2` was rejected by the move-choice gate despite slightly
  better broad static MAE. Gate result versus native 1.5: baseline
  `1139/2487`, candidate `1138/2487`, fixed `11`, regressed `12`,
  `delta_avg_margin=-2.0cp`, `delta_loss_weighted_margin=-3.0cp`.

Replay-loss ranking status:

- Target-only replay-pair training proved the ranking signal exists:
  `movechoice-proof-replaypair-targetonly-lr1e4-e24-20260604` improved the
  gate to `1401/2487`, fixed `810`, regressed `548`,
  `delta_avg_margin=+29.5cp`, but collapsed broad eval to MAE `282.459` and
  sign `49.20%`.
- Preserved scalar fine-tunes were not enough:
  - `pair_weight=2`: candidate `1150/2487`, fixed `38`, regressed `27`,
    `delta_avg_margin=+0.9cp`, `delta_loss_weighted_margin=-1.1cp`,
    broad static MAE `130.239`.
  - `pair_weight=3`: candidate `1155/2487`, fixed `42`, regressed `26`,
    `delta_avg_margin=+0.9cp`, `delta_loss_weighted_margin=-1.3cp`,
    broad static MAE `131.933`.
  - `pair_weight=5`: candidate `1180/2487`, fixed `164`, regressed `123`,
    `delta_avg_margin=+5.4cp`, `delta_loss_weighted_margin=+1.2cp`,
    but broad static MAE worsened to `151.115`.
- The immediate tooling fix is epoch checkpoint export in `train_pairwise.py`.
  Use it to validate the actual broad-vs-ranking tradeoff points before
  launching another scalar candidate. Do not rely on the final epoch alone.

If this architecture branch fails gates or SPRT:

- Try at most one more independent small architecture branch before reassessing.
- The next best candidate is king-bucket refinement with full trainer/engine
  support, not a folded conversion.
- If two independent architecture branches fail, stop spending bulk GPU/search
  time and reassess base net, architecture family, and teacher source.


## 2026-06-10 public Stockfish master-binpack scale rejection

Hypothesis:

- The public Stockfish master binpacks might improve native-1.23.0 if the
  earlier 100k probes were simply too small. Test larger filtered `.bullet`
  slices from `farseerT75.binpack` and `nodes5000pv2_UHO.binpack`, initialized
  from `native-1.23.0`, with a low Bullet learning rate.

Results:

- `native-7.2.0-rc1-sfmaster-nativeblend-v23init-lr1e8-sb128-20260610`
  used native-1.23 rows plus 100k farseerT75 and 100k UHO rows. The final net
  failed confirmation versus native-1.23.0: `353-360-287/1000`, Elo
  `-2.43 +/- 18.19`, LOS `39.7%`.
- Its checkpoint `sb64` looked better in short tests:
  `111-95-94/300`, Elo `+18.55 +/- 32.66`, then
  `372-353-275/1000`, Elo `+6.60 +/- 18.35`, LOS `76.0%`. A 4000-game
  screen invalidated the signal and was stopped at about 2000 games:
  `690-732-578/2000`, Elo `-7.30 +/- 12.84`, LOS `13.3%`, LLR sum `-1.59`.
- `native-7.3.0-rc1-sfmaster2m-v23init-lr5e9-sb256-20260610` scaled the same
  idea to 1M filtered farseerT75 rows plus 1M filtered UHO rows, keeping the
  native-1.23 baseline rows and lowering LR to `5e-9 -> 1e-9`. Static improved
  only marginally: Bullet `mae=149.312`, sign `82.97%`; engine static
  `mae=138.522`, sign `83.00%`.
- 7.3.0 failed its 300-game smoke versus native-1.23.0:
  `108-112-80/300`, Elo `-4.63 +/- 33.73`, LOS `39.4%`, LLR sum `-0.19`.
- 7.3.0 checkpoint sweep did not reveal a real rescue candidate. `sb64` was
  best static (`Bullet mae=149.000`, engine `mae=138.315`) but the improvement
  was tiny and repeats the failed 7.2 checkpoint-rescue pattern.

Conclusion:

- Close the public Stockfish master-binpack scalar fine-tune lane for now.
  Scaling 100k to 2M rows did not transfer to games.
- Do not spend more SPRT time on same-lane 7.x binpack mixes or checkpoint
  rescues unless the objective/trainable scope changes and a separate gate shows
  a concrete failure-mode improvement.
- Next work should focus on a new signal path or gate: replay/move-ranking as
  a separate policy/ranking path, threat/attack features with acceptable NPS, or
  a failure-suite source that passes a move-choice gate before training.


## 2026-06-11 threat-input runtime preflight rejection

Purpose:

- Test the Enyo `feature/nnue-threat-inputs` branch as a genuinely new
  non-scalar signal path before spending NNUE training or game budget.

Checks:

- Enyo branch: `feature/nnue-threat-inputs` at `15d45be`.
- Focused tests passed locally: `network_model.*`, `nnue_audit.*`, and threat
  tests, `34/34`. Full Enyo test binary also passed, `107/107`.
- A synthetic active threat-block net was built from `enyo-native-1.23.0-rc1.nn`
  with zero threat weights and nonzero output weights. It is eval-identical on
  startpos (`60cp` for both legacy and active-threat nets), so timing isolates
  runtime overhead rather than strength changes.

NPS result:

- Single-thread local depth timing across three FENs, same Enyo branch/binary:
  - depth `10`: active threat mean `1.31M nps` versus baseline `2.09M nps`,
    about `-37.4%`;
  - depth `11`: active threat mean `0.74M nps` versus baseline `1.23M nps`,
    about `-39.3%`;
  - depth `12`: active threat mean `0.79M nps` versus baseline `1.33M nps`,
    about `-40.8%`.

Conclusion:

- Do not train or SPRT this threat-input implementation. The inactive loader
  gate is correct, but any shipped threat net would pay roughly `-40%` NPS, far
  beyond the `3-5%` preflight limit.
- Keep the branch only as reference for threat indexing/parity tests. Revisit
  threat/attack features only with a materially cheaper runtime design, likely
  an incremental or search-integrated representation rather than fresh
  per-eval threat collection.

## 2026-06-11 Berserk-relative delta rerun rejection

Purpose:

- Recheck the one remaining positive-looking `native-6.1.0` result on the
  current Enyo runtime. The original 2026-06-09 games used an older Enyo build
  with noisy PV/50-move warnings, so the weak positive screen was not clean
  enough to treat as a promotion signal.

Results:

- Final checkpoint rerun on current Enyo (`v.9f3a006`) versus `native-1.23.0`:
  `native-6.1.0-vs-native-1.23.0-current-confirm1000-20260611` finished
  `345-346-309/1000`, Elo `-0.35 +/- 17.91`, LOS `48.5%`.
- Checkpoint rescue test at Bullet superbatch 32:
  `native-6.1.0-sb32-vs-native-1.23.0-current-smoke300-20260611` finished
  `96-112-92/300`, Elo `-18.55 +/- 32.82`, LOS `13.4%`.

Conclusion:

- Close the current Berserk-relative delta lane. The final checkpoint is neutral
  on a clean current-engine confirm, and the only earlier checkpoint tested is
  clearly negative.
- Do not spend more games on `native-6.x` checkpoint rescue or same-lane
  Berserk-relative scalar targets. If Berserk guidance is revisited, it needs a
  different representation or a separate policy/ranking path with its own gate.

## 2026-06-11 move-choice gate baseline

Purpose:

- Freeze the current replay-loss move-choice suite so future native candidates
  have a pre-SPRT rejection gate instead of relying on static MAE and noisy
  300-game smokes.

Baseline:

- Suite: `runs/move-policy-loss-full-gate-20260531/cases.jsonl`, `2487` rows.
- Reference net: `native-1.23.0`.
- Result directory: `runs/move-gate-native123-self-baseline-20260611/`.
- Self-baseline result: `baseline_prefers_best=1141/2487`,
  `candidate_prefers_best=1141/2487`, `fixed=0`, `regressed=0`,
  `candidate_better_margin=0/2487`, `baseline_avg_margin=2.8cp`,
  `candidate_avg_margin=2.8cp`, `delta_avg_margin=0.0cp`, and
  `delta_loss_weighted_margin=0.0cp`.

Historical sanity reference:

- `runs/native-1.23.0-vs-native-1.16.0-movegate-20260608/` compares the
  current reference against the previous confirmed native baseline on the same
  suite: baseline `1135/2487`, candidate `1139/2487`, `fixed=16`,
  `regressed=12`, `candidate_better_margin=1132/2487`,
  `delta_avg_margin=+0.8cp`, and `delta_loss_weighted_margin=+2.6cp`.
- This is only a mild aggregate positive with visible tail regressions, so the
  gate is a rejection filter and not proof of game strength.

Command shape:

```sh
.score-venv/bin/python tools/validate/eval_move_gate.py \
  --cases runs/move-policy-loss-full-gate-20260531/cases.jsonl \
  --engine "$ENYO_ENGINE" \
  --baseline-net "$NATIVE_123_NET" \
  --candidate-net "$NATIVE_123_NET" \
  --threads 1 \
  --hash 64 \
  --timeout 20.0 \
  --output runs/move-gate-native123-self-baseline-20260611/move_gate.jsonl \
  --summary-json runs/move-gate-native123-self-baseline-20260611/summary.json
```

Conclusion:

- The gate is stable for identical nets and now has a documented zero-delta
  baseline.
- Do not launch another native candidate unless it first improves this gate or
  is explicitly an engine/runtime experiment with separate parity and NPS
  justification.


## 2026-06-11 native-1.39.0 move-gate pairwise repair

Hypothesis:

- The replay-loss move-gate pairs still contain a useful ranking signal, but
  `native-1.32.2` under-dosed it: `pair_weight=8`, `lr=1e-6`, and an `800cp`
  target cap only moved the frozen gate by `+0.2cp` loss-weighted.
- Retest the same signal from `native-1.23.0` with a stronger but capped update:
  `pair_weight=32`, `lr=3e-6`, `max_target_margin=300cp`, `epochs=12`, and the
  same broad-preserve data.

Run:

- `native-1.39.0-rc1-v23-movegatepair-pw32-cap300-lr3e6-e12-sb8192-20260611`.
- Backend: `pairwise`.
- Init and move-gate baseline: `native-1.23.0`.
- Pair rows: `runs/native-1.32.2-rc1-v23-gatepairrepair-pw8-lwcp-lr1e6-e8-sb8192-20260609/pairs/move_gate_pairs.jsonl`.
- Frozen gate: `runs/move-policy-loss-full-gate-20260531/cases.jsonl`.

Stop criteria:

- Reject before games unless the move gate passes all configured thresholds:
  candidate not below baseline, `fixed >= 40`, `regressed <= 40`,
  `delta_avg_margin >= 2cp`, and `delta_loss_weighted_margin >= 5cp`.
- If it passes, run only a 300-game smoke first.

Result:

- Rejected before games. Final checkpoint passed broad engine-static superficially
  (`mae=140.290` versus baseline `147.001`, sign `83.16%` versus `83.14%`),
  but the frozen move gate did not move in the needed direction:
  candidate `1143/2487` versus baseline `1141/2487`, `fixed=6`, `regressed=4`,
  `delta_avg_margin=-0.2cp`, `delta_loss_weighted_margin=-1.0cp`.
- Checkpoint sweep `epoch-0000..0011` also failed. Best fixed count was only
  `7`; every checkpoint had negative average and loss-weighted margin deltas.
- Conclusion: stronger broad-preserved scalar pairwise pressure still cannot
  repair the replay-loss move choices. Do not run games for `native-1.39.0`.

## 2026-06-11 native-1.39.1 target-only move-gate diagnostic

Hypothesis:

- Before closing the replay-loss pair lane completely, test whether the scalar
  eval can fit the pair signal at all when broad preservation is removed.
- If target-only training still cannot clearly improve the frozen move gate,
  the pair construction or scalar representation is not useful for this gate.
- If target-only improves the gate but damages broad static, the signal belongs
  in a separate ranking/policy path, not more scalar broad-preserved training.

Run:

- `native-1.39.1-rc1-v23-movegatepair-targetonly-cap300-lr1e4-e80-sb8192-20260611`.
- Backend: `pairwise`.
- Init and move-gate baseline: `native-1.23.0`.
- Same pair rows and frozen gate as `native-1.39.0`.
- Training change: `pairwise_broad_weight=0`, `lr=1e-4`, `epochs=80`,
  `max_target_margin=300cp`.

Stop criteria:

- This is diagnostic only. Do not run games directly from this result.
- Inspect the move gate and broad engine-static result to decide whether the
  scalar pairwise lane is dead or whether a separate policy/ranking path is
  warranted.


Result:

- Rejected before games as a candidate, but it answered the diagnostic.
- Target-only training can move the frozen gate: final candidate `1257/2487`
  versus baseline `1141/2487`, `fixed=398`, `delta_avg_margin=+12.8cp`, and
  `delta_loss_weighted_margin=+5.4cp`.
- It is destructive: `regressed=282` and broad engine-static collapsed to
  `mae=225.831`, sign `66.92%`, corr `0.459`, slope `0.193`.
- Checkpoint sweep found no usable point. `epoch-0009` already had
  `regressed=48` and negative margin deltas; later checkpoints improved margin
  only by accepting hundreds of regressions.
- Conclusion: the pair signal is learnable, but fitting CP/loss magnitude
  through the scalar eval path is not stable. Next test should convert this
  into a small-margin ranking target instead of a CP-margin target.

## 2026-06-11 native-1.39.2 small-margin move-gate ranking

Hypothesis:

- `native-1.39.1` showed that large CP-margin pair fitting is the wrong target:
  it learns the gate by damaging broad behavior and many already-correct cases.
- A small, unweighted ranking target should ask only for `best > played` by a
  modest margin, avoiding the mate-like/loss-cp outlier pressure.

Run:

- `native-1.39.2-rc1-v23-movegatepair-rank30-pw64-lr1e5-e24-sb8192-20260611`.
- Same init, pair rows, broad data, and frozen gate as `native-1.39.x`.
- Training change: restore `pairwise_broad_weight=1`, use
  `pairwise_max_target_margin=30cp`, `pairwise_loss_weight_by_cp=false`,
  `pairwise_pair_weight=64`, `lr=1e-5`, `epochs=24`.

Stop criteria:

- Reject before games unless the frozen move gate passes the same hard criteria
  as `native-1.39.0` and broad engine-static does not collapse.


Result:

- Rejected before games. The small-margin ranking target was less destructive
  than target-only training, but still failed the gate and compressed broad
  static too much.
- Broad static: candidate `mae=175.025`, sign `81.91%`, corr `0.797`, slope
  `0.360` versus baseline `mae=147.001`, sign `83.14%`, corr `0.831`, slope
  `0.843`.
- Move gate: candidate `1175/2487` versus baseline `1141/2487`, `fixed=153`,
  `regressed=119`, `delta_avg_margin=+0.9cp`,
  `delta_loss_weighted_margin=-5.2cp`.
- Conclusion: small-margin ranking still over-trains already-correct cases.
  If this lane gets one last scalar test, train only on the baseline-wrong gate
  pairs and keep broad preservation.

## 2026-06-11 native-1.39.3 baseline-wrong-only move-gate ranking

Hypothesis:

- `native-1.39.2` damaged too many already-correct gate cases. Restricting the
  pairwise rows to baseline-wrong cases should reduce regression pressure while
  broad preservation protects normal static behavior.

Run:

- `native-1.39.3-rc1-v23-movegate-wrongonly-rank30-pw64-lr1e5-e24-sb8192-20260611`.
- Pair rows: `runs/move-policy-loss-full-gate-20260531/pairs_baseline_wrong_rank30.jsonl`.
- Filtered rows: `1346` baseline-wrong gate cases, `2692` child rows.
- Same rank30 objective and broad preservation as `native-1.39.2`.

Stop criteria:

- Reject before games unless it passes the frozen move gate with
  `regressed <= 40` and broad engine-static does not collapse.

Result:

- Rejected before games. Filtering to baseline-wrong cases did not remove the
  destructive scalar tradeoff.
- Broad engine-static collapsed versus the native-1.23 baseline: candidate
  `mae=176.259`, sign `81.73%`, corr `0.792`, slope `0.354`; baseline
  `mae=147.001`, sign `83.14%`, corr `0.831`, slope `0.843`.
- Final move gate: candidate `1186/2487` versus baseline `1141/2487`,
  `fixed=173`, `regressed=128`, `delta_avg_margin=+1.8cp`,
  `delta_loss_weighted_margin=-4.3cp`. It failed the regression, average
  margin, and loss-weighted margin thresholds.
- Checkpoint sweep also found no rescue point:
  `epoch-0003` was `fixed=9`, `regressed=7`,
  `delta_loss_weighted_margin=-0.9cp`; `epoch-0015` was the closest on
  regressions at `fixed=65`, `regressed=40`, but still had only
  `delta_avg_margin=+0.1cp` and `delta_loss_weighted_margin=-3.8cp`; later
  checkpoints fixed more cases only by creating a much worse bad tail.

Conclusion:

- Close the scalar replay-loss/move-gate pairwise repair lane. The four tests
  now cover the useful variants: broad-preserved CP fitting did not move the
  gate (`native-1.39.0`), target-only fitting proved the signal is learnable
  but destroys broad eval (`native-1.39.1`), small-margin ranking still
  regressed too many cases (`native-1.39.2`), and baseline-wrong-only ranking
  still created a worse loss-weighted tail (`native-1.39.3`).
- Do not run games for `native-1.39.x`, and do not spend another run on scalar
  replay-loss pair fitting without a materially different representation. If
  this signal is revisited, it should be a separate policy/ranking path or a
  new non-scalar feature path with its own parity and NPS gates.

## 2026-06-11 native-1.39.4 capture-over-quiet scalar diagnostic

Purpose:

- Check whether the scalar pairwise lane only failed because the replay-loss
  gate was too broad. A failure taxonomy of
  `runs/move-policy-loss-full-gate-20260531/cases.jsonl` showed the dominant
  pattern: `68.4%` middlegame, `65.6%` queens-on, and the played move was a
  capture in `60.6%` of cases while the oracle best move was quiet in `76.8%`
  of cases.

Run:

- `native-1.39.4-rc1-v23-capturequiet-rank30-pw48-lr3e5-e40-sb8192-20260611`.
- Pair rows: `runs/move-policy-loss-full-gate-20260531/pairs_capturequiet_rank30.jsonl`.
- Filtered subset: `801` baseline-wrong quiet-over-capture pairs, `1602`
  child rows, mostly queen middlegames.
- Init and move-gate baseline: `native-1.23.0`.

Result:

- Rejected before games. The final checkpoint learned the targeted signal only
  by collapsing broad scalar eval: engine-static went from baseline
  `mae=147.001`, sign `83.14%`, corr `0.831`, slope `0.843` to candidate
  `mae=236.911`, sign `61.63%`, corr `0.263`, slope `0.057`.
- Final move gate also failed: candidate `1230/2487` versus baseline
  `1141/2487`, `fixed=591`, `regressed=502`, `delta_avg_margin=-2.8cp`,
  `delta_loss_weighted_margin=-14.9cp`.
- Early checkpoint screen found no rescue point:
  - `epoch-0003`: static `mae=138.086`, sign `82.72%`, corr `0.818`; gate
    `fixed=36`, `regressed=29`, `delta_avg_margin=-0.1cp`,
    `delta_loss_weighted_margin=-2.9cp`.
  - `epoch-0007`: static `mae=169.605`; gate `fixed=154`, `regressed=115`,
    `delta_loss_weighted_margin=-5.0cp`.
  - later checkpoints fixed more cases only by damaging static fit and the
    loss-weighted bad tail.

Conclusion:

- This focused subset confirms the closure of the scalar replay-loss/move-gate
  repair lane. The quiet-over-capture signal is real, but scalar eval cannot
  absorb it without broad regressions. Do not run games for `native-1.39.4`.
- The next attempt at this signal must be non-scalar: a separate policy/ranking
  path, a search-side feature, or a cheap attack/threat representation with
  explicit NPS and parity gates.

## 2026-06-11 hard-delta scalar continuation check

Purpose:

- Decide whether the hard-delta d20 scalar lane had enough gate movement to
  justify more games or continuation runs.

Results:

- `native-1.37.0-rc1-harddelta-d20-v29src-v23init-lr1e7-sb64-20260610` passed
  static validation but its completed 300-game smoke versus native-1.23.0 was
  neutral-negative: `elo=-1.2`, `ci=34.7`, draw `25.7%`. The frozen move gate
  was essentially unchanged: `fixed=2`, `regressed=3`,
  `delta_avg_margin=+0.4cp`, `delta_loss_weighted_margin=+1.2cp`.
- `native-1.38.0-rc1-harddelta-cont-v137-lr3e8-sb64-20260610` worsened the
  hard-delta static set and did not improve the move gate enough: `fixed=2`,
  `regressed=4`, `delta_loss_weighted_margin=+1.6cp`.
- `native-1.38.1-rc1-harddelta-output-v137-lr1e6-sb64-20260610` moved the
  margin more (`fixed=10`, `regressed=11`, `delta_loss_weighted_margin=+6.5cp`),
  but broad engine-static degraded heavily to `mae=162.797` versus the usual
  native-1.23 baseline around `147`, so it is not a game candidate.

Conclusion:

- Do not extend the hard-delta scalar continuation lane. It improves selected
  static or margin metrics without producing a convincing move gate or smoke
  result. Future hard-case work needs a stronger pre-game gate or a different
  representation, not more same-lane scalar continuation.

## 2026-06-10 guarded move-policy sidecar rejection

Hypothesis:

- The exported loss-log sidecar policy could improve root choices without
  pushing the signal through the saturated scalar `.nn` eval path.
- Candidate and reference used the same Enyo binary and the same
  `native-1.23.0` net. The candidate only enabled
  `move_policy_file=runs/move-policy-export-x1-heldout-20260531/model.json`
  with threshold `18` and `move_policy_max_eval_drop=80`.

Offline gate recap:

- Export recheck reproduced the held-out gate: at threshold `18`, the sidecar
  selected `101/622` held-out mistake cases with no wrong selected cases.
- Guard recheck selected `1/596` no-override guard cases, with `0` harmful
  overrides.

Result:

- The first 300-game smoke attempt had two pwa-mbp0 task failures caused by a
  generated worker cache path mismatch for the opening book. The completed
  200 games were already negative: `44-61-95/200`, Elo `-29.60`.
- A clean relaunch without pwa-mbp0 completed successfully:
  `69-94-137/300`, Elo `-29.02 +/- 29.6` versus the same engine/net baseline.

Runtime audit:

- SF14 child-scored root-trigger audit:
  `runs/sidecar-rootguard-vs-native15-smoke1000-20260603/audit-engine-full-mt20.sf14.jsonl`.
- Of `252` runtime trigger contexts, `203` had numeric Stockfish child deltas.
  At the runtime threshold (`policy_margin_vs_baseline >= 18`,
  `eval_drop_cp <= 80`), the selected set was `11` helpful, `38` neutral, and
  `154` harmful using +/-`10cp` as the helpful/harmful cutoff.
- Tightening the policy threshold did not reveal a safe action region: examples
  include threshold `40` with `eval_drop_cp <= 80` selecting `2` helpful, `6`
  neutral, and `46` harmful; threshold `10000` with `eval_drop_cp <= 80` still
  selected `2` helpful, `6` neutral, and `42` harmful.

Conclusion:

- Reject the current move-policy sidecar runtime path. The offline loss-log
  move-choice gate did not transfer to games.
- Do not integrate or tune this sidecar threshold further without a new runtime
  action-rate/audit gate that explains why overrides should help in actual
  search games.
- This also reinforces that replay/move-choice signal cannot simply be bolted
  on at root after search; if revisited, it needs either a safer in-search use
  or a different policy/ranking design.

## 2026-06-10 material-shape output-head rejection

Hypothesis:

- A small material/shape output head trained only on the output layer could
  correct hard-delta positions without disturbing the saturated native-1.23.0
  feature transformer.
- `native-8.0.0-rc1-materialshape-v23-d20head100k-lr1e2-e16-20260610`
  initialized from `native-1.23.0`, used the `native-1.37.0` hard-delta d20
  `.bullet` slice, trained `output` only for 16 epochs at LR `0.01`, and added
  `output_head_features=material-shape`.

Result:

- Static hard-delta gate improved MAE slightly versus baseline:
  candidate `mae=192.707`, sign `88.96%`; baseline `mae=194.747`, sign
  `88.97%`.
- Static broad gate also improved MAE slightly versus baseline:
  candidate `mae=147.147`, sign `83.08%`; baseline `mae=148.703`, sign
  `83.00%`.
- 300-game smoke versus `native-1.23.0` looked strongly positive:
  `126-94-80/300`, Elo `+37.20 +/- 33.85`, LOS `98.5%`.
- 1000-game confirmation rejected the candidate:
  `341-380-279/1000`, Elo `-13.56 +/- 18.30`, LOS `7.3%`.

Conclusion:

- Do not promote or extend `native-8.0.0`.
- Small output-head/static-MAE improvements are not reliable enough to spend
  confirmation or screen budget without a separate game-relevant gate.
- Treat material-shape output-only tuning as another short-smoke false positive
  unless a future version has a stronger move/failure-mode gate before games.

## Historical Notes

Important failed signals:

- Native 1.11.0 d10/d18 Stockfish instability slice did not produce a game
  improvement. Crucible scanned 48 shards from the native 1.10 source file and
  merged `21,716` clean rows with no task failures. A low-dose Bullet
  fine-tune from native 1.5 improved engine-static MAE on that selected slice
  from `177.748` to `164.960`, but the distributed 256-game smoke versus
  native 1.5 was `91 - 97 - 68`, score `0.4883`, about `-8.1 Elo`. Treat the
  d10/d18 instability slice as a static-overfit/rejection result. Do not scale
  it without a broader mixed source and a move-choice or game gate that explains
  why this selected slice should transfer.

- Native 1.9.1 checkpoint screen did not produce a promotion candidate.
  The best 256-game checkpoint smokes were only ck128 and ck256 at about
  +5.4 Elo versus native 1.5.0, and both failed the move-choice gate:
  ck128 had fixed=36, regressed=25, delta_avg_margin=-0.7cp,
  delta_loss_weighted_margin=-0.8cp; ck256 had fixed=41, regressed=30,
  delta_avg_margin=-0.6cp, delta_loss_weighted_margin=-0.7cp. Close the
  native 1.9 continuation lane and do not extend these checkpoints to long
  SPRT.

- Native 2.0.0 output4 architecture probes did not beat native 1.5.0:
  - rc1 head-only smoke: `-16.3 +/- 36.2`, LOS `18.8%`, 256 games.
  - rc2 all-layer adaptation smoke: `-5.4 +/- 37.4`, LOS `38.8%`, 256 games.
  Output4 is closed for now unless a new failure analysis points directly at
  material-count output heads.
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

## 2026-06-08 native 1.24.0 hard-disagreement duplicate-eval static labels

Hypothesis:

- The embedded Enyo+Stockfish duplicate-eval scorer can target positions where
  the current `native-1.16.0` parent and Stockfish static NNUE materially
  disagree, producing a cleaner correction signal than broad Stockfish-static
  labels.
- This changes only data selection and teacher target construction. Source
  positions stay the completed `native-1.16.0` fresh source, architecture stays
  scalar `16 x 12`, objective stays Bullet Huber/WDL, and init stays
  `native-1.16.0`.

Result:

- `native-1.24.0-rc1-sfstatic-harddelta300k-v16src-lr3e7-sb512-20260608`
  completed with the intended direct `.bullet` chain:
  `positions -> enyo_sf_static_datagen -> .bullet -> Bullet`.
- Crucible scoring produced `96/96` verified shards and `149,445` hard-delta
  rows. The deploy wrapper returned nonzero twice because pwa-5090 self-SSH
  sync/heartbeat sessions reset after the run state was already complete; the
  phase was recovered only after `crucible verify` returned `ok`.
- Static validation on the hard-delta `.bullet` set: `mae=205.051`,
  sign `89.09%`, corr `0.862608`.
- 300-game smoke versus `native-1.16.0`: `112-105-83`, Elo `+8.11 +/- 33.50`,
  LOS `68.2%`.
- 1000-game confirmation versus `native-1.16.0`: `357-375-268`,
  Elo `-6.25 +/- 18.44`, LOS `25.3%`.

Conclusion:

- Do not promote or extend `native-1.24.0`.
- The hard-disagreement slice was too small/narrow to transfer in games.
  Avoid another direct Stockfish-static-only variant unless it changes a real
  failure theory, not just the threshold or row count.
- Next useful action: validate the already-built `native-1.23.0`
  searched-label candidate with a clean 300-game smoke before starting new
  training.

## 2026-06-08 native 1.29.0 v23 self-play plus Stockfish-static mix

Hypothesis:

- A small fresh self-play source from the current `native-1.23.0` parent,
  labeled through the direct Stockfish-static `.bullet` path, may add useful
  local correction signal without abandoning the stronger searched-label
  lineage.
- This changes the data source and target mix only. Architecture remains
  scalar `16 x 12`, objective remains Bullet Huber/WDL, and init stays
  `native-1.23.0`.

Result:

- `native-1.29.0-rc1-v23self20k-sfstatic300k-v23init-lr1e7-sb512-20260608`
  completed training and provenance/static validation.
- Bullet static gate on 100k rows: `mae=151.717`, sign `81.18%`,
  corr `0.867304`, slope `1.226959`.
- 300-game smoke versus `native-1.23.0`: `113-100-87`, Elo
  `+15.06 +/- 33.20`, LOS `81.3%`.
- 1000-game confirmation versus `native-1.23.0`: `368-352-280`, Elo
  `+5.56 +/- 18.28`, LOS `72.4%`.
- 4000-game screen versus `native-1.23.0`: `1449-1516-1035`, Elo
  `-5.82 +/- 9.27`, LOS `10.9%`.

Conclusion:

- Do not promote or extend `native-1.29.0`.
- The short smoke/confirm result did not hold at 4000 games. Close this
  fresh-selfplay/static-label mix and switch failure theory instead of tuning
  the same recipe.

## 2026-06-11 LC0 test91 policy/oracle audit

Hypothesis:

- Recent LC0 `test91` training data may provide a cleaner strong-play policy
  distribution than the older LC0 samples used in previous native policy
  probes.
- Before spending game budget or runtime complexity, validate the source with
  Stockfish oracle checks and a parent-heldout sidecar transfer test.

Result:

- Downloaded `training-run2-test91-20260128-1817.tar` from LC0 test91
  (`3.8G` archive).
- 500k-row extraction was clean: `500000/500000` rows, played move legal
  `99.90%`, best move legal `99.89%`, top-policy legal `95.77%`.
- Stockfish oracle pass over a 30k preselected subset produced `10000`
  high-confidence parent positions and `65708` best-vs-played move pairs.
- LC0 policy agreed very strongly with the Stockfish oracle inside this
  selected slice: oracle best was LC0 top1 `9791/10000`, top3 `9972/10000`,
  top8 `10000/10000`.
- Parent-heldout move-policy sidecar audit:
  - compact features: `9398/13157` holdout, `71.43%`;
  - board features: `9825/13157` holdout, `74.68%`.
- Transfer to the fixed Enyo loss move gate failed:
  - compact features: `1236/2487`, `49.7%`;
  - board features: `1037/2487`, `41.7%`.

Conclusion:

- LC0 test91 is a clean and useful strong-play data source, but the current
  standalone sidecar formulation does not transfer to Enyo's actual loss gate.
- Do not promote a runtime move-policy sidecar from this data alone.
- Do not spend SPRT budget on the LC0-test91 sidecar lane without a new bridge
  that proves transfer on Enyo-specific failures first.

## 2026-06-11 LC0 test91 plus Enyo loss-gate bridge audit

Hypothesis:

- LC0 test91 may still help if the sidecar is anchored on Enyo's own replay
  loss gate instead of trained on LC0-only data.
- Split the fixed `2487` Enyo loss-gate cases by source game, train on `1969`
  cases, hold out `518` cases, then compare Enyo-only versus Enyo+LC0 mixes.

Result:

- Enyo-only sidecar holdout:
  - compact features: `418/518`, `80.69%`;
  - board features: `415/518`, `80.12%`.
- Enyo x5 plus LC0-test91 5k holdout:
  - compact features: `351/518`, `67.76%`;
  - board features: `333/518`, `64.29%`.
- Enyo x5 plus LC0-test91 10k holdout:
  - compact features: `352/518`, `67.95%`;
  - board features: `314/518`, `60.62%`.

Conclusion:

- LC0-test91 data actively hurts Enyo loss-gate generalization in this
  bridge setup.
- Close LC0-test91 as a direct move-policy bridge for now. Its value is as a
  strong-play source audit, not as an immediate Enyo failure-correction signal.
- Do not spend SPRT or runtime-integration budget on this mixed sidecar lane.

## 2026-06-11 LC0 test91 q-value scalar probe

Hypothesis:

- LC0 `test91` value targets may provide useful strong-play value information
  even though LC0 policy sidecar transfer failed.
- Calibrate LC0 `root_q` to Stockfish oracle CP on the audited 10k oracle
  subset, convert the 500k LC0 slice to `.bullet`, mix it with the native
  `1.23.0` searched-label data, and train a very low-dose scalar continuation.

Result:

- `native-7.4.0-rc1-lc0test91q500k-v23mix-lr5e9-sb128-20260611`
  completed from `native-1.23.0` init using native data plus `500000` LC0
  q-value rows.
- Q calibration used the clean bounded fit `cp ~= 16.3 + 294.3 * root_q`
  from the Stockfish-oracle subset; `.bullet` conversion was CP-only
  (`bullet_wdl=0.0`).
- Bullet static gate on native `1.23.0` data: `mae=148.246`, sign `83.02%`,
  corr `0.813327`, slope `0.775087`.
- Engine static gate on the material-phase mixed set: `mae=146.442`, sign
  `83.18%`, corr `0.831152`, slope `0.839410`.
- 300-game smoke versus `native-1.23.0`: `106-105-89`, Elo
  `+1.16 +/- 33.03`, LOS `52.7%`.
- Narrow hard-disagreement follow-up `native-7.5.0-rc1-lc0qharddelta100k-v23mix-lr1e8-sb128-20260611`
  trained from the `100000` LC0-test91 rows with the largest native-vs-LC0-q
  scalar disagreement, mixed with native `1.23.0` data.
- `native-7.5.0` static gates were sane but not game-positive: Bullet static
  on native `1.23.0` data `mae=147.986`, sign `83.04%`, corr `0.813329`,
  slope `0.773158`; engine static on the material-phase mixed set
  `mae=146.090`, sign `83.18%`, corr `0.831144`, slope `0.837194`.
- `native-7.5.0` 300-game smoke versus `native-1.23.0`: `105-113-82`, Elo
  `-9.27 +/- 33.58`, LOS `29.4%`.

Conclusion:

- Do not promote `native-7.4.0`; the smoke is neutral, not an improvement.
- This is the best 7.x public/LC0 scalar probe so far, but still does not show
  enough game movement to justify a 1000-game confirm.
- Close the direct LC0 value-mixing scalar lane unless a later analysis shows a
  narrower, concrete failure slice where LC0 value targets beat native/Stockfish
  targets before games.
- The hard-disagreement LC0-q slice did not transfer either. Do not continue the
  `native-7.x` public/LC0 scalar value family without a non-scalar bridge or a
  new pre-game gate that predicts game Elo, not only static fit.

## 2026-06-11 material-shape scalar head rejection

Hypothesis:

- A compact material/shape output head could let the saturated scalar eval use
  different calibration by material shape without changing the main feature
  transformer.

Result:

- `native-8.0.0-rc1-materialshape-v23-d20head100k-lr1e2-e16-20260610`
  improved broad static fit relative to the native `1.23.0` baseline on the
  same validation slice: candidate `mae=147.147`, sign `83.08%`, corr
  `0.813002`, slope `0.764777`; baseline `mae=148.703`, sign `83.00%`, corr
  `0.813298`, slope `0.778735`.
- The 300-game smoke was misleadingly strong: `126-94-80`, Elo
  `+37.20 +/- 33.85`, LOS `98.5%`.
- The 1000-game confirm rejected it: `341-380-279`, Elo `-13.56 +/- 18.30`,
  LOS `7.3%`.

Conclusion:

- Do not promote `native-8.0.0` and do not add more material-shape scalar heads
  without a stronger pre-game gate. The 300-game smoke was a false positive.
- Static fit and short smokes are now known to over-admit candidates in this
  lane; 1000-game confirms remain required before any promotion.

## 2026-06-11 native-1.23 output-scale sweep

Hypothesis:

- The native `1.23.0` net might be under- or over-scaled for Enyo search even
  if the raw scalar ordering is unchanged.

Result:

- Smoke tests scaled the native `1.23.0` output and played each scale against
  unscaled native `1.23.0` for `300` games:
  - scale `0.85`: `102-118-80`, Elo `-18.55 +/- 33.76`, LOS `14.0%`.
  - scale `0.95`: `109-109-82`, Elo `0.00 +/- 33.57`, LOS `50.0%`.
  - scale `1.05`: `108-113-79`, Elo `-5.79 +/- 33.80`, LOS `36.8%`.
  - scale `1.15`: `89-129-82`, Elo `-46.60 +/- 33.76`, LOS `0.3%`.

Conclusion:

- No output-scale variant is an improvement candidate. Scale `0.95` is neutral,
  modestly higher scale is negative, and `1.15` is clearly bad.
- Treat the existing native `1.23.0` scale as close enough to optimal for the
  current search. Do not spend more games on scalar output-scale tuning unless
  the search/eval interface changes materially.

## 2026-06-11 capture-over-quiet policy guard audit

Purpose:

- After `native-1.39.4` proved the quiet-over-capture signal cannot be pushed
  into scalar eval safely, test whether a separate move-policy sidecar can learn
  the same diagnostic slice and whether it has any safe no-override action
  region.

Result:

- Built `runs/move-policy-capturequiet-audit-20260611` from the
  quiet-over-capture subset of the fixed replay-loss gate:
  `1076` cases across `97` source games, split by source game into `874`
  train and `202` held-out cases.
- Compact sidecar: train `874/874`, held-out `200/202` (`99.01%`).
- Board sidecar: train `874/874`, held-out `170/202` (`84.16%`).
- On the full fixed replay-loss gate, the compact sidecar ranked
  `1918/2487` (`77.12%`) best-vs-played pairs correctly; the board sidecar
  ranked `1932/2487` (`77.68%`) correctly.
- No-override guard evaluation rejected runtime use:
  - compact threshold `20`: `39/2386` overrides, `33` harmful;
  - compact threshold `12`: `116/2386` overrides, `96` harmful;
  - board threshold `20`: `193/2386` overrides, `170` harmful.
- Restricting the guard to the exact learned pattern, where the current move is
  a capture and the policy alternative is quiet, did not help:
  - compact threshold `12`: `62/2386` overrides, `0` helpful, `8` neutral,
    `54` harmful;
  - compact threshold `20`: `2/2386` overrides, `0` helpful, `1` neutral,
    `1` harmful;
  - board threshold `12`: `35/2386` overrides, `0` helpful, `5` neutral,
    `30` harmful.
- The old policy-gate threshold report was clarified in
  `eval_move_policy_gate.py`: nonnegative thresholds cannot produce
  `selected_wrong` by construction, so the report now prints
  `selected_incorrect` and `missed_correct` instead.

Conclusion:

- A separate policy/ranking model can identify the quiet-over-capture pattern
  offline, but the current sidecar/guard formulation has no safe runtime action
  region. Do not integrate or SPRT this sidecar.
- The signal remains useful as a diagnostic target for a future representation
  or search feature, but not as the existing JSON sidecar threshold mechanism.

## 2026-06-11 replay-loss gate depth sweep

Purpose:

- Check whether the fixed replay-loss move gate is mostly exposing scalar NNUE
  ranking errors, or whether the current search already recovers many gate-best
  moves when allowed a little more depth.

Result:

- Ran Enyo `9f3a006` with native `1.23.0` on all `2487` fixed replay-loss gate
  positions and compared the engine root best move at depths `1`, `4`, and `8`
  against the gate oracle move.
- Overall gate-best recovery:
  - depth `1`: `1082/2487` (`43.5%`), played move `261`, other `1144`.
  - depth `4`: `1357/2487` (`54.6%`), played move `158`, other `972`.
  - depth `8`: `1544/2487` (`62.1%`), played move `131`, other `812`.
- By phase at depth `8`:
  - opening: `141/201` (`70.1%`);
  - middlegame: `1093/1701` (`64.3%`);
  - endgame: `310/585` (`53.0%`).
- The dominant capture-over-quiet slice also improves with search depth:
  depth `8` selects the gate-best quiet move in `628/1071` (`58.6%`) cases and
  the original played capture in only `15/1071`.

Conclusion:

- The replay-loss gate is not a pure scalar-eval training target. A large part
  of it is shallow search recovery: the existing net plus deeper search already
  finds many oracle moves.
- Do not keep treating this gate as direct scalar NNUE supervision. Future work
  should target the search/eval interface, move ordering, pruning/reduction
  conditions, or a representation feature that changes root decisions without
  broad scalar collapse.
- A useful next diagnostic is to isolate the `943/2487` depth-8 residual
  failures (`131` played move, `812` other) and study them separately; those
  are more likely to represent real missing evaluation/representation signal
  than the full replay-loss gate.

Follow-up:

- Extracted child positions for the depth-8 residual cases and labeled them
  with Stockfish d12. Usable paired labels: `921`; Stockfish preferred the
  gate-best move in `728`, the Enyo depth-8 move in `178`, and was equal in
  `15`.
- High-confidence subset with Stockfish margin `>=100cp`: `175` parent pairs.
- `native-1.40.0-rc1-v23-depth8resid175-lr1e6-e24-sb8192-20260611` trained
  only this high-confidence residual subset from native `1.23.0`, preserving
  broad scalar fit.
- Broad engine-static stayed sane and slightly improved:
  - baseline: `mae=147.001`, sign `83.14%`, corr `0.831156`,
    slope `0.843495`;
  - candidate: `mae=142.492`, sign `83.02%`, corr `0.831594`,
    slope `0.804543`.
- The residual move gate failed: baseline preferred the best move in
  `89/175`; candidate `90/175`; fixed `2`; regressed `1`;
  `delta_avg_margin=+0.5cp`; `delta_loss_weighted_margin=+1.0cp`.

Conclusion:

- Even after removing shallow-search-recovered cases, direct scalar pairwise
  training barely moves the relevant root choices. Do not SPRT `native-1.40.0`.
- Before spending more game budget on this lane, run a target-only overfit
  diagnostic on the same `175` residual pairs. If the current scalar network
  cannot memorize this slice when broad preservation is disabled, the failure is
  representational; if it can memorize it, the problem is preserving broad eval
  while moving those decisions.

Target-only diagnostics:

- `native-1.40.1-rc1-v23-depth8resid175-targetonly-lr1e4-e80-sb8192-20260611`
  disabled broad preservation and appeared to fit the pair rows in float
  PyTorch training (`98.9%` pair-correct), but the exported `.nn` reloaded in
  Python fell back to `54.3%` pair-correct. Engine move gate only reached
  `93/175`, with `fixed=32`, `regressed=28`, `delta_avg_margin=+9.9cp`, and
  `delta_loss_weighted_margin=+17.5cp`. Broad static collapsed to
  `mae=236.377`, sign `65.62%`, corr `0.541719`.
- `train_pairwise.py` now reports export-reloaded pair metrics and supports
  `--project-export-weights-each-step` so these runs cannot hide improvement in
  unexportable float deltas.
- `native-1.40.2-rc1-v23-depth8resid175-targetonly-qproj-lr1e4-e80-sb8192-20260611`
  repeated the target-only diagnostic with export-grid projection. Exported
  metrics matched training: `67.4%` pair-correct, `pred_margin=45.65cp`.
  Engine move gate improved to `117/175`, with `fixed=63`, `regressed=35`,
  `delta_avg_margin=+43.7cp`, and `delta_loss_weighted_margin=+50.0cp`.
  Broad static collapsed harder: `mae=255.345`, sign `52.38%`, corr `0.211712`.

Conclusion:

- The residual signal is real enough to move the exported engine gate when
  trained on the quantized grid, but target-only fitting destroys broad scalar
  eval and still leaves a large harmful tail. Do not run games for `1.40.1` or
  `1.40.2`.
- The only remaining scalar follow-up worth running is a small
  export-projected preservation-balanced probe. If that cannot keep broad static
  sane while moving the residual gate materially, close the depth-8 residual
  scalar lane and move back to representation/search-interface changes.

Preservation-balanced export-projected check:

- `native-1.40.3-rc1-v23-depth8resid175-qproj-preserve-lr3e5-e80-sb8192-20260611`
  repeated the export-projected residual run with broad preservation restored
  (`broad_weight=16`, `pair_weight=8`).
- Broad engine-static stayed sane and modestly improved on MAE, but correlation
  and slope softened:
  - baseline: `mae=147.001`, sign `83.14%`, corr `0.831156`,
    slope `0.843495`;
  - candidate: `mae=143.013`, sign `82.26%`, corr `0.820356`,
    slope `0.603012`.
- The residual move gate still failed: baseline preferred the best move in
  `89/175`; candidate `99/175`; fixed `12`; regressed `2`;
  `delta_avg_margin=+17.9cp`; `delta_loss_weighted_margin=+24.2cp`.
  The gate required at least `30` fixes and `+30cp` loss-weighted margin.

Conclusion:

- The depth-8 residual scalar lane is closed. Target-only export-projected
  training can move the selected root cases but destroys broad eval; preserved
  training keeps static eval usable but moves too few decisions.
- Do not run SPRT for `native-1.40.3`.
- Further progress should not be another scalar pairwise replay-loss knob.
  Move effort to representation changes, search/eval interface changes, or a
  separate policy-like signal that does not fight broad scalar preservation.
