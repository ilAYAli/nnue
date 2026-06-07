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

Run `native-1.20.0-rc1-v16self40k-sf-d16-lr3e7-sb512-20260607`.

Purpose: test whether the confirmed `native-1.16.0-rc1` parent can improve from
one same-size fresh self-play iteration generated by itself. This changes one
knob only: parent for self-play and initialization.

Initial gates:

- Keep the `16 x 12` scalar architecture and Huber/WDL objective fixed.
- Keep the same scale as `native-1.16.0`: `40,000` self-play games, Stockfish
  d16 labels, `lr=3e-7`, `epochs=8`, `batch_size=8192`, `skip_plies=8`, and
  `signed-balanced-v1` sampling.
- Use the confirmed `native-1.16.0-rc1` model for both self-play `nnue_file`
  and training init.
- Use Crucible for distributed self-play/scoring and aggregate AI_stdout/SPRT
  notifications for game validation.
- Exclude pwa-wsl while SSH resets during key exchange; send the ping blocker
  notice if it is still unavailable for all-worker work.

Promote this direction only if source generation, scoring, provenance, static
validation, and a distributed smoke versus `native-1.16.0-rc1` are clean.

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
