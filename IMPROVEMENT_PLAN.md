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

Current build intent:

- The `native-2.1.0` output-only material/phase head lane is closed. It is a
  useful diagnostic because it improved static MAE and bias, but it weakened
  the move-choice gate and therefore is not game-test worthy.
- The first replay-loss ranking proof was diagnostically positive but not a
  candidate. Target-only pairwise training from `native-1.5.0-rc1` on
  `replay_full_pairs.jsonl` improved the fixed replay-loss move gate from
  `1139/2487` to `1401/2487`, with `fixed=810`, `regressed=548`,
  `delta_avg_margin=+29.5cp`, and `delta_loss_weighted_margin=+23.7cp`.
  However, broad engine-static collapsed on 2000 native 1.5 labeled rows:
  `mae=282.46`, `sign=49.20%`, `bias=+100.82`, `corr=0.148`.
- Next useful NNUE action: preserve or separate that ranking signal. Do not
  SPRT the target-only proof net. The next proof must keep the move-gate gain
  while preventing broad scalar eval collapse, either through much stronger
  broad preservation or by keeping the ranking signal outside the scalar `.nn`
  eval path.
- Same-architecture self-play continuation is still incrementally useful, but
  no longer the fastest lane: `native-1.5.1-rc1` regressed against
  `native-1.5.0-rc1`, `native-1.6.0-rc1` failed its longer confirm, and
  `native-1.6.1-rc1` failed when the best native 1.6 checkpoint was continued.
- Scalar replay-pair mixing is closed for now. The only replay result with a
  clear signal is the separable sidecar move-policy gate, not another scalar
  `.nn` fine-tune.
- Output-bucket preflight is complete:
  - Enyo runtime loads legacy single-head and new multi-head `.nn` files.
  - Python `.nn` load/write/export preserves output bucket count.
  - Bullet trainer compiles with `--enyo-output-buckets`.
  - A generated 4-output-bucket `.nn` loads through Enyo and `evalnet`.
  - Startpos fixed-node NPS was within noise versus the single-head net.
- First output-bucket game gate was negative:
  `native-2.0.0-output4-vs-native-1.5.0-smoke256-20260603` finished
  `-16.3 +/- 36.2 Elo`, LLR `-0.39/2.94`, LOS `18.8%`. Do not extend the
  head-only lane.
- `native-2.0.0-rc2` also failed to produce a promotion signal. Close the
  output-bucket lane until the representation or source data changes.
- Move-policy sidecar runtime preflight is complete in Enyo:
  - `0c2598a` adds the exported `enyo.move_policy.v1` JSON loader and
    `movepolicy` diagnostic command.
  - `8b0426b` adds a default-off root guard behind `move_policy_file`, with a
    static eval safety cap from `move_policy_max_eval_drop`.
  - Engine-side checks matched Python on held-out sidecar rows, including a
    selected row with margin `22.029897`.
- `sidecar-rootguard-vs-native15-smoke1000-20260603` hard-rejected the
  generic root-override deployment at 144 games:
  `-266.9 +/- 66.1 Elo`, LLR `-2.95/2.94`, LOS `0.0%`.
  The sidecar is not safe as a broad root move selector, even with
  `move_policy_max_eval_drop=80`. Keep the default-off engine path only as
  instrumentation for active learning and targeted audits.
- Sidecar/root-search audit tooling now parses the failed SPRT PGN and compares
  policy top moves against baseline Enyo search/eval. On the first 500
  candidate plies, 19 choices passed the current root-guard trigger. Stockfish
  d14 could score 13 of those pairs: 9 harmful, 1 neutral, 3 helpful. Huge
  sidecar margins were still sometimes harmful, so raising the threshold is not
  sufficient.
- Current intended task: convert harmful sidecar/root-search disagreements into
  negative hard examples or abandon the sidecar as a runtime selector. Do not
  launch another sidecar SPRT until an oracle-confirmed trigger has a clean
  helpful/harmful split offline.
- Current build config is `native-3.0.0-rc1-halfka-v2hm-n19data`: a controlled
  feature-layout probe that keeps the single scalar output head but changes the
  input representation to `32 x 11 x 64` HalfKAv2-style mirrored king buckets
  with merged king piece channels. It reuses native-1.9.1 SF18-labeled data and
  initializes from the clean `native-1.5.0-rc1` baseline.
- Generate positions from Enyo self-play/replay only. Self-play generated with
  Berserk, `default.net`, or an empty NNUE fallback is contaminated and rejected.
- Allow Stockfish only as a fixed oracle labeler, not as a position source.
- Require `net_provenance.py --require-clean-enyo-owned` before static
  validation or SPRT.
- First promotion threshold is "not worse than Berserk", not merely "close".
- Do not rerun a random-init candidate or another same-architecture LR/data-dose
  variant unless the architecture, label objective, or source quality changes.
- Do not run another scalar replay-pair dose variant until the representation
  changes or a move-choice gate shows a concrete reason.

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

No candidate build is currently selected.

The next experiment must be a preserved or separated move-choice/ranking proof,
not another scalar eval calibration:

1. Use the fixed replay-loss move gate as the first target suite.
2. Preserve the current clean-native baseline:
   `native-1.5.0-rc1-n14dist60k-c8-sf-d14-lr2e6-sb512-20260602`.
3. Build or select child-ranking rows with real oracle scores for the `played`
   and `best` children. Rows where both children have placeholder `score=0` are
   invalid for this proof.
4. Train or evaluate a ranking signal with explicit broad preservation, or keep
   it in a separate ranking/policy path. Target-only scalar fine-tuning is
   already proven to collapse broad eval.
5. Require offline `eval_move_gate.py` improvement before any SPRT:
   candidate prefers best at least as often as baseline, fixed > regressed, and
   both average margin deltas non-negative.
6. Also require broad engine-static to remain near baseline. A proof that only
   improves the move gate while dropping broad sign toward random is diagnostic
   only.
7. Only after both gates pass, decide whether the signal belongs in a scalar
   `.nn` fine-tune, a separate sidecar/ranking path, or a new representation.

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

- disabled on purpose. `./build.py create -c build.json` should fail until the
  next preserved/separated move-choice proof is written into a reviewable
  config.
- rejected near-RC nets: `d16-continue-latest20m-huber-sign-*`,
  `native-2.1.0-rc2`, and all scalar pairwise repair nets.

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
