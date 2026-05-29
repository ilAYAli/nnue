# Enyo NNUE Improvement Plan

`README.md` documents how to create a candidate. This file records the current
strategy for producing a stronger net.

Goal: add new signal. Do not keep rerunning the same architecture on the same
kind of Stockfish-labeled Enyo self-play.

## Current State

No trained Enyo net is currently a keeper.

Latest result:

- `policy-mix533-compact-nolc0-selectdeploy-h64-t2-lr2e4-e1200` is rejected.
  Deploy-gate checkpoint selection worked, but it selected a no-op checkpoint:
  threshold `2` had `0` overrides, `0` good, `0` bad, and stayed at the base
  validation score `59/133`. Lower thresholds had useful action but unsafe
  bad overrides. This closes the broad compact sidecar setup as a near-term
  lane. The only sidecar variant with repeated held-out signal was the narrow
  `mate_like` board-feature lane; rerun that once with deploy-gate checkpoint
  selection and broad non-mate no-action gating.
- `policy-mix533-compact-nolc0-h64-nh2-bb2-t2-lr2e4-e1200` is rejected by
  the deploy gate. Removing LC0 helped but did not make the sidecar safe:
  validation base was `59/133`, raw policy was `66/133`, and threshold `2`
  reached `65/133` with `6` good overrides but `2` bad, including a `-686cp`
  harm. This exposed another process issue: the trainer selected checkpoints
  by raw validation top1/sum-gap, while deployment rejects on thresholded
  good/bad overrides. Checkpoint selection now uses the deploy threshold and
  bad-tolerance criterion. Rerun the no-LC0 diagnostic once with the fixed
  selector before closing the sidecar lane.
- `policy-mix1533-compact-h64-nh2-bb2-t4-lr2e4-e1200` is rejected. Compact
  features did not solve the mixed sidecar problem: validation base was
  `210/383`, raw policy was `206/383`, and deploy threshold `4` had only `1`
  good validation override with `6` bad. The failure is source-specific:
  LC0-oracle validation produced `0` good and `6` bad overrides at threshold
  `4`, while search-descendant was clean and smoke-loss was clean at threshold
  `2`. The next diagnostic removes LC0-oracle rows and tests only the
  real-game/search-descendant sidecar signal.
- `policy-mix1533-board-h128-nh1-bb1-t4-lr1e4-e800-r2` is a clean rejection
  after fixing checkpoint selection. The saved best checkpoint still failed to
  generalize: base validation was `210/383`, raw policy validation was
  `203/383`, and deploy threshold `4` had validation `210/383` with only `3`
  good overrides and `4` bad. The board-feature sidecar overfits this
  moderate corpus. The next diagnostic keeps the same corpus but switches to
  compact features, smaller hidden width, and stronger no-harm/base-best
  preservation.
- `policy-mix1533-board-h128-nh1-bb1-t4-lr1e4-e800` is not a clean rejection.
  It trained strongly on the mixed `1533` corpus but failed validation:
  deploy threshold `4` had all `852/1533`, validation `211/383`, `10` good
  validation overrides, and `8` bad. During inspection, the policy trainer had
  a checkpoint bug: it stored `model.state_dict()` by reference, so the
  selected best checkpoint kept mutating and the saved file was the final
  epoch, not the selected epoch. This is fixed in
  `fix: preserve best policy ranker checkpoint`; rerun the same mixed config
  as `-r2` before interpreting the result.
- `policy-desc74-board-h128-t0-val25-r14-lr2e4-e2000` passed the held-out
  sidecar diagnostic. The validation split improved from base `5/18`,
  `1107cp` summed gap to policy `12/18`, `258cp` summed gap, with `8` good
  overrides, `0` bad overrides, and export parity. The signal is therefore not
  pure memorization inside this small search-descendant family. However, this
  sidecar does not transfer directly to a mixed `1533`-row corpus without
  retraining: threshold `0` produced `79` good and `169` bad validation
  overrides; threshold `4` produced only `7` good and `5` bad. The next run
  trains directly on the moderate mixed corpus with no-harm and base-best
  preservation in the policy loss.
- `policy-desc74-board-h128-t0-r14-lr2e4-e2000` passed as an in-sample
  capability proof. On the `74` search-descendant groups, r14/base was
  `34/74` with `3285cp` summed gap. The sidecar reached `74/74`, `0cp` summed
  gap, with `40` good overrides, `0` bad overrides, and export parity
  (`max_abs_diff=0.00000095`, `argmax_mismatches=0`). This proves the
  descendant signal is learnable outside scalar eval. It does not prove
  generalization because the gate had no held-out split. The next run uses the
  same sidecar setup with `25%` validation.
- `child-ranking-desc74-refpreserve100-r14-lr5e5-e960` is rejected at the
  model gate. Raising LR from `2e-5` to `5e-5` under the same
  reference-preserve objective moved only from `.pt/.nn 35/74` to `.pt/.nn
  36/74`. This closes the preserved scalar child-ranking variant for these
  `74` search-descendant rows. Target-only scalar training can learn them, but
  preserved scalar training cannot move them enough to be useful.
- `child-ranking-desc74-refpreserve100-r14-lr2e5-e960` is rejected at the
  model gate. It reached only `.pt/.nn 35/74` on the `74`
  search-descendant groups, versus r14 baseline `34/74`. This is not an
  export mismatch; `.pt` and `.nn` agree. The preserve term is suppressing the
  descendant signal too much at this target pressure. The next run keeps the
  same target file and reference preservation but raises LR to `5e-5`.
- `child-ranking-desc74-targetonly-r14-lr5e5-e960` proves the
  search-descendant rows are learnable through export and engine eval, but is
  not a candidate. It reached `.pt/.nn 61/74`, engine `61/74`, sum gap `732cp`,
  and worst engine margin `-37cp` on the `74` descendant groups. Broad static
  collapsed badly (`MAE 157.262` versus r14 `121.808`, sign `58.54%` versus
  `91.60%`). The next run must reintroduce reference broad preservation and
  test whether a useful fraction of the descendant gain can survive normal
  broad gates.
- `child-ranking-desc74-targetonly-r14-lr1e5-e480` is rejected as a weak
  capability result. With broad preservation disabled, the direct descendant
  subset moved from r14 `.nn 34/74`, engine `34/74` to `.pt/.nn 40/74`,
  engine `41/74`. The same net scored `2473/6212` on the full r21 mixed
  engine gate, so the signal is real and export-visible, but far below the
  `55/74` capability bar. Broad static is not the blocker in this diagnostic:
  MAE improved versus r14 (`82.100` versus `121.808`) while near-zero sign
  dropped by about `3.62pp`. Run exactly one stronger target-only capability
  check. If it cannot reach the `55/74` bar, stop treating these
  search-descendant rows as scalar child-ranking targets.
- `child-ranking-fast4-r21-r14init-searchdescx5-listwise-qfwd-refpreserve30-dz5-lr1e6-e240`
  is rejected at the model gate. The mixed corpus had `6212` rows: the stable
  fast4 corpus plus the `74` search-descendant groups repeated five times.
  Baseline r14 was model `2422/6212`, engine `2398/6212`; r21 reached only
  `.pt/.nn 2424/6212` and failed the `2450` gate. On the direct
  search-descendant subset, r21 stayed exactly flat at `.pt/.nn 34/74` and
  engine `34/74`, matching r14. Broad static was safe and improved versus r14
  (`MAE 118.486` versus `121.808`, sign `91.58%` versus `91.60%`), so this is
  not broad-preserve collapse. The immediate next run is a target-only
  capability check on the `74` descendant groups. If that cannot move exported
  model and engine eval, these rows are not useful scalar child-ranking targets
  for the current architecture/objective.
- `child-ranking-fast4-r20-r14init-smokeguard29-listwise-qfwd-refpreserve30-childguard5-dz5-lr1e6-e240`
  is rejected. It trained on the base `5842` mixed corpus and used the `29`
  r18-vs-r14 smoke-worse rows only as a child-level reference-preserve guard
  against r14. Broad static stayed safe (`MAE 120.162` versus r14 `121.808`,
  sign `91.58%` versus `91.60%`), but root search did not preserve r14:
  combined search gate candidate `3704/6132`, reference `3936/6132`,
  `candidate_better=538`, `reference_better=679`, `sum_diff=-8144cp`, and
  `missing_selected=254`. On the focused `29` smoke-worse rows, r20 improved
  over r19 (`16/29` versus `9/29`) but remained far behind r14 (`27/29`).
  The important diagnosis is that these rows are search-emergent. r14 static
  `.nn`/engine child eval is only `14/29` on the same rows, while r14 root
  search is `27/29`. More root-child static guards are therefore the wrong
  next experiment; the next target must capture search/PV-descendant behavior
  or search policy, not just root child eval.
- `child-ranking-mixed-replaylatest2854-lc0oracle1000-smoker1-listwise-qfwd-refpreserve20-dz5-lr5e5-e360`
  trained on the latest synced loss-log replay rows plus LC0-oracle and the
  previous failed-smoke rows. It passed model, engine, and broad-static gates:
  `.pt/.nn` `1462/4045`, engine `1444/4045`, static MAE improved
  `129.476 -> 124.522`, sign stayed within cap (`92.32% -> 91.71%`).
  The first root-search gate exposed `79` missing selected moves. After r2
  root augmentation (`missing_after=0`), the complete replay-search verdict was
  not promotion-grade: candidate `1872/2854`, reference `1863/2854`,
  `candidate_better=301`, `reference_better=284`, but `sum_diff=-1958cp`.
  Do not smoke this net. The next run trains on
  `loss_replay_child_targets_rootaug_r2_latestcandidate.jsonl` and must improve
  over this complete-gate baseline before any SPRT.
- `child-ranking-mixed-replayrefresh2838-lc0oracle1000-smoker1-listwise-qfwd-refpreserve20-dz5-lr5e5-e360`
  passed the refreshed replay/LC0/smoke mixed child gates and broad static
  checks. After root-selected move augmentation, replay-loss root search was
  complete (`missing_selected=0`) and directionally positive on the refreshed
  loss set (`1888/2838` versus reference `1853/2838`, compare
  `candidate_better=295`, `reference_better=257`, `sum_diff=+4307cp`).
  The `256`-game smoke still rejected it: `-13.6 +/- 30.4`, LOS `19.0%`,
  draw `49.2%`. Do not promote it. Treat root-complete replay as a useful
  rejection/diagnostic gate, not a sufficient Elo predictor.
- `fkW8Ha8V` adds a fresh Lichess-loss diagnostic. Timed replay with the
  actual local `v.90c53aa` binary reproduces both bad moves: `25. Qa7??`
  instead of `b6+` (about `206cp` worse by oracle), and `29. Kh1??` instead
  of `Kg2` (mate-scale loss). Fixed-node replay avoids the second blunder, so
  `Kh1` is a search/time-path instability target rather than a plain static
  child-ranking row. Keep it as a search diagnostic unless a fixed-node gate
  also reproduces it.

Tooling correction:

- `train_child_ranking.py` now trains child margins in parent POV. The first
  target-only run exposed that the previous loss used child side-to-move POV
  with the wrong sign: loss went down while `.pt`/`.nn` gates got worse.
- `child_rank_engine_gate.py` uses the engine eval path. Current reference
  binaries do not expose `eval2`, so the gate falls back to `evalnet`. A bad
  fallback to plain `eval` made old engine-gate results invalid because it did
  not evaluate the requested child FEN.
- `nnue_event_ntfy.sh` now sends long-run `done` and `fail` events to
  `AI_stdin` by default. Phase spam stays on the normal `nnue` topic. The hook
  always posts directly to `AI_stdin` and only uses `notifai.sh` as a
  best-effort additional path, because `notifai.sh` can report success without
  reaching the active Codex session.
- Policy-ranker runs now export an engine-loadable
  `policy_ranker.json` artifact and validate exported-score parity against the
  PyTorch checkpoint. Engine-side integration must consume this artifact, not a
  Python checkpoint.
- Replay JSONL target extraction now treats replay/history-sensitive rows as
  diagnostic-only. Normal `backend=replay-jsonl` runs do not pass
  `--include-history-sensitive` and validation fails if any such row appears.
- Broad-preserve packs must match the current feature layout. The old
  `runs/bullet-enyo-format-smoke-100k/pack/train` pack has feature indices up
  to `24574`, while the current 16-bucket model has `12288` input rows. Do not
  use that pack for broad preservation. Use a compatible pack such as
  `runs/imported/fresh_d12self18h64_d16_labels_20260519_113826/pack/train`
  (`max_feature=12287`) or another audited pack.

Latest LC0 diagnostic:

- LCZero V6 records from `training-run1--20210605-0516.tar` can now be decoded
  to FEN JSONL and converted to child-ranking targets. On the first `100k`
  records, decoding produced `100000` valid rows, with played legality
  `99.94%`, best legality `99.94%`, and top-policy legality `96.22%`.
- Raw LC0 policy-logit child targets are noisy for scalar NNUE training. A
  `2k` target-only diagnostic moved export-visible behavior only slightly:
  baseline engine top1 `692/2000`, trained engine top1 `713/2000`. It did,
  however, improve worst engine margin from `-1639cp` to `-375cp`.
- Filtering to high-confidence LC0 rows is necessary. The current filtered set
  requires best policy `>=0.55` and top-2 policy gap `>=50cp`, producing
  `10000` groups from `22107` rows. Target-only training moved engine top1
  `4541/10000 -> 4649/10000`, sum gap `926800cp -> 896818cp`, and worst margin
  `-3058cp -> -479cp`.
- Adding weak reference preservation with a compatible broad pack kept only part
  of the LC0 gain: engine top1 `4583/10000`, sum gap `906914cp`, worst margin
  `-2845cp`, final broad excess about `33cp`. This is not candidate-quality.
  LC0 policy labels are useful as a diagnostic signal and position source, but
  not yet as a direct scalar eval training lane.
- New LC0-oracle path: use LC0 policy only to select plausible legal moves,
  then score the child positions with Stockfish in parent POV. This produces
  normal child-ranking JSONL with oracle cp/mate scores, not LC0 policy-logit
  pseudo-scores.
- `lc0-oracle-smoke-200-20260528`: generated `200` groups at `20k` nodes.
  Training with reference preservation moved the engine target gate from
  reference `104/200` to candidate `123/200`.
- `lc0-oracle-1k-n50k-20260528`: generated `1000` groups at `50k` nodes from
  `20000` LC0 rows. Phase mix was `79` endgame, `505` late, `364` midgame,
  `52` opening.
- `child-ranking-lc0oracle1k-listwise-qfwd-refpreserve05-dz20-lr1e4-t35-e320-20260528`:
  improved the LC0-oracle engine gate from reference `544/1000` to candidate
  `560/1000`, and improved replay-loss cross-gate from reference `701/2677`
  to candidate `731/2677`. Static MAE improved from `129.5` to `120.3`, but
  sign dropped from `92.32%` to `90.77%`. The `256`-game smoke was neutral
  negative: `-4.1 +/- 31.4`, LOS `40.0%`, draw `45.7%`.
- `child-ranking-mixed-replayloss2677-lc0oracle1000-listwise-qfwd-refpreserve10-dz10-lr7e5-e360`:
  combined `2677` replay-loss rows with `1000` LC0-oracle rows. It improved the
  combined exported/engine child gate from the reference baseline `1245/3677`
  to engine `1299/3677`, but the new broad static gate rejected it before smoke:
  sign dropped `92.32% -> 91.13%` and the near-zero bucket dropped
  `84.53% -> 81.59%`. This confirms the new broad gate catches unsafe target
  gains early. Next run keeps the same mixed target file and tightens reference
  preservation instead of changing data.
- `child-ranking-mixed-replayloss2677-lc0oracle1000-listwise-qfwd-refpreserve20-dz5-lr5e5-e360`:
  passed local gates: `.pt` `1287/3677`, engine `1271/3677`, and broad static
  stayed within caps with sign `91.63%` versus reference `92.32%`, near-zero
  bucket `82.56%` versus reference `84.53%`. The required `256`-game smoke
  rejected it anyway: `-35.4 +/- 29.4`, LOS `0.9%`, draw `52.3%`. This is the
  first clean demonstration that passing mixed target gates plus broad static
  preservation is still not sufficient for game strength.
- Conclusion: LC0-oracle child targets are a real exported/engine-side signal,
  but target-gate gains still do not imply Elo. Do not promote this candidate.
  Do not scale LC0-oracle blindly until broad sign/order preservation is part
  of the gate and/or loss.

Latest child-ranking result:

- `child-ranking-replayloss-dense-v1-listwise-qfwd-preserve02-lr1e4-e640-r2`:
  passed the dense replay-loss JSONL gate generated from Lichess losses. The
  replay target set has `4432` groups and `14105` valid rank pairs. Baseline
  exported model/engine top1 was `1024/4432` and `964/4432`; the trained net
  reached `.pt`/`.nn` `1438/4432` and engine `1492/4432`. Engine sum gap
  improved from about `1011033cp` to `644668cp`, and worst engine margin
  improved from about `-2663cp` to `-2348cp`. This proves the dense replay-loss
  child-ranking signal moves exported engine choices. The problem is broad
  drift: training `broad_excess` rose to about `229cp`, so this is not a smoke
  candidate yet. Next run keeps the same target source and raises reference
  preservation from `0.02` to `0.10`.
- `child-ranking-replayloss-dense-v1-listwise-qfwd-refpreserve10-lr1e4-e640`:
  failed the model gate. Stronger reference preservation worked on broad drift
  (`broad_excess` about `41cp`) but blocked too much target learning:
  `.pt`/`.nn` reached only `1222/4432`, below the `1300/4432` gate. This
  brackets the useful preservation range: `0.02` learns but drifts too much,
  `0.10` preserves but underlearns.
- `child-ranking-replayloss-dense-v1-listwise-qfwd-refpreserve05-lr1e4-e640`:
  passed the local replay-loss child-ranking gates exactly enough to be useful:
  `.pt`/`.nn` `1300/4432`, engine `1281/4432`, and final broad excess about
  `85cp`. It is still rejected before smoke because static broad sign collapsed:
  candidate sign `85.40%` versus reference `90.52%`, with the near-zero bucket
  `68.04%` versus `77.98%`. MAE improved, but this is the same unsafe pattern
  as earlier child-ranking candidates: better target ranking plus worse broad
  sign/order. Next run keeps the replay-loss dense target source but uses
  zero-deadzone reference preservation at lower weight:
  `child-ranking-replayloss-dense-v1-listwise-qfwd-refpreserve02-dz0-lr1e4-e640`.
- `child-ranking-replayloss-dense-v1-listwise-qfwd-refpreserve02-dz0-lr1e4-e640`:
  passed the local replay-loss child-ranking gates but made broad behavior much
  worse. Export-quantized `.pt` and exported `.nn` both reached `1433/4432`;
  engine reached `1488/4432`. Training broad excess climbed to about `247cp`.
  Static validation rejected it hard: sign `79.85%` versus reference `90.52%`,
  wrong-sign rows `18838/93479` versus reference `8861/93479`, and the near-zero
  bucket fell to `64.45%` versus reference `77.98%`. This closes dense scalar
  replay-loss child-ranking as a near-term promotion lane. The signal is useful,
  but it must not replace or fine-tune the scalar eval surface.
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
- `policy-ranker-lossv5-v3x6-lowmat-board-h64-d25-lr3e4-e120-r3`: broadened
  the board-aware sidecar to 6906 unique groups from the losslogs/v3/lowmat
  mix. It failed the split-aware gate. Policy-only improved validation only
  from `440/1726` to `469/1726`. Low thresholds produced many held-out bad
  overrides; zero-bad held-out thresholds had only `2-3` good overrides. Tag
  breakdown showed the only promising held-out slice was `mate_like`
  (`5` good, `0` bad at threshold `32`; `3` good, `0` bad at threshold `40`).
  The universal sidecar is therefore too broad; the next diagnostic is a
  motif-restricted `mate_like` policy ranker.
- `policy-ranker-matelike-mix-board-h64-d35-lr2e4-e600`: failed the
  split-aware gate. It solved train (`354/398`) but validation stayed weak
  (`45/133`) and every useful threshold had held-out bad overrides. Breakdown
  showed the bad overrides came mainly from `source:lichess_lowmat_mc12`
  (`6` good / `8` bad at threshold `32`), while losslogs/gamebalanced
  mate-like rows were much cleaner and their worst non-lowmat bad was only a
  tiny oracle tie around `-4cp`. The next run excludes
  `source:lichess_lowmat_mc12` and uses a `10cp` bad-override tolerance so
  real harm is separated from noise-level ties.
- `policy-ranker-matelike-nolowmat-board-h64-d35-lr2e4-e600`: failed. Removing
  the lowmat source from training made generalization worse, not better:
  validation policy top1 was only `19/54` with real `-800cp` bad overrides.
  The lowmat rows are unsafe as a deployment/gate slice, but their presence in
  training appears to regularize the sidecar. The next diagnostic trains on
  all `mate_like` rows and gates only the non-lowmat slice.
- `policy-ranker-matelike-trainglobal-gatenolowmat-board-h64-d35-lr2e4-e600`:
  passed the split-aware policy gate. Training used all `mate_like` rows, while
  the gate excluded `source:lichess_lowmat_mc12`. On the non-lowmat validation
  split, base top1 was `19/54`, policy-only was `39/54`, and threshold `4`
  selected `38/54` with `22` overrides, `20` good, `0` bad, and worst
  noise-level harm `-4cp` under the `10cp` tie tolerance. This is the first
  held-out policy-sidecar result with useful action. It is still narrow and
  must replicate under a different train/validation split seed before any
  engine-integration work.
- `policy-ranker-matelike-trainglobal-gatenolowmat-board-h64-d35-lr2e4-e600-s2`:
  the first pass failed only because the global all-row gate required
  `bad=0`, while the train side had `2` bad overrides. The actual held-out
  result replicated: non-lowmat validation improved from `16/54` to `45/54`,
  and threshold `4` gave `28` overrides, `26` good, `0` bad, and worst harm
  `-1cp`. For split-replication diagnostics, use the validation constraints as
  the hard pass/fail rule; global all-row safety is a later deployment gate.
  Rerunning the same checkpoint with `policy_gate_max_bad=-1` passed. The
  policy-sidecar lane has now replicated useful held-out action on two splits.
  Exported `policy_ranker.json` parity also passed: `215` valid groups,
  `1695` scores, max score drift `0.00024414`, and `0` argmax mismatches.
  Enyo branch `feature/policy-ranker-diagnostic` now has an artifact loader and
  C++ board-feature construction with Python-generated parity fixtures. It also
  has a disabled UCI `policyrank` diagnostic that scores legal root moves with
  current static NNUE evals and reports the gated sidecar override without
  changing `go`. On `startpos`, the current artifact already fires
  (`g1f3 -> e2e4`), so it is not deployment-safe. Do not integrate it into
  search until broad no-action calibration passes.
- Broad no-action diagnostic on the same exported checkpoint confirmed the
  problem. On non-`mate_like` groups, threshold `4` produced thousands of
  overrides and many bad ones; even much higher thresholds still had bad broad
  overrides. The next run must include non-mate preserve/no-action rows during
  training and must fail automatically if the deployed threshold has broad bad
  overrides.
- `policy-ranker-matelike-preserve-nonmate-board-h64-d35-pw50-lr2e4-e800`
  failed the broad no-action gate. The mate-like split gate still passed:
  validation threshold `4` gave `17` overrides, `16` good, `0` bad. But the
  broad gate at threshold `4` had `1231` overrides, `407` good, `489` bad, and
  worst harm `-800cp`. Threshold calibration alone did not solve it: even
  threshold `256` still had broad bad overrides. This proves the previous
  preserve sample was not enough and the sidecar confidence is not safety
  calibrated.
- `policy-ranker-matelike-preserve-all-nonmate-board-h64-d35-pw100-m8-t40-lr2e4-e400`
  also failed, but was much closer. It trained on all non-mate preserve rows
  except the preserve validation split. At threshold `40`, the broad gate had
  only `8` overrides, but `5` were bad and worst harm was `-535cp`. At the
  same threshold the mate-like validation split had only `3` good overrides,
  below the deployment-action requirement. Threshold `24` had enough held-out
  target action before the broad gate, so the next run keeps threshold `24` and
  removes the preserve validation split so every known broad row is a no-action
  training row.

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
- Static child-eval ranking is not a promotion gate by itself. A candidate can
  pass child model/engine gates, preserve broad static eval, and still lose
  games because root search selects different unscored moves.
- Every candidate that reaches local gates must now pass a root-search
  child-ranking gate built from failed smoke/replay positions before any game
  smoke.
- Use LC0 as a source of diverse positions and plausible candidate moves only
  after oracle rescoring. Do not train scalar eval directly from LC0 policy
  logits.
- Use replay JSONL as the preferred source for Enyo loss data because it stores
  scored legal moves and provenance in one format. CSV-era target files are
  legacy only.
- Each target group must contain one parent FEN, the oracle best move, the
  engine/logged bad move, and a few engine-plausible neighbors.
- Neighbor moves must be moves the engine actually considered or selected, not
  random legal moves.
- Oracle settings and `max_gap_cp` belong in the stored target data so target
  semantics do not change silently between runs.
- Broad preservation is a deadzone/leash, not a normal competing scalar
  objective.
- New candidates must use `./build.py create`; no manual training pipelines.
- A candidate must pass an early game smoke after target gates. Recent LC0 and
  replay-loss candidates improved engine target gates but failed or stayed
  neutral in games.

Secondary lanes:

- Architecture/features are paused until child-ranking can prove or disprove
  exported move-ranking learning on small groups.
- Stronger teacher data is useful only for high-value child groups first:
  disagreement, PV-instability, failure-suite, and high-loss move-choice rows.
- Static MAE/sign remains a rejection filter only.

## Current Result

- `r14` remains the keeper baseline. It was neutral versus Berserk in the
  latest 1000-game smoke (`-0.3 +/- 15.7`, LOS `48.3%`, draw `47.1%`).
- `r18` improved root-search gates after root-selected augmentation, but failed
  a 256-game smoke versus r14 (`-12.2 +/- 30.8`, LOS `21.7%`). The failed smoke
  produced `29` candidate-worse rows.
- `r19` trained those `29` rows as repeated primary targets. It regressed the
  combined search gate and scored only `9/29` on the focused smoke-worse rows.
- `r20` used those `29` rows as child-level reference guards instead. It
  protected broad scalar eval but still scored only `16/29` in root search,
  while r14 scores `27/29`.

Conclusion: the recent smoke regressions are not ordinary root child-eval
failures. They are positions where r14's correct move is produced by search
despite weak static child eval. Static root-child ranking and child-level
reference preservation cannot reliably preserve this behavior.

## Next Concrete Experiment

Rerun the narrow mate-like sidecar with deploy-gate checkpoint selection:

1. Keep scalar NNUE eval unchanged.
2. Use the Berserk reference net from `enyo/net/berserk-d43206fe90e4.nn` as the
   frozen base feature/eval source.
3. Train on `mate_like` rows only, excluding the unsafe
   `source:lichess_lowmat_mc12` slice from the hard validation gate.
4. Use all non-`mate_like` rows as no-action preservation rows.
5. Require held-out mate-like action with zero bad overrides and broad
   non-mate zero bad overrides at the deployed threshold.

Interpretation:

- If this passes, the next step is engine-side policy integration as a
  diagnostic only, not an automatic strength candidate.
- If this fails, pause simple sidecar MLP work. It has useful narrow signal but
  not a safe deployable action policy.

## Candidate Workflow

Normal candidate creation:

```sh
./build.py create -c build.json
```

Current `build.json` intent:

- candidate name:
  `child-ranking-mixed-replayloss2677-lc0oracle1000-smoker1x5-listwise-qfwd-refpreserve20-dz5-lr5e5-e360`
- backend: `child-ranking`
- target format: child-move groups with stored capped gaps
- base net: current reference `.nn`
- objective: listwise child ranking with quantized-forward export behavior
- broad preserve: reference-anchor deadzone (`weight=0.20`, `deadzone=5cp`)
- current target mix:
  - replay-loss dense rows;
  - LC0-oracle rows;
  - failed-smoke root-search rows weighted x5.
- hard gates:
  - model/engine child gates above reference baseline;
  - broad static gate;
  - root-search gate on failed-smoke rows before any game smoke.
- main knobs:
  - `policy_hidden`
  - `policy_feature_set`
  - `policy_dropout`
  - `policy_preserve_weight`
  - `policy_preserve_margin`
  - `policy_preserve_max_groups`
  - `policy_preserve_val_fraction`
  - `policy_broad_gate_max_bad`
  - `policy_broad_gate_max_overrides`
  - `policy_val_fraction`
  - `policy_target_temperature_cp`
  - `policy_thresholds`
  - `policy_gate_min_top1`
  - `policy_gate_max_bad`
  - `rank_temperature_cp`
  - `min_groups`

Current hypothesis:

- The sidecar can learn `mate_like` corrections, but without an explicit
  no-action loss it fires on normal positions. The next test is whether
  non-mate preserve rows can keep broad action near zero while retaining useful
  mate-like overrides. If this fails cleanly, pause simple sidecar MLP work and
  move to stronger policy data construction or a different representation.

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
