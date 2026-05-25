# Enyo NNUE Improvement Plan

This file is the active working plan. Long experiment notes are archived in
`docs/archive/IMPROVEMENT_HISTORY_20260523.md`.

Goal: produce a stronger Enyo net without repeating already-failed NNUE farming
loops.

Only `Current State`, `Closed Lanes`, `Active Bottleneck`,
`Next Research Task`, `Do Not Do`, `Gates`, and `Candidate Workflow` are
authoritative. Older result entries below are evidence, not invitations to
restart a lane.

## Current State

No trained Enyo net is currently a keeper.

The current Berserk-derived Enyo net appears locally saturated for ordinary
fine-tuning. Static MAE/sign can improve while search behavior and SPRT do not.
The main failure pattern is:

- head/output changes can improve narrow diagnostics, but fail broad gates or
  mate-like tails.
- normal input/L1 fine-tunes receive gradients but usually do not cross exported
  quantization thresholds.
- forced sparse movement is possible, but has not improved engine-search move
  choice.
- scratch/native training works technically. Current Enyo self-play volume has
  not recovered reference move-choice strength, but the first Stockfish-binpack
  native run was materially better than prior Enyo-selfplay scratch attempts.

Current `build.json` state:

- `native-searchaware-override-child-recover-lr2e7-e80` is rejected and left as
  the last completed reproducible config.
- Do not rerun or continue it. It proved that a small child-continuation target
  set can be retained after export (`149/160` top1, `160/160` top3), but the
  corrected native-32 `10k` unified search gate still failed the original
  parent (`414/1409` top1, `183` candidate-better vs `720` parent-better,
  capped `-65230`).
- `native-searchaware-unified-mpe-continue-lr1e6-e64` is also rejected.
- Do not rerun or continue it. It cleared the exported model gate
  (`647/1409` top1, `1065/1409` top3) and moved sparse tensors. The original
  engine-search rejection was invalid because it used an engine build that
  silently took the legacy loader path for a 32-bucket native net. Corrected
  native-32 validation still rejects the candidate, but less catastrophically:
  at `300k` nodes all top1 `560/1409`, top3 `914/1409`, compare all `207`
  candidate-better vs `553` reference-better, capped sum `-35908`, median
  nonzero diff `-28cp`, worst regression `-32000cp`.
- Corrected low-node native-32 gates show the model/search gap narrows with
  nodes but remains a fail. At `1k` nodes it scored all top1 `338/1409`,
  top3 `632/1409`, `177` candidate-better vs `855` reference-better, capped
  sum `-94698`. At `10k` nodes it scored all top1 `446/1409`, top3
  `765/1409`, `187` vs `728`, capped sum `-66065`.
- The active artifact is now the unified search-aware target corpus:
  `runs/native-searchaware-unified-targets-20260525/search_aware_unified_targets.jsonl`.
  It combines existing scored legal-move sources into `1409` deduped targets
  and `20067` scored moves; `235` are `mate_like`, `1174` are `non_mate`, and
  marker coverage is complete (`missing_marked=0`).
- The current reference engine baseline on that unified corpus at `300k` nodes
  is saved in
  `runs/native-searchaware-unified-targets-20260525/reference_gate_300k/reference.csv`:
  all top1 `761/1409`, top3 `1130/1409`; mate-like top1 `108/235`,
  non-mate top1 `653/1174`.
- The current preflight target is
  `runs/native-searchaware-reference-distill-targets-20260525/reference_distill_targets.jsonl`.
  It rewrites the unified corpus toward the reference engine's actual `300k`
  selected root move. It contains `1400` targets; `9` unified rows were skipped
  because the reference move was not present in the retained legal-move list.
  The rejected native parent scores only `491/1400` top1 and `856/1400` top3
  on this retargeted corpus, so this is now a stricter test of searched-policy
  imitation than the prior oracle child-eval gate.
- The last useful prior result remains diagnostic only: target-only search-policy
  overfit can move exported input/L1 and can exactly fit a 64-row policy slice,
  but target preservation did not clear the predeclared gate and is not
  playable.

## Closed Lanes

These lanes are closed for the current architecture/export format unless a new
mechanistic hypothesis is written before the run:

- head-only, output-only, material-head, and float-head fitting.
- sparse/input LR multiplier sweeps from existing exported weights.
- pairwise/local repair loops and target-only policy preservation.
- scalar child-row blends and same-data search-aware patching.
- bucket-index sweeps without a new feature geometry and parity/NPS proof.
- current same-architecture scratch scalar scaling as a promotion lane.
- Reckless existing-weight deltas that only affect dense/head tensors or a
  small bucket mask.

## Active Bottleneck

The current bottleneck is not ordinary scalar training loss. The bottleneck is
safe, export-visible representation movement plus search-aware supervision.

Do not treat data volume, teacher depth, LR, or head structure as the primary
unknown unless a run first explains how it should improve exported
representation movement and broad search move choice without creating tactical
tails.

Native work should now optimize for:

1. exported sparse/input/L1 movement that is intentional and measured.
2. broad move-choice and tactical-tail behavior before scalar MAE/sign.
3. CP/WDL/policy supervision together, not scalar-only training.
4. architecture/data hypotheses that reduce data starvation instead of adding
   another local correction layer.

Code audit status:

- PyTorch Kaiming input init already uses active-feature fan-in
  `sqrt(2 / 32)`, not full matrix fan-in.
- scalar training already supports MPE/WDL blending.
- search-aware training supports ranking/policy targets. As of
  `native-searchaware-unified-mpe-preflight-lr3e6-e48`, its broad scalar term
  can also use the same `mpe25`/WDL objective as normal scalar training.
  The old target-only preservation family remains closed. The follow-up
  continuation proves that exported model-gate improvement is not sufficient:
  search can collapse even when direct child-eval top1/top3 improve.
- `tools/validate/validate.py quant-scan` now reports whether `.pt` float
  movement crosses exported input/L1 integer boundaries before a candidate is
  considered for broader gates.
- remaining gap: define a scalable native training path that combines
  CP/WDL/policy with export-aware sparse movement checks, and pair it with a
  simpler/native feature geometry before spending GPU time.

## Track Definitions

`nnue_reckless` is the near-term Elo lane:

- start from existing Enyo/Berserk-derived weights.
- make small existing-weight-compatible architectural/objective changes.
- use Bullet only when it preserves the same lane goal.
- do not turn this into a scratch Reckless-style net.

`nnue_native` is the long-term provenance lane:

- no Berserk initialization.
- train an Enyo-owned net from scratch.
- may borrow architecture ideas, but the result is native to Enyo.
- currently useful for background experiments, not promotion.

## Next Research Task: 2026-05-25

Update: the one-bucket scale-up, same-architecture native scalar scaling, WDL
on test79, and target-only policy preservation are all rejected. Do not continue
scalar bucket-count continuations, local search-aware patching, WDL retunes, or
preservation-weight sweeps.

The input-factorized 32-bucket Bullet Enyo smoke is rejected. It was a concrete
feature-geometry/training change aimed at bucket data starvation, but it made
move-choice behavior worse on the external policy gate.

The controlled one-bucket Test80 data-source test is rejected. Better external
data alone did not recover move-choice behavior.

The real-8-runtime Test80 run is rejected. Test80 scalar training is worse than
the older Test79 scalar runs for this native architecture, both with one bucket
and with the best prior lower-bucket geometry. Stop scalar-only Test80 variants.

The next native task is target/objective design. Use the unified search-aware
target corpus as the broad gate and training-objective preflight source before
any new GPU training family.

`native-searchaware-unified-mpe-preflight-lr3e6-e48` completed. It is not a
promotion candidate. The run tested MPE/WDL broad supervision plus
policy/ranking child-move loss under quantized forward on the unified
`1409`-target corpus. It improved the exported model gate from the native init
baseline `337/1409` top1 and `594/1409` top3 to exact `.pt`/`.nn` parity at
`499/1409` top1 and `898/1409` top3, but failed the predeclared `600/1409`
hard gate. Net-diff and quant-scan confirm this was export-visible movement:
`2.81M/25.20M` exported values changed, including `11.161%` of input weights and
`3.665%` of L1 weights.

`native-searchaware-unified-mpe-continue-lr1e6-e64` is rejected. It was the
single allowed continuation of the mixed MPE/WDL/policy preflight. It improved
the exported model gate to `.pt`/`.nn` parity at `647/1409` top1 and
`1065/1409` top3, improved static scalar metrics on the next `100k` broad rows
(`mae 129.63 -> 111.47`, `sign 76.81% -> 80.40%` versus the parent), and kept
export-visible sparse movement (`1.72M` input weights, `290` L1 weights changed
versus the parent).

The original 300k-node engine-search gate is invalid and must not be cited as
evidence: the engine build used for that gate did not support native 32-bucket
loads and could silently fall back to the legacy loader on oversized `.nn`
files. Enyo commit `c99a7b1` fixes this by supporting native 32-bucket loads
and by requiring an exact legacy-size match before the legacy loader is used.
NNUE commits `169733e` and `d244611` add `--require-native-net-load` and pass it
through the validation wrappers.

Corrected native-32 engine gates still reject the candidate. At `1k` nodes:
all top1 `338/1409`, top3 `632/1409`, compare all `177` candidate-better vs
`855` reference-better, capped sum `-94698`, median nonzero diff `-78cp`,
worst regression `-32000cp`. At `10k` nodes: all top1 `446/1409`, top3
`765/1409`, `187` vs `728`, capped `-66065`, median `-50cp`. At `300k` nodes:
all top1 `560/1409`, top3 `914/1409`, mate-like top1 `109/235`, non-mate top1
`451/1174`, compare all `207` vs `553`, capped `-35908`, median `-28cp`,
worst regression `-32000cp`.

This is not an SPRT candidate. The correction changes the diagnosis from
"catastrophic collapse" to "direct model-gate progress does not transfer enough
to engine search." The `.nn` model-gate top1 is `647/1409`, but the corrected
300k search top1 is only `560/1409` and still loses badly to the reference
engine's `761/1409`.

Decision: no SPRT, no more continuation, and no more same-objective model-gate
training. Direct child-eval/model-gate improvement can be a false positive.
The model/search join explains the remaining loss. When corrected 300k search
chooses the same move as the exported model (`430/1409` positions), the
candidate is essentially neutral against reference: `60` candidate-better vs
`57` reference-better, capped sum `+618`, median nonzero diff `+1cp`, and worst
regression only `-383cp`. The failure is in the `979/1409` positions where
search overrides the model: `147` vs `496`, capped `-36526`. The clearest
subset is the `303` positions where the model's direct child eval has the
target as top1 but search does not play it; that subset is `28` vs `199`,
capped `-17453`, median `-67cp`, and includes a `-32000cp` tail. Most broad
loss comes from `lichess_policy` and `broad_nonmate` rows, not only mate-like
tails.

The next native step must build a search-stability objective or gate around the
model-top1/search-override failures. Do not spend GPU time on another
child-eval-only target run until the target construction includes the positions
where search refuses the model-preferred move.

Forced-search probe on the `303` model-top1/search-override positions confirms
that this is not just root move ordering noise. With `go nodes 300000
searchmoves <model_move>`, the forced model move scored worse than normal
search in `201/303` positions, equal in `27`, and better in `75`; median forced
minus normal score was `-45cp`, with `145` positions worse by at least `50cp`
and `46` worse by at least `200cp`. The same pattern holds on the
`reference_better` subset (`132/199` forced worse, median `-42cp`) and on both
`mate_like` and `non_mate` rows. Therefore the next target construction must
teach the net the searched continuation after the desired root move, not only
raise that root child at ply one.

`native-searchaware-continuation-mix-lr5e7-e16` is rejected. It continued from
the rejected parent and added repeated reference-depth-10 scalar labels for
`1143` forced-PV continuation positions from the `145` roots where the
model-preferred root move searched at least `50cp` worse than normal search.
The exported model gate stayed healthy and slightly improved over the parent
(`651/1409` top1, `1061/1409` top3, exact `.pt`/`.nn` parity, versus parent
`647/1409` and `1065/1409`). The 303-position override subset also improved:
parent was `28` candidate-better vs `199` reference-better, capped `-17453`;
the continuation mix was `49` vs `155`, capped `-11555`. But the unified 10k
gate did not improve: parent was `446/1409`, `187` vs `728`, capped `-66065`;
the continuation mix was `439/1409`, `187` vs `737`, capped `-66243`. No full
300k gate and no SPRT.

Conclusion: continuation scalar labels move the intended failure subset, but
they do not generalize to broad search. Do not repeat this as a scalar mix.
The next native attempt needs an explicit searched-policy/ranking objective or
a broader continuation corpus, not a small repeated PV-label blend.

`native-searchaware-reference-distill-init-lr5e7-e24` is rejected. It continued
from the rejected native-32 parent and changed the target meaning: rank 1 was
the reference engine's corrected `300k` root move, not the oracle child-eval
move. Broad rows distilled the parent net (`search_broad_target=init`,
`wdl_lambda=1.0`). The selected target-best checkpoint reached only
`486/1400` top1 during training and `498/1400` in the final exported `.pt`/`.nn`
model gate, versus parent baseline `491/1400` and the required `540/1400`.
The soft reference policy was too diffuse to move root choice.

`native-searchaware-reference-distill-hard-lr5e7-e32` is rejected. It used the
same reference engine root moves, but with a hard rank-1 policy
(`policy_temperature_cp=0`) and `50cp` minimum gap, weaker broad init
distillation (`search_broad_weight=10`), and stronger search-aware loss. The
selected checkpoint still reached only `497/1400` top1 in the exported model
gate, versus the required `580/1400`. Stronger policy pressure did not move the
root-choice boundary.

`native-searchaware-reference-distill-targetonly-lr1e6-e80` passed as a
diagnostic, not as a candidate. It removed broad preservation entirely
(`search_broad_weight=0`) and trained only the hard reference-root target
corpus. The selected checkpoint reached `1072/1400` top1 and `1351/1400` top3
with exact `.pt`/`.nn` parity, so the objective can fit the target corpus.
But broad drift against the parent rose to roughly `175cp`, so the resulting
net is not playable. The blocker is preservation/generalization, not target
plumbing.

`native-searchaware-reference-distill-recover-lr2e7-e80` passed target
retention, but is not the model to gate. It initialized from the target-only fit
and distilled broad rows back toward the rejected native parent. The target gate
passed at `1073/1400` top1, but `search_select_best_target` saved epoch `0`;
later epochs had lower broad drift (`~105cp`) while still retaining about
`1052/1400` target top1.

`native-searchaware-reference-distill-recover-final-lr2e7-e80` is rejected as
an engine candidate. It fixed the checkpoint-selection error and saved the
lower-drift final epoch: training ended around `105cp` broad drift while
retaining `1052/1400` target top1. The exported model gate passed with exact
`.pt`/`.nn` parity at `1064/1400` top1 and `1346/1400` top3. But corrected
native-32 `10k` search on the unified corpus failed against the parent:
top1 `425/1409`, top3 `785/1409`, and compare `181` candidate-better vs
`717` parent-better with capped sum `-65161`.

The useful diagnosis is that the exported model is better than search on the
same unified targets: model top1 was `682/1409`, but Enyo search followed the
model root move on only `442/1409`. In the subset where the model had the
target root move but search overrode it, the parent was better `345` times vs
only `9` candidate-better, capped `-40361`. Root-only search-aware fitting is
therefore not sufficient; the next test must train the continuation positions
that make search reject the static root preference.

`native-searchaware-override-child-targetonly-lr1e6-e120` passed as a
diagnostic. It used `160` child positions generated from the worst model-top1
/ search-override cases, scored at `50k` nodes per legal move and rebuilt as
`override_child_targets.jsonl` with `2166` move rows. Parent baseline on this
child corpus was `38/160` top1; the rejected recovered model was `45/160`.
The target-only run started from the recovered model, used no broad
preservation, and fit the child-continuation corpus with exact `.pt`/`.nn`
parity: `151/160` top1, `158/160` top3, sum gap `23cp`, worst gap `6cp`.
This validates the search-override failure theory, but the target-only model is
not a candidate because broad drift is uncontrolled.

`native-searchaware-override-child-recover-lr2e7-e80` is rejected as an engine
candidate. It initialized from the child target-only fit and distilled broad
rows back toward `native-searchaware-reference-distill-recover-final-lr2e7-e80`.
The exported `.pt`/`.nn` model gate retained the child-continuation signal with
exact parity: `149/160` top1, `160/160` top3, sum gap `22cp`, worst gap `6cp`.
But corrected native-32 `10k` search on the unified corpus still failed against
the original parent reference CSV: top1 `414/1409`, top3 `788/1409`,
compare `183` candidate-better vs `720` parent-better, capped sum `-65230`,
median nonzero diff `-51cp`, worst regression `-32000cp`. This is no 300k gate
and no SPRT.

Decision: close the small child-continuation patch family. The target
construction is learnable and export-stable, but `160` continuation positions
are too narrow to change broad search behavior. Any further native work must
either scale this idea into a broad search-stability corpus or change the
native architecture/data source; do not run another preservation-weight sweep
on the same child set.

Reckless remains paused until there is a new written hypothesis; the recent
existing-weight architectural deltas were rejected by confirmation or smoke.

Waste-control rule: do not start another NNUE training family from the rejected
families. No native or Reckless training is justified until the next build config
contains a concrete architecture change, data source, and gates. Reckless remains
paused.

Native lane:

- `native-bullet-test80-enyo8-runtime-cp-smoke-eval400-lr1e3-sb2048` is
  rejected. It tested the newer Test80 source with the prior best real
  8-runtime-bucket native geometry. Checkpoints `512`, `1024`, `1536`, and
  `2048` all failed badly; checkpoint `2048` was all top1 `195/800`,
  `69` vs `525`, capped `-76578`, worst regression `-32000cp`. Do not run
  more scalar-only Test80 bucket/data variants.
- `native-bullet-test80-enyo1-cp-smoke-eval400-lr1e3-sb2048` is rejected.
  It was a controlled data-source smoke with one shared train-time input bucket
  expanded to the current runtime layout. Checkpoints `512`, `1024`, `1536`,
  and `2048` stayed around `24.5-25.8%` top1 on the 800-target gate; checkpoint
  `2048` ended at all `70` vs `501`, capped `-67617`, worst regression
  `-32000cp`. Do not continue one-bucket CP-only Test80 training.
- `native-bullet-enyo16-sfbinpack-smoke-eval400-lr1e3-sb2048` is rejected.
  Best checkpoint was `2048`, but it still had all `91` vs `280`,
  capped `-20039`, and worst regression `-32000cp`.
- `native-bullet-enyo8-sfbinpack-smoke-eval400-lr1e3-sb2048` is rejected.
  Checkpoints `512` and `1024` stayed around `45.9%` top1 on the 800-target
  gate. The best broad row was still `97` vs `296`, capped `-22313`,
  and the `1024` mate-like row had worst regression `-31425cp`.
- `native-bullet-enyo1-sfbinpack-smoke-eval400-lr1e3-sb2048` is rejected as a
  promotion smoke, but is the best bucket-ladder signal. Checkpoint `1536`
  reached all `106` vs `246`, capped `-15116`, top1 `397/800`; this is still
  bad, but better than the longer 32-bucket checkpoint on the same 800-target
  gate.
- `native-bullet-lichess-eval-500k-wdl0-sb256` is rejected. The only useful
  checkpoint gate was a 200-target smoke at checkpoint `32`, and it collapsed:
  top1 `39/200`, `12` vs `146`, capped `-20416`, worst regression `-32000cp`.
  Do not scale direct Lichess eval scalar training as-is.
- `native-bullet-enyo1-sfbinpack-long-eval400-lr1e3-sb8192` is rejected.
  The run was stopped after checkpoint `1024`: all `71` vs `531`, top1
  `181/800`, capped `-73291`, worst regression `-32000cp`; non-mate was
  `66` vs `506`, capped `-70161`.
- `native-bullet-enyo16-h1280-sfbinpack-smoke-eval400-lr1e3-sb2048` is
  rejected. Training completed cleanly, but checkpoint `2048` was effectively
  flat against the 16-bucket/1024 baseline: all top1 `375/800`, `92` vs `285`,
  capped `-20283`, worst regression `-32000cp`; mate-like was worse than the
  parent (`4` vs `14`, capped `-1061`, worst `-31703cp`).
- `native-bullet-enyo8-runtime-sfbinpack-smoke-eval400-lr1e3-sb2048` is
  rejected. It used `ENYO_NNUE_BUCKETS=8` in the engine and
  `bullet_enyo_runtime_input_buckets=8` in the exporter, so this was a real
  runtime layout change, not a low-bucket net expanded at load time. The best
  checkpoint was `1536`: all top1 `371/800`, `82` vs `277`, capped `-19627`,
  worst regression `-32000cp`; non-mate was `76` vs `268`, capped `-19431`.
  Checkpoints `512`, `1024`, and `2048` were also clear fails.
- `native-bullet-lichess-filtered-5m-eval400-lr1e3-sb1024` is rejected.
  This was a scratch native scalar run on a material-filtered, score-balanced
  Lichess eval DB slice. The source import worked, writing `276830` rows from
  the first `5M` input rows, but checkpoint `256` failed the 800-target
  external gate decisively: all top1 `205/800`, `77` vs `519`, capped
  `-7207724`, median nonzero `-257cp`, worst regression `-32000cp`. The
  remaining checkpoint sweep was stopped.
- `native-bullet-sfbinpack-scratch-long-eval400-lr1e3-sb32768` is rejected.
- best checkpoint-sweep rows had positive capped sums but still retained
  mate-like catastrophic tails around `-31k cp`; e.g. checkpoint `12288`
  had all `68` vs `58`, capped `+1142`, but mate-like worst regression
  `-31311cp`; checkpoint `24576` had all `63` vs `60`, capped `+856`,
  but mate-like worst regression `-31804cp`.
- `native-bullet-sfbinpack-tailmix-12288-lr1e5-sb4096` is rejected.
- every checkpoint is worse than the parent gate; `4096` ended at
  `41` vs `97`, capped `-6335`, with mate-like worst regression `-31814cp`.
- stop the old 32-bucket long scratch, tailmix, search-aware patching, and
  same-architecture LR continuation families.
- do not spend more GPU time on bucket-count scalar continuations. The next GPU
  run must follow a committed export-visible architecture diff or a materially
  new data-scale plan.

Reckless lane:

- bucket 6 is rejected by confirmation:
  `-4.5 +/- 7.7 Elo`, `LOS=12.4%`, `draw=49.5%` over `4000` games.
- bucket 7 was the only gate-clean checkbucket candidate:
  1M search gate `2` vs `0`, capped `+25`, worst regression `0`;
  failure-suite replay is exact parity over `913` positions.
- smoke was weakly positive:
  `+5.2 +/- 15.2 Elo`, `LLR=+0.23/2.94`, `LOS=75.0%`, `draw=50.5%`
  over `1000` games.
- confirmation rejected it:
  `-1.7 +/- 7.5 Elo`, `LLR=-0.83/2.94`, `LOS=32.4%`, `draw=51.9%`
  over `4000` games.
- close the checkbucket family. Do not run bucket 8 or another bucket-index
  sweep without a new architecture hypothesis.
- `floathead_delta_020.nn` is rejected. It narrowly passed the external
  Lichess-policy gate, but match smoke finished `-6.2 +/- 14.9 Elo`,
  `LLR=-0.86/2.94`, `LOS=20.6%`, `draw=52.0%` over `1000` games. The
  follow-up smoke failure mine found slightly more local wins than reference
  (`33` vs `25`) but a negative capped result (`-275cp` at `100cp` cap) and
  huge losing outliers (`worst_regression_cp=-31433`).
- `reckless-enyo32-input-threshold-clampfix-lr3e4-sb16` is rejected. It was
  the only recent exported-input-movement probe with a positive cheap cp16
  search gate (`34` vs `30`, capped `+740`, worst regression `-125cp`), but
  the 1k smoke finished `-10.8 +/- 14.4 Elo`, `LLR=-1.33/2.94`,
  `LOS=7.1%`, `draw=55.3%`.

Current next action:

- no Reckless training or SPRT run is justified until a new written hypothesis exists.
- native has no active training config.
- `native-bullet-enyo1-sfbinpack-wdl075-smoke-eval400-lr1e3-sb2048` is
  rejected. Checkpoint 512 scored top1 `185/800`, candidate_better `59` vs
  reference_better `532`, capped sum `-74484`, worst regression `-32000`.
  The comparable CP-only 1-bucket checkpoint 512 was much better at top1
  `369/800`, candidate_better `93` vs reference_better `289`, capped sum
  `-21127`.
- SF-binpack result/WDL semantics were audited with `audit_sfbinpack` and do
  not look inverted. The failed uncapped stream is simply bad for WDL-heavy
  scratch training: first 1M filtered rows were `56.43%` STM losses, only
  `13.75%` draws, mean score `-1131cp`, mean abs score `3254cp`, and `59.55%`
  of rows were above `1600cp`.
- the cap400 smoke used the same loader with `max_abs_cp=400`, which audited
  as `76.07%` draws, mean score `-17cp`, mean abs score `119cp`, and
  score/result agreement `83.27%` on non-draws. It used a 64 MB loader buffer
  because the first cap400 attempt with the default 1024 MB buffer spent too
  long filling the shuffle buffer before producing smoke feedback.
- `native-bullet-enyo1-sfbinpack-cap400-wdl075-buf64-smoke-eval400-lr1e3-sb1024`
  is rejected. Checkpoint 256 scored top1 `135/800`, candidate_better `43` vs
  reference_better `600`, capped sum `-94397`, worst regression `-32000`.
  This is worse than both the uncapped WDL smoke and the CP-only 1-bucket
  baseline, so WDL on the test79 source is closed for now.
- the Bullet Enyo layout audit/fix is complete. The Bullet Enyo trainer had
  been hard-coded to the legacy 16-king-bucket input layout while the
  documented/runtime native design is 32 buckets. That means older "native"
  Bullet `.nn` runs produced legacy-layout payloads that Enyo expanded at load
  time.
- do not assume 32 buckets is the right scratch-training start. The 16-, 8-,
  and 1-bucket scalar rungs all failed as promotion paths. A lower bucket count
  may still be part of a native architecture, but it must be paired with a real
  architecture hypothesis instead of another same-loss continuation.
- 32-bucket init/export parity passed in
  `runs/bullet-enyo-32-init-roundtrip-20260524_031011`: exported size
  `50368836`, and `net_diff --float-atol 1e-6 --fail-if-different` reported
  zero changed tensors. Explicit 16-bucket legacy parity also passed in
  `runs/bullet-enyo-16-init-roundtrip-20260524_031037`.
- additional Bullet tooling fix: trainable Enyo `l0w`/`l0b` must use full
  int16 optimizer bounds, not `+/-4095`. Existing Enyo nets legally contain
  larger accumulator values; the narrower clamp corrupts existing-weight
  checkpoints on the first saved superbatch. Reckless clampfix audit verified
  checkpoints `0` through `8` keep input/L1/L2 tensors unchanged, with only
  `<=1.49e-8` float roundtrip noise in four output weights.
- rejected smoke:
  `native-bullet-enyo32-sfbinpack-smoke-eval400-lr1e3-sb2048`.
  It was the first true 32-bucket Bullet Enyo scratch run. Checkpoint `512`
  failed the 300k-node search gate (`34` vs `150`, capped `-18672`,
  worst `-31837cp`, top1 `27/232`). Checkpoint `1024` also failed
  (`35` vs `142`, capped `-17815`, worst `-31803cp`, top1 `43/232`).
  The sweep was stopped before `1536/2048` to avoid wasting compute.

Decision: no SPRT, no longer 32-bucket scratch continuation. The Bullet Enyo
layout bug is fixed, but true 32-bucket scratch still does not recover
reference move-choice strength at this data scale.


Latest native bucket-ladder result:

- `native-bullet-enyo1-sfbinpack-long-eval400-lr1e3-sb8192` is rejected.
- checkpoint `1024` was tested against the cached 800-target reference gate.
- all split: top1 `181/800`, top3 `346/800`, `71` vs `531`, capped
  `-73291`, worst regression `-32000cp`.
- mate-like split: top1 `30/66`, `5` vs `25`, capped `-3130`, worst
  regression `-31991cp`.
- non-mate split: top1 `151/734`, `66` vs `506`, capped `-70161`, worst
  regression `-32000cp`.

Decision: stop this run and close scalar bucket-count continuation as the next
near-term native lane. The next native attempt must be a clean architecture
branch with parity/NPS gates before training.

Latest native data-source smoke:

- `native-bullet-lichess-eval-500k-wdl0-sb256` is rejected.
- data: high-depth Lichess eval DB rows, material-independent signed buckets,
  scanning up to `5M` input rows for about `500k` balanced training rows.
- result: checkpoint `32` on a 200-target smoke had top1 `39/200`, `12` vs
  `146`, capped `-20416`, and worst regression `-32000cp`.
- decision: the importer is useful tooling, but direct Lichess scalar eval
  training is not a near-term native promotion path without a new sampling or
  objective hypothesis.


## Data-Scale Audit: Lichess Eval DB

Run directory: `runs/native-lichess-eval-audit-20260524`.

Findings:

- source: `/home/petter/code/cpp/chess/assets/lichess_db_eval.jsonl.zst`.
- the source mostly has four-field FENs, so true ply is unavailable after
  normalization; use material-count filters, not `min_ply`, for this source.
- 100k-input smoke reached `10000` eligible rows after only `33367` input rows.
- 1M-input balanced-bucket probe saw `186419` eligible rows and wrote `46392`
  sampled rows.
- sign balance was good after bucketing: `20712` positive, `20680` negative,
  `5000` zero.
- all requested buckets up to `300-800cp` filled; rare `800-1600cp` buckets
  were naturally sparse (`712` positive, `680` negative from first 1M input
  rows).
- material-filtered scalar audit
  `runs/native-lichess-eval-filter-audit-20260524/import.log` wrote `276830`
  Bullet-text rows from the first `5M` input rows using material count `10-30`,
  depth `>=18`, knodes `>=100000`, and abs cp `<=1200`. Buckets up to
  `25-100cp` filled; larger score buckets were sparse.

Decision:

- this is usable tooling, but not a proven training source.
- the `46k`, `500k`, and material-filtered `276k` scalar-training smokes all
  failed search gates badly.
- do not scale Lichess eval scalar training without a new sampling/objective
  hypothesis.
- do not create a huge JSONL -> Bullet text -> Bullet data chain as the normal
  path; disk pressure is already a problem.

## Native Lichess Policy Targets

Target construction is now available through `build.py target-score` with
`target_score.lichess_eval_input`. The importer normalizes the Lichess eval
DB four-field FENs to six-field FENs before legal-move scoring.

Smoke results:

- `native-lichess-policy-targets-smoke-20260524` was invalid: imported
  `350` rows but scored `0` targets because four-field FENs were rejected by
  `python-chess`.
- `native-lichess-policy-targets-smoke2-20260524` fixed FEN normalization and
  produced `120` targets / `3541` scored legal moves, but the source lacks true
  ply counters and the sample was opening-heavy (`63/120` opening by material).
- `native-lichess-policy-targets-smoke3-20260524` added
  `max_material_count=26`; it produced `120` targets / `3488` scored legal
  moves and removed opening-tagged targets: `70` midgame, `34` late,
  `16` endgame, `9` mate-like.

Decision:

- material-count filtering is required for Lichess eval policy-target use; do
  not rely on `min_ply` for this source.
- `native-lichess-policy-targets-800-mc26-20260524` produced `800` targets /
  `8833` child moves with no opening-tagged targets: `465` midgame, `244`
  late, `91` endgame, `66` mate-like.
- reference baseline at `300k` nodes is sane: `top1=506/800` (`63.2%`) and
  `top3=690/800` (`86.2%`), with mate-like `top1=47/66` (`71.2%`).
- native scratch checkpoint `12288` is rejected on this external gate:
  `top1=387/800`; compare `90` vs `263`, capped `-18472`,
  median nonzero `-32cp`, worst regression `-32000cp`.
- native scratch checkpoint `24576` is also rejected:
  `top1=386/800`; compare `86` vs `249`, capped `-18489`,
  median nonzero `-35cp`, worst regression `-32000cp`.
- reckless `bucket_7.nn` is rejected on this external gate despite the neutral
  4k SPRT confirmation: `top1=185/800`; compare `56` vs `537`, capped
  `-74691`, median nonzero `-211cp`, worst regression `-32000cp`.
- `floathead_delta_020.nn` passes the external gate narrowly but fails match
  smoke: external gate `top1=508/800`, compare `69` vs `71`, capped `+400`;
  smoke SPRT finished `-6.2 +/- 14.9 Elo`, `LLR=-0.86/2.94`, `LOS=20.6%`,
  `draw=52.0%` over `1000` games.
- do not train from these targets yet. Use the target set first as a cheap
  external-policy sanity gate for future candidates.
- `search_target_gate.py` now accepts `--reference-csv` so future candidate
  sweeps can reuse the cached reference pass instead of rerunning it.

## Latest Result

Rejected true 32-bucket native Bullet smoke:

- `native-bullet-enyo32-sfbinpack-smoke-eval400-lr1e3-sb2048` used
  `bullet_enyo_input_buckets=32`, no init net, SF-binpack input, and exported
  full-size native `.nn` checkpoints (`50368836` byte payload).
- training completed successfully in about 20 minutes and wrote checkpoints
  `512`, `1024`, `1536`, and `2048`.
- checkpoint gate was stopped after `1024` because both completed checkpoints
  were catastrophic against the reference:
  - `512`: all `34` vs `150`, capped `-18672`, worst `-31837cp`;
    non-mate `10` vs `107`, capped `-14834`.
  - `1024`: all `35` vs `142`, capped `-17815`, worst `-31803cp`;
    non-mate `10` vs `102`, capped `-14886`.
- conclusion: fixing the Bullet layout did not make same-architecture native
  scratch a near-term Elo lane. Do not continue this family without a much
  larger data-scale plan or a different architecture objective.

Rejected conservative native SF-binpack retry:

- `native-bullet-sfbinpack-continue3-eval400-lr1e5-sb4096` retried the same
  native SF-binpack continuation with lower pressure: `lr=1e-5 -> 1e-6`, no
  weight decay, `4096` superbatches, checkpoints every `512`.
- checkpoint `0` was exact parity, so init/export was correct.
- no checkpoint passed the 300k-node search gate.
- best broad-looking checkpoint was `1536`: all split `37` vs `36`,
  capped `+144`; non-mate `29` vs `24`, capped `+323`; but mate-like failed
  `8` vs `12`, capped `-179`, worst `-7098cp`.
- checkpoint `3072` had the strongest non-mate split (`30` vs `21`,
  capped `+247`) but failed overall (`capped=-727`) and mate-like
  (`5` vs `15`, capped `-974`, worst `-7098cp`).
- checkpoint `4096` had positive non-mate capped sum (`+309`) but failed all
  (`35` vs `39`, capped `-181`) and mate-like (`11` vs `16`,
  capped `-490`, worst `-7098cp`).
- net-diff shows the low-pressure retry was dense-only after export:
  checkpoints `1536`, `3072`, and `4096` changed `0` input weights,
  `0` input biases, `0` L1 weights, and `0` L1 biases; only L2/output tensors
  moved (`577/25200209` exported values).

Comparison to the rejected higher-pressure continuation:

- `continue2` checkpoints crossed sparse export thresholds:
  `5120` changed `41767` input weights and `450` L1 weights; `7168` changed
  `82197` input weights and `576` L1 weights.
- those sparse-moving checkpoints still failed the deeper 1M-node gate:
  `5120` lost non-mate (`23` vs `26`, capped `-320`, worst `-305cp`);
  `7168` failed all (`capped=-71`, worst `-397cp`) and non-mate
  (`capped=-598`, worst `-397cp`).

Decision: no SPRT. Stop this native SF-binpack continuation family. The result
is the same structural tradeoff seen elsewhere: exported sparse movement is
possible, but destructive; conservative training is exported dense-only and
cannot fix the search tails. The next training attempt needs a different
architecture/data-scale plan, not another LR continuation.

Rejected long native SF-binpack scratch:

- `native-bullet-sfbinpack-scratch-long-eval400-lr1e3-sb32768`.
- no `init_net`; this is an Enyo-owned native scratch run.
- same native architecture (`1024` hidden, `16` L2) so the test isolated data
  scale and scratch learning, not architecture.
- `32768` superbatches, checkpoint every `2048`.
- checkpoint sweep completed after training finished.
- no checkpoint was keeper-safe. Checkpoints `12288` and `24576` had the best
  broad-looking all-split results (`68` vs `58`, capped `+1142`; and `63` vs
  `60`, capped `+856`), but both retained unacceptable mate-like tails
  (`worst_regression_cp=-31311` and `-31804`).

Decision: no SPRT. Stop same-architecture native scratch on this data recipe.
The next native attempt must change the data-scale plan materially or introduce
an export-visible architecture hypothesis.

Rejected native SF-binpack continuation:

- `native-bullet-sfbinpack-continue2-eval400-lr5e5-sb8192` continued the best
  previous native checkpoint on external SF-binpack data for `8192`
  superbatches.
- shallow checkpoint sweep showed target top1/top3 gains, but the deeper
  1M-node gate rejected both plausible checkpoints.
- checkpoint `5120`: all split was neutral by counts (`36` vs `36`) with
  positive capped sum (`+677`), but non-mate regressed (`23` vs `26`,
  capped `-320`, worst `-305cp`).
- checkpoint `7168`: better by counts (`41` vs `36`) but failed capped/tail
  behavior (`capped=-71`, worst `-397cp`); non-mate was also negative
  (`28` vs `26`, capped `-598`, worst `-397cp`).

Decision: no SPRT. The run learned some search-target signal, but too much
pressure still creates broad/non-mate regressions. Retry only once with a more
conservative continuation (`lr=1e-5 -> 1e-6`, no weight decay, `4096`
superbatches, checkpoints every `512`); if that also fails deep gates, stop this
native SF-binpack continuation family and reassess architecture/data scale.

Rejected native broad-target search-aware run:

- `native-searchaware-broadtargets-w2-lr5e7-e8` used `250` broad non-mate
  move-choice targets built from ordinary training positions.
- the internal model gate improved the target set (`top1=67/244`,
  `top3=129/244`, `sum_gap_cp=11983`) and `.pt` matched exported `.nn`.
- exported movement was still dense-only: `215/25200209` values changed, all in
  L2 float tensors; input and L1 were unchanged.
- static validation collapsed (`mae=784.736`, `sign=72.97%`,
  `bias=-308.248`) on held-out packed rows.
- broad 300k-node search gate was worse than the reference
  (`candidate_better=34`, `reference_better=41`,
  `capped_sum_diff_cp=-132`, `worst_regression_cp=-32000`).

Decision: no SPRT. The broad target set is useful, but this objective/config is
not. Native search-aware training must preserve teacher static behavior and
produce intentional exported representation movement before another candidate.

Rejected teacher-preserved broad-target retry:

- `native-searchaware-broadtargets-teacher-w10-lr2e7-e6` restored broad scalar
  supervision to teacher labels and reduced search-target pressure.
- internal model gate did not improve (`top1=66/244`, `top3=126/244`,
  `sum_gap_cp=12811`).
- export again moved only dense L2 floats (`210/25200209` changed; input/L1
  unchanged).
- static remained rejection-level (`mae=753.923`, `sign=74.00%`,
  `bias=-200.817`).
- broad 300k-node search gate failed (`candidate_better=42`,
  `reference_better=53`, `capped_sum_diff_cp=-346`,
  `worst_regression_cp=-32000`).

Decision: stop this search-aware/native config family. Do not spend more runs
on LR/objective/weight tweaks around the same broad target JSONL.

Search-aware target construction and training are implemented on
`feature/nnue-search-aware-targets`:

- target set: `assets/failure_suite/search_aware_targets_20260523.jsonl`.
- size: `232` positions and `5,345` scored legal moves.
- reference-vs-reference search-gate sanity check has zero compare diffs.

Rejected search-aware sparse diagnostics:

- `search-aware-existing-preflight-lr1e7-e2`: exported only dense/output floats;
  search-gate failed (`candidate_better=39`, `reference_better=146`,
  `capped_sum_diff_cp=-17929`).
- `search-aware-sparseprobe-init-in800-l1x1000-e2`: moved sparse floats but
  exported an identical `.nn`.
- `search-aware-sparsecross-init-in3000-l1x3000-e2`: crossed export thresholds
  (`90/25200209` changed) but search-gate was unchanged and still failed.
- `search-aware-targetonly-sparse-overfit-in3000-l1x3000-e8`: moved
  `523527/25200209` exported values and improved internal child-ranking/static
  metrics, but engine-search gate still failed (`candidate_better=41`,
  `reference_better=147`, `capped_sum_diff_cp=-18167`).

Decision: stop search-aware sparse LR/multiplier sweeps. If this lane continues,
change target construction or objective so success is measured by engine-search
move choice, not only child-eval ranking.

## Latest Result: Bullet SF-binpack

`build.json` now defines `bullet-sfbinpack-legacy-floathead-lr1e7-sb1`.

Purpose:

- use a proven external Stockfish NNUE binpack as the next data source.
- keep the near-term lane existing-weight based.
- avoid another Enyo self-play or JSONL relabeling cycle.
- use Bullet directly on SF binpack data after first proving the legacy-layout
  Enyo init export is behaviorally identical to the reference.

Init-export parity passed:

- exported net size matched the current reference exactly (`25203012` bytes).
- all integer tensors matched; only four output floats differed by `1.49e-08`.
- search-gate compare was exactly zero on all/mate-like/non-mate subsets.

Rejected SF-binpack pressure setting:

- `bullet-sfbinpack-legacy-existing-init-preflight`: two-superbatch
  `lr=1e-6 -> 2e-7`, weight decay `1e-6`.
- exported movement: `505` input weights, no L1 movement, dense/output moved.
- search gate failed (`top1=84/232`, `capped_sum_diff_cp=-2749`,
  mate-like worst regression `-31591`).
- checkpoint 1 was also rejected (`top1=87/232`, `capped_sum_diff_cp=-2788`,
  mate-like worst regression `-31591`).

Rejected lower-pressure all-weight setting:

- `bullet-sfbinpack-legacy-lr1e7-sb1-nodecay`: one superbatch,
  `lr=1e-7 -> 2e-8`, no weight decay.
- exported movement still crossed `505` input weights, plus dense/output moved.
- search gate failed (`top1=82/232`, `capped_sum_diff_cp=-928`,
  mate-like worst regression `-31248`).
- non-mate subset was mixed (`candidate_better=32`, `reference_better=26`,
  worst regression `-155`), but top1 still regressed (`49/128` vs `53/128`).

Rejected float-head setting:

- `bullet-sfbinpack-legacy-floathead-lr1e7-sb1`: one superbatch,
  `lr=1e-7 -> 2e-8`, no weight decay, `trainable=float-head`.
- `net-diff` was intentional: sparse/L1 stayed identical; dense/output floats
  moved.
- search gate improved top1 overall (`91/232` vs reference baseline `89/232`)
  and non-mate top1 (`60/128` vs `53/128`), but failed mate-like behavior.
- failure-suite replay rejected it: `positions=913`, `candidate_better=73`,
  `reference_better=68`, `sum_diff_cp=-627`, `worst_regression_cp=-512`.

Scaled float-head deltas were also rejected:

- `25%` delta: search gate looked best (`top1=101/232`,
  `candidate_better=57`, `reference_better=30`,
  `capped_sum_diff_cp=+1301`), but failure-suite tail vetoed it
  (`sum_diff_cp=+1060`, `worst_regression_cp=-511`).
- `20%` delta: search gate stayed positive (`candidate_better=53`,
  `reference_better=33`, `capped_sum_diff_cp=+933`), but failure-suite replay
  rejected it (`sum_diff_cp=-559`, `worst_regression_cp=-512`).
- `5%`, `10%`, `15%`, `50%`, and `75%` deltas did not improve the overall gate
  enough to justify replay or SPRT.

Decision: no SPRT. Stop SF-binpack float-head delta scaling as a near-term Elo
lane. It contains some broad signal, but repeatedly creates or preserves
unacceptable mate/endgame tails.

Rejected output-only signfit delta scaling:

- `reckless-output-signfit-delta-scale2-20260523_080836`: scales `0.25`,
  `0.50`, `0.75`, `1.00`, and `1.25` were bit-equivalent in engine-search
  behavior on the broad search gate: `top1=46/232`,
  `candidate_better=36`, `reference_better=140`,
  `capped_sum_diff_cp=-18244`, `worst_regression_cp=-31837`.
- The least-bad scale then failed failure-suite replay:
  `positions=913`, `candidate_better=78`, `reference_better=497`,
  `sum_diff_cp=-123658`, `median_nonzero_diff_cp=-179`,
  `worst_regression_cp=-923`.

Decision: no SPRT. Output-only deltas are too weak for the current broad gate
and can make failure-suite behavior much worse despite zero runtime overhead.

## Latest Result: Native SF-binpack Continuation

`native-bullet-sfbinpack-continue-eval400-lr2e4-sb4096`

Purpose was to:

- continue the native SF-binpack lane after the first run showed clear
  baseline-building progress but failed promotion gates.
- initialize from the Enyo-owned
  `native-bullet-sfbinpack-scratch-eval400-sb4096/model.nn`.
- keep the data source and architecture fixed.
- use a lower continuation LR and save checkpoints for the same cheap move-gate
  sweep before any broader validation.

This run is a provenance/native-baseline test, not a promotion candidate by
default.

Checkpoint sweep selected `2048` as the least-bad checkpoint, but it still
failed the original search gate:

- `top1=68/232`, `candidate_better=58`, `reference_better=86`,
  `capped_sum_diff_cp=-4925`.
- the old gate showed a repeated `-31814cp` mate-like tail.

The search gate was then fixed to send `ucinewgame`/`isready` before each target
instead of reusing one UCI game/hash across unrelated positions. Fresh-hash
validation for checkpoint `2048` gave:

- `300k nodes`: `top1=65/232`, `candidate_better=39`,
  `reference_better=86`, `capped_sum_diff_cp=-5202`.
- `1M nodes`: `top1=78/232`, `candidate_better=48`,
  `reference_better=73`, `capped_sum_diff_cp=-2436`,
  `worst_regression_cp=-276`.
- `1M mate-like`: `candidate_better=21`, `reference_better=16`,
  `capped_sum_diff_cp=+503`, `worst_regression_cp=-237`.
- `1M non-mate`: `candidate_better=27`, `reference_better=57`,
  `capped_sum_diff_cp=-2939`, `median_nonzero_diff_cp=-25`.

Decision: no SPRT. Stop scalar native SF-binpack continuation. The useful next
diagnostic is non-mate search behavior, not another same-objective continuation
run.

Follow-up non-mate diagnostic:

- `native-nonmate-deep-diagnosis-20260523_055831`
- `3M nodes` on the `128` non-mate targets:
  `top1=42/128`, `candidate_better=21`, `reference_better=60`,
  `capped_sum_diff_cp=-3397`, `median_nonzero_diff_cp=-23`.

Decision: deeper search did not close the non-mate gap. The next native lane
needs a changed target/objective, not more scalar SF-binpack epochs.

## Latest Result: Native Search-Aware Non-Mate

`native-searchaware-nonmate-initdistill-w8-lr1e6-e6`

Purpose was to start from the best Enyo-owned SF-binpack checkpoint and train a
non-mate-weighted search-aware objective with broad init-net distillation as a
guardrail.

Result:

- training finished quickly, but internal target metrics did not improve:
  `target_top1` stayed around `32-33/232`.
- `net-diff`: exported input/L1/output tensors did not move. Only
  `191/512` L2 weights and `24/32` L2 biases changed.
- fresh-hash `300k` search gate failed badly:
  `top1=39/232`, `candidate_better=31`, `reference_better=151`,
  `capped_sum_diff_cp=-20372`.
- `non_mate`: `top1=12/128`, `candidate_better=13`,
  `reference_better=108`, `capped_sum_diff_cp=-15887`.
- `mate_like`: `capped_sum_diff_cp=-4485`, with reopened `-31924cp` tail.

Decision: no SPRT and no broader replay. This is not an Elo candidate and not a
useful native baseline. The next work is to fix search-aware objective behavior
so it can improve training-set move choice and move intended exported tensors;
do not launch another weighted-search config with the current plumbing.

Follow-up search-aware plumbing audits:

- `search-aware-target-overfit-audit-20260523_063642`: 16 non-mate targets,
  quantized target-only, no broad loss. It moved exported input/L1/dense values
  (`183835/25200209` total changed), but only reached `8/16` top1 after 80
  epochs and damaged broad init-net agreement. This proves exported sparse
  movement is possible, but the schedule is not yet a usable candidate recipe.
- `search-aware-4target-forward-audit-20260523_063827`: 4 non-mate targets,
  target-only. Float and quantized paths both overfit from `0/4` to `4/4`.
  This rules out a simple sign/orientation bug in the child-eval target loss.
- `search-aware-16target-quantized-audit-20260523_064051`: 16 non-mate
  targets, quantized target-only. It overfit from `0/16` to `16/16`, but broad
  init-net MAE drifted to about `1653cp`.
- `search-aware-16target-preserve-audit-20260523_064326`: simultaneous broad
  init-net preservation pins the targets. `w=0.05` ended at `1/16` top1 with
  broad MAE `8.10cp`; `w=0.20` ended at `1/16` top1 with broad MAE `28.15cp`.
- `native-searchaware-stagewarmup-audit16-lr3e6-e24`: first staged
  warmup/ramp attempt. It still failed to cross the target-choice boundary
  (`1/16` final top1), ended with broad MAE `39.89cp`, and exported only
  dense/output movement (`165/25200209` total changed; no input/L1 movement).
- `native-searchaware-stagewarmup-strong16-lr3e5-e96`: stronger staged
  warmup/ramp moved exported input/L1 (`169656/25200209` total changed), but
  still ended at `1/16` top1 with broad MAE about `2398cp`. This exposed a
  tooling gap: `search_target_limit=16` selected the first 16 mixed targets,
  not the intended first 16 non-mate targets.
- `native-searchaware-stagewarmup-strong16-lr3e5-e96` tag-filtered rerun:
  correctly used the first 16 non-mate targets and moved exported sparse/L1
  heavily (`2802198/25200209` total changed), but ended at `0/16` top1,
  `1/16` top3, and broad MAE about `150cp`. The schedule is destructive:
  sparse movement alone is not useful if it does not learn the search targets.
- `native-searchaware-recover16-w002-lr1e6-e80`: initialized from the previous
  target-fit checkpoint and distilled broad rows against the separate native
  checkpoint. It recovered broad MAE from about `2032cp` to `597cp`, but target
  choice stayed at `0/16` top1 and `1/16` top3. Later CPU reload checks showed
  the target-fit checkpoint was valid; the recovery run itself failed to retain
  those target choices.
- `native-searchaware-targetonly-exportcheck16-lr3e5-e240`: originally failed
  the required saved/exported target gate because the gate used the CUDA audit
  path. Manual CPU validation of the same saved `.pt` and exported `.nn`
  artifacts passes at `16/16` top1, `16/16` top3, and zero gap for both
  models. The first 16 non-mate target IDs and first 8 moves matched the
  earlier audit, so this was not a target-file mismatch.
- Reloading the old `search-aware-16target-quantized-audit-20260523_064051`
  artifacts also passes at `16/16` on CPU. The apparent `0/16` reload failure
  was CUDA validation divergence, not non-persistent exported artifacts.
- `native-searchaware-rootpolarity-exportcheck16-lr3e5-e80`: rejected. Best
  selected checkpoint was only `6/16` in-run, and CPU revalidation reached only
  `8/16`. Root-high scoring is not the path forward.

Decision: exported search-target model gates must run on CPU unless the CUDA
forward path is independently fixed. CPU quantized validation matches exported
`.nn` semantics; CUDA validation currently gives misleading failures for these
tiny target audits. For the next small search-aware audit, train on CPU too, so
training metrics and exported validation use the same semantics.

Follow-up CPU recovery audit:

- `native-searchaware-cpurecover16-w0p001-lr1e6-e80`: retained the tiny target
  set on CPU after export (`16/16` top1/top3), and intentionally moved exported
  sparse/L1 values (`176432/25200209` total changed). Broad behavior collapsed:
  static `mae=745.077`, `sign=57.47%`; fresh-hash `300k` search gate was
  `top1=40/232`, `candidate_better=32`, `reference_better=151`,
  `capped_sum_diff_cp=-19669`, `worst_regression_cp=-31924`. Non-mate was
  especially bad (`top1=11/128`, `reference_better=106`).

Decision: no SPRT. Stop the direct child-eval search-aware recovery lane for
now. Passing tiny target retention is not enough; broad engine-search behavior
must be part of the objective/gate before this lane resumes.

## Latest Result: Native SF-binpack Scratch

`native-bullet-sfbinpack-scratch-eval400-sb4096`

Completed with Bullet directly on the Stockfish NNUE binpack:

- no Berserk initialization.
- final training loss: about `0.0266`.
- training time: `0h 41m 47s`.
- checkpoints: `1024`, `2048`, `3072`, `4096`.

Checkpoint search-gate sweep:

- `1024`: `top1=62/232`, `candidate_better=46`,
  `reference_better=89`, `capped_sum_diff_cp=-5341`.
- `2048`: `top1=56/232`, `candidate_better=51`,
  `reference_better=95`, `capped_sum_diff_cp=-6724`.
- `3072`: `top1=64/232`, `candidate_better=53`,
  `reference_better=93`, `capped_sum_diff_cp=-5359`.
- `4096`: `top1=64/232`, `candidate_better=56`,
  `reference_better=91`, `capped_sum_diff_cp=-4663`.

Decision: no SPRT. This is not close to the reference, and every checkpoint
keeps the `-31814cp` mate-like tail. It is still materially better than prior
native Enyo-selfplay scratch checkpoints, so continue once with lower LR from
this Enyo-owned checkpoint before changing architecture or data again.


## Latest Result: Native Lichess Eval Bullet Smoke

`native-bullet-lichess-eval-smoke-46k-wdl0`

Purpose: validate direct `lichess_db_eval.jsonl.zst` to Bullet data without a
large intermediate JSONL/train pack, then run a tiny scratch native smoke.

Pipeline result:

- direct Bullet text import succeeded from the first `1M` input rows.
- Bullet format parsed and validated `46392` positions.
- result balance used for Bullet parser compatibility: `20712` wins,
  `5000` draws, `20680` losses.
- training completed in about `9s`; loss dropped from `0.070535` to
  `0.012504`.
- checkpoints: `8`, `16`, `24`, `32`.

Checkpoint search-gate sweep on v3 targets rejected every checkpoint:

- `8`: `top1=41/279`, `candidate_better=40`, `reference_better=177`,
  `capped_sum_diff_cp=-24554`, `worst_regression_cp=-32000`.
- `16`: `top1=48/279`, `candidate_better=44`, `reference_better=163`,
  `capped_sum_diff_cp=-22213`, `worst_regression_cp=-32000`.
- `24`: `top1=40/279`, `candidate_better=41`, `reference_better=170`,
  `capped_sum_diff_cp=-22976`, `worst_regression_cp=-32000`.
- `32`: `top1=34/279`, `candidate_better=39`, `reference_better=176`,
  `capped_sum_diff_cp=-23779`, `worst_regression_cp=-32000`.

Decision: no SPRT and no immediate scale-up of this recipe. The direct Lichess
eval data path is useful infrastructure, but the tiny eval-only smoke does not
show move-choice promise. Any future Lichess-eval use must be part of a larger
native data strategy with stronger gates, not a direct promotion lane.

## Do Not Do

Do not restart these as near-term Elo lanes:

- another same-architecture Stockfish-d16 Enyo self-play recipe.
- fresh d10/d12/mixed-depth self-play without new signal.
- Lichess blends as the main lever.
- bulk d18/d20 relabeling.
- head-only/output-only LR/objective sweeps.
- material/phase head masks.
- raw SF-binpack float-head update without tail scaling.
- SF-binpack float-head delta scaling.
- output-only signfit delta scaling: all tested scales failed the broad gate
  identically and the least-bad scale failed failure-suite by `-123658cp`.
- current king-bucket split variants.
- current king-pressure/check-state output bucket variants.
- target-only sparse multiplier sweeps.
- Bullet/Reckless-like scratch checkpoints as a direct replacement candidate.
- native scratch nets trained only on current Enyo self-play labels as promotion
  candidates.
- `native-bullet-sfbinpack-scratch-eval400-sb4096` as a promotion candidate:
  best checkpoint was still `top1=64/232`, `candidate_better=56`,
  `reference_better=91`, `capped_sum_diff_cp=-4663`.
- `native-bullet-sfbinpack-continue-eval400-lr2e4-sb4096` as a promotion
  candidate: fresh-hash 1M validation improved mate-like behavior, but non-mate
  stayed strongly negative (`candidate_better=27`, `reference_better=57`,
  `capped_sum_diff_cp=-2939`).
- scalar native SF-binpack continuation at deeper search: the `3M` non-mate
  diagnostic stayed negative (`candidate_better=21`, `reference_better=60`,
  `capped_sum_diff_cp=-3397`).
- `native-searchaware-nonmate-initdistill-w8-lr1e6-e6`: only L2 exported floats
  moved, internal target choice did not improve, and fresh-hash search gate
  collapsed (`capped_sum_diff_cp=-20372`).
- `search-aware-target-overfit-audit-20260523_063642` as a candidate recipe:
  it can move exported sparse/L1 tensors, but only reached `8/16` training-set
  top1 and caused very large broad init-net drift.
- simultaneous search-aware broad preservation as a candidate recipe:
  `search-aware-16target-preserve-audit-20260523_064326` kept broad drift small
  but did not cross the target-choice boundary.
- weak staged search-aware warmup as a candidate recipe:
  `native-searchaware-stagewarmup-audit16-lr3e6-e24` still moved only
  dense/output floats and ended at `1/16` top1.
- mixed-target strong staged search-aware warmup as a candidate recipe:
  `native-searchaware-stagewarmup-strong16-lr3e5-e96` moved sparse/L1 but
  targeted the wrong mixed 16-row subset and still ended at `1/16` top1.
- tag-filtered strong staged search-aware warmup as a candidate recipe:
  `native-searchaware-stagewarmup-strong16-lr3e5-e96` moved sparse/L1 heavily,
  but ended at `0/16` top1 on the intended non-mate subset.
- `native-searchaware-recover16-w002-lr1e6-e80` as a candidate recipe:
  broad behavior recovered, but target top1 stayed `0/16`; the input
  target-fit checkpoint was not export-persistent.
- `native-searchaware-targetonly-exportcheck16-lr3e5-e240` as a candidate
  recipe: it is a 16-target plumbing audit only. CPU exported validation passes,
  but the run has no broad-preservation evidence and is not a promotion path.
- `native-searchaware-cpurecover16-w0p001-lr1e6-e80`: CPU target retention
  passed, but broad static/search behavior collapsed (`top1=40/232`,
  `capped_sum_diff_cp=-19669`, `worst_regression_cp=-31924`).
- search-aware mateguard with broad init distillation and `mate_like=8`: it
  exported only dense/output changes and failed the search gate
  (`candidate_better=39`, `reference_better=146`,
  `capped_sum_diff_cp=-17929`).
- mask2 speed as the explanation for the bad smoke SPRT:
  `reckless-check-bucket-mask2-speed-audit-20260523_083628` measured the
  bucketed base at `search_ratio=0.999` and mask2 at `search_ratio=0.997`
  against reference. The earlier large slowdown was not reproducible, so
  runtime is not the reason mask2 fell over in match testing.
- `reckless-check-bucket-mask2-lr1e6-e4` as a promotion candidate:
  the current broad `300k` search-aware gate rejected it
  (`candidate_better=8`, `reference_better=17`,
  `capped_sum_diff_cp=-1242`, mate-like `worst_regression_cp=-31082`).
  The old composite gate was a false positive.
- `floathead_delta_020.nn` as a promotion candidate:
  the external Lichess-policy gate is slightly positive (`top1=508/800`,
  compare `69` vs `71`, capped `+400`) but match smoke rejected it:
  `-6.2 +/- 14.9 Elo`, `LLR=-0.86/2.94`, `LOS=20.6%`, `draw=52.0%`
  over `1000` games. Do not run 4k confirmation.
- shared mate-like tail diagnosis:
  `reckless-shared-tail-diagnosis-20260523_085539` showed the worst FEN is a
  horizon/target mismatch for Enyo search. Stockfish ranks `e5e6` clearly best
  and marks `b8d8`/`e4f3` as mate-losing, but Enyo forced searches at
  `100k`/`300k`/`1M` score the watched moves almost identically near `-2045cp`.
  Treat these mate-like oracle tails as diagnostic, not automatic SPRT
  blockers, unless repeated match evidence confirms them.
- output-signfit delta scaling as a near-term reckless lane:
  `reckless-floathead-delta-scale-300k-sweep-20260523_090331` rejected all
  scales on the current broad `300k` gate:
  `0.05 cap=-158`, `0.10 cap=-124`, `0.15 cap=-243`, `0.20 cap=-173`.
  The non-mate subset has small signal at some scales, but the broad score is
  not positive and this is not a SPRT candidate.
- native search-aware failure inspection:
  `native-searchaware-failure-inspection-20260523_091629` confirmed that the
  best native scratch/continuation checkpoints are not close to promotion.
  `continue2048` still loses `non_mate` badly (`candidate_better=31`,
  `reference_better=62`, `capped_sum_diff_cp=-4162`) and especially loses the
  `stable` source (`candidate_better=8`, `reference_better=58`,
  `capped_sum_diff_cp=-5696`). The problem is broad move choice, not only
  mate-like tails.
- native search-aware v2 targets:
  `native-searchaware-targets-v2-20260523` built a broader target set. The
  deduped file has `279` targets and `2022` move rows:
  `201 non_mate`, `78 mate_like`, with sources
  `pairwise_sprtfail:73`, `sprt_material:80`, `sprt_output:80`, `stable:46`.
  This is suitable for a preflight, but not proof that training will improve.
- `native-searchaware-v2-initdistill-w2-lr5e7-e8` as a native candidate:
  rejected. The internal target model gate improved to `42/201`, but exported
  movement was only `l2` (`214/25200209` parameters) and the broad `300k`
  gate lost to the native init checkpoint (`candidate_better=28`,
  `reference_better=36`, `capped_sum_diff_cp=-1605`; non-mate
  `candidate_better=25`, `reference_better=29`,
  `capped_sum_diff_cp=-1671`). Do not SPRT.
- target-builder issue:
  the v2 target builder truncated to top-8 moves before preserving known
  oracle/reference/candidate moves. That can misclassify rows as `non_mate`
  and then make `search_target_gate` assign synthetic `32000cp` gaps to
  previously scored moves. Fix target construction before another
  search-aware training run.
- native search-aware v3 targets:
  `native-searchaware-targets-v3-20260523` rebuilt the same source set after
  preserving marked moves. Deduped v3 is `279` targets and `2123` move rows:
  `200 non_mate`, `79 mate_like`, with `missing_marked=0` versus `101` missing
  marked moves in v2. The training/model-gate loader must preserve marked moves
  across its own top-K filter too; otherwise `search_max_moves=8` partially
  reintroduces the same blind spot.
- `native-searchaware-v3-moveaudit-cpu-lr1e6-in300-l1x300-e4` as a native
  candidate or audit continuation: rejected. The loader fix and cleaned v3
  targets did not improve target choice (`38/200` top1, `90/200` top3), and
  exported `input_weights`, `input_biases`, `l1_weights`, and `l1_biases` were
  still identical to the init checkpoint. Float input/L1 deltas existed but
  stayed far below export thresholds (`max_abs` about `0.04`, no values
  `>=0.1`, none `>=0.5`). Only small L2 float changes exported
  (`166/25200209` total values).

- `native-bullet-lichess-eval-smoke-46k-wdl0` as a candidate recipe:
  direct Lichess eval import and Bullet training work, but all checkpoints were
  far behind the reference on v3 search targets. Best checkpoint was `16` with
  `top1=48/279`, `candidate_better=44`, `reference_better=163`,
  `capped_sum_diff_cp=-22213`, `worst_regression_cp=-32000`.
- native Lichess policy objective audit:
  `native-policy-overfit64-lichess-cpu-lr3e5-e240` proved the current
  search-aware objective/export path can exactly learn a tiny policy slice:
  exported `model.nn` and `model.pt` both reached `64/64` top1 on 64 non-mate
  Lichess policy targets. This is an objective/plumbing result, not a playable
  net.
- native Lichess policy preservation audit:
  `native-policy-preserve64-lichess-w005-lr3e6-e80` kept broad init-net MAE
  controlled by the final epoch (`broad_mae=72.33`) and moved exported
  representation tensors (`input_weights changed=4041620/25165824`,
  `l1_weights changed=2278/32768`), but the final exported policy gate missed
  the required retention (`57/64`, required `60/64`). One lower-preservation
  retry is allowed before stopping this family.
- native Lichess policy preservation retry:
  `native-policy-preserve64-lichess-w002-select-lr3e6-e80` also rejected. The
  target-best checkpoint was epoch `45` with `61/64` top1, `63/64` top3,
  `sum_gap_cp=14`, `worst_gap_cp=12`; exported `model.nn` matched `model.pt`,
  but the predeclared gate was `62/64`. It moved exported representation
  tensors (`input_weights changed=4667621/25165824`,
  `l1_weights changed=2621/32768`) but still did not produce a broad-gateable
  candidate. Stop this target-only preservation family.

## Gates

A candidate must pass cheap gates before match testing:

1. `net-diff`: exported movement must match the intended change.
2. Static validation: no clear sign regression; MAE/sign are rejection filters,
   not proof.
3. Search/move gate: improve top1/top3 or capped gap against reference without
   losing mate-like or non-mate subsets.
4. Failure-suite replay: no unacceptable tail regression.
5. SPRT only after the above are clean.

A narrow positive gate does not justify SPRT if a broader gate or mate-like tail
is negative.

## Candidate Workflow

The next action should be one of these, in order:

1. Gate the unified search-aware corpus with the current reference engine and
   save `reference.csv` as the baseline for future model gates.
2. Implement or verify a mixed native objective path:
   CP/MPE-WDL plus policy/ranking from top-N child scores, with quantized
   forward/export checks.
   Current implementation status: search-aware training now forwards the broad
   batch through the same scalar `score_loss` helper used by normal training,
   so `objective=mpe25` and `wdl_lambda` are active for broad rows.
3. Add preflight gates before scale-up:
   10k-100k overfit, gradient reach, quantization-boundary scan after early
   batches, exported `net-diff`, small move-choice gate, and tactical-tail gate.
4. Only after those pass, run a larger Bullet/binpack native training job.
5. Keep Reckless paused unless a new existing-weight-compatible architecture
   hypothesis changes representation in a measurable way.
6. No SPRT from current native/search-aware/reckless output-delta runs.

Do not launch another training run until `build.json` names the lane,
hypothesis, data source, objective, architecture/export change, and gates.

## Workflow Rules

- Use `./build.py -c build.json` for candidate creation.
- Commit `build.json` with the experiment decision before running it.
- Keep run data under `runs/<run-name>/`.
- Use `nnue_reckless` for near-term existing-weight work.
- Use `nnue_native` for scratch/native work.
- Emit NNUE event notifications for long-running phases.
- Update this file only for durable conclusions or a changed next action.
- Put long evidence dumps in `docs/archive/`, not in this active plan.
