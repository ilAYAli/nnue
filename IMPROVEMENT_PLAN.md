# Enyo NNUE Improvement Plan

This file is the active working plan. Long experiment notes are archived in
`docs/archive/IMPROVEMENT_HISTORY_20260523.md`.

Goal: produce a stronger Enyo net without repeating already-failed NNUE farming
loops.

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
`reckless-checkbucket-bucket-sweep`, rejected after 4k confirmation SPRT.
This was a reckless-lane diagnostic: keep existing weights, mix one trained
check-state output bucket at a time into the copied bucketed base, and run
broad gates before any SPRT.

Active native background run:
`native-bullet-sfbinpack-scratch-long-eval400-lr1e3-sb32768` is still running.
Checkpoint `12288` remains the best live gate so far by broad capped sum
(`+1142`), and checkpoint `24576` has the best top-1 score so far
(`80/232`). Every checked checkpoint through `26624` is still vetoed by the
mate-like tail (`-31kcp` class worst regression). No native checkpoint is an
SPRT candidate yet.

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




## Latest Diagnostic: Bullet Enyo Input Clamp Fix

Completed `reckless-enyo32-input-threshold-clampfix-lr2e7-sb8` after fixing
the Bullet Enyo optimizer clamp for trainable input tensors:

- root cause: the trainer clamped `l0w`/`l0b` to `+/-4095`, but Enyo stores
  accumulator weights and biases as full int16 values. The existing reference
  net legitimately contains values outside `+/-4095`.
- the previous Enyo32 input-only runs were therefore invalid: checkpoint `1`
  was mostly an export clamp artifact, not learned sparse movement.
- fix: trainable Enyo `l0w`/`l0b` now use full int16 optimizer bounds.
- verification: checkpoint `0` through `8` now have `input_weights=0`,
  `input_biases=0`, `l1=0`, and `l2=0` versus the reference.
- only four output weights differ by `<=1.49e-8`, already present at
  checkpoint `0`, which is Bullet float roundtrip noise and not training.

Decision: keep the clamp fix and discard the prior Enyo32 input-only gate
results that depended on clipped checkpoints. `lr=2e-7` is below exported sparse
movement after the fix; the next reckless input test must intentionally search
for the first real exported input movement, then gate that checkpoint before any
SPRT.

## Latest Result: Low-Pressure Enyo32 Input Divergence

Rejected `reckless-enyo32-existing-input-lr2e7-sb256`:

- this repeated the true-32 existing-weight input-only test at lower pressure
  (`lr=2e-7 -> 5e-8`, `256` superbatches).
- checkpoints `64`, `128`, and `256` all exported the same movement:
  `input_weights 505/25165824`, `input_biases 7/1024`; L1/L2/output unchanged.
- final checkpoint `256` is exported-tensor identical to the previous rejected
  `reckless-enyo32-existing-input-lr1e5-sb1024` checkpoint `256`
  (`total changed 0/25200209` when compared directly).

Decision: reject without search gate. Lowering LR did not create a smaller
quantized input step; it landed on the same exported net that already failed
`36` vs `140`, capped `-18181`, worst `-31837cp`.

## Latest Result: Existing-Weight Enyo32 Input Divergence

Rejected `reckless-enyo32-existing-input-lr1e5-sb1024`:

- this was a reckless-lane test, not native scratch. It initialized from the
  current existing Enyo net, expanded it to true 32 Bullet Enyo input buckets,
  and trained only input accumulator tensors.
- checkpoint `0` was exact export parity after 16->32 expansion:
  no changed input, L1, L2, or output tensors.
- checkpoint `256` moved only the intended exported tensors:
  `input_weights 505/25165824`, `input_biases 7/1024`; L1/L2/output unchanged.
- checkpoint `256` failed the 300k-node search gate badly:
  all `36` vs `140`, capped `-18181`, worst `-31837cp`;
  mate-like `26` vs `36`, capped `-3626`, worst `-31837cp`;
  non-mate `10` vs `104`, capped `-14555`, worst `-701cp`.

Decision: reject. True 32-bucket existing-weight input-only movement is possible
after the Bullet layout fix, but this LR/pressure immediately destroys broad
move-choice behavior. No SPRT and no later-checkpoint sweep for this run.

## Latest Result: Reckless Check-Bucket Bucket Sweep

Rejected `reckless-checkbucket-bucket-sweep-20260523_162724` /
`reckless-checkbucket-bucket6-stronggate-20260523_165548`:

- The bucket sweep mixed buckets `0` through `7` one at a time from
  `reckless-check-bucket-output-1m-lr1e6-e4` into the copied bucketed base.
- Bucket `6` was the best 300k-node broad-gate result:
  all `9` vs `3`, capped `+99`, median nonzero `+9.5cp`,
  worst regression `-47cp`; mate-like unchanged; non-mate `9` vs `3`,
  capped `+99`.
- Bucket `7` was also positive but weaker:
  all `3` vs `2`, capped `+47`, worst regression `-15cp`.
- Bucket `4` was weak-positive but less clean:
  all `14` vs `10`, capped `+82`, worst regression `-211cp`.
- Bucket `6` survived the stronger 1M-node broad gate:
  all `7` vs `2`, capped `+98`, median nonzero `+11cp`,
  worst regression `-34cp`; mate-like unchanged.
- Bucket `6` also survived failure-suite replay:
  `positions=913`, `candidate_better=10`, `reference_better=7`,
  `sum_diff_cp=+250`, `median_nonzero_diff_cp=12`,
  `worst_regression_cp=-27`, `best_gain_cp=92`.
- Despite those clean gates, match testing rejected it:
  - 1000-game smoke: `Elo -13.9 +/- 15.0`,
    `LLR -1.59/2.94 (-54%)`, `LOS 3.5%`, draw `51.4%`.
  - 1000-game repeat smoke: `Elo +4.9 +/- 15.1`,
    `LLR 0.20/2.94 (7%)`, `LOS 73.7%`, draw `51.2%`.
  - 4000-game confirmation:
    `Elo -4.5 +/- 7.7`, `LLR -1.47/2.94 (-50%)`, `LOS 12.4%`,
    draw `49.5%`.

Decision: reject. Bucket `6` is the cleanest pre-SPRT result from the
check-state-head family, but it did not produce match Elo. Do not launch
another check-state bucket/head SPRT from this family without a new hypothesis
that explains why broad search-target and failure-suite gates were positive
while match Elo was neutral/negative.

This also weakens the current promotion gate: passing broad search-targets and
failure-suite replay is necessary, but not sufficient. The gate needs either a
broader opening-distribution slice, direct smoke-SPRT mining, or a search metric
that better predicts match Elo.

## Previous Result

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

Next run:

- `native-bullet-sfbinpack-scratch-long-eval400-lr1e3-sb32768`.
- no `init_net`; this is an Enyo-owned native scratch run.
- same native architecture (`1024` hidden, `16` L2) so the test isolates data
  scale and scratch learning, not architecture.
- `32768` superbatches, checkpoint every `2048`.
- gate checkpoints before any SPRT; if it does not materially improve
  move-choice gates over the shorter native scratch runs, stop same-architecture
  native scratch and move to an architecture change.

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

## Hard Rejections

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
  the current broad `300k` search-aware gate did not confirm the old `100k`
  signal (`candidate_better=29`, `reference_better=33`,
  `capped_sum_diff_cp=-173`, mate-like `worst_regression_cp=-31084`).
  It has non-mate signal, but the mate-like tail is still a hard veto.
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

## Promotion Gates

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

## Next Decision

The next action should be one of these, in order:

1. Stop near-term reckless output-only work unless a new architecture hypothesis
   changes more than the final dense/output terms.
2. Stop sparse/input LR multiplier sweeps on the small failure-derived target
   set. It is too small and too brittle to move exported representation safely.
3. Stop the broad-target search-aware native config family after the teacher-
   preserved retry also failed static and broad search gates.
4. Continue the native baseline using larger Bullet/SF-binpack training and
   checkpoint sweeps.
5. Do not launch another tiny target-only child-eval recovery run.
6. No SPRT from the current search-aware native or reckless output-delta runs.

Do not launch another training run until `build.json` names the lane,
hypothesis, data source, and gates.

## Workflow Rules

- Use `./build.py -c build.json` for candidate creation.
- Commit `build.json` with the experiment decision before running it.
- Keep run data under `runs/<run-name>/`.
- Use `nnue_reckless` for near-term existing-weight work.
- Use `nnue_native` for scratch/native work.
- Emit NNUE event notifications for long-running phases.
- Update this file only for durable conclusions or a changed next action.
- Put long evidence dumps in `docs/archive/`, not in this active plan.
