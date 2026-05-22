# Enyo NNUE Improvement Plan

`README.md` documents how to create a candidate. This file records the current
strategy for producing a stronger net.

Goal: add new signal. Do not keep rerunning the same architecture on the same
kind of Stockfish-labeled Enyo self-play.

## Current State

No trained Enyo net is currently a keeper.

Current gate status:

- The 1M-node composite false-positive gate now combines the stable historical
  move-choice targets plus failed-SPRT mined targets from the material-head and
  output-signfit false positives: 213 target positions and 4,956 legal moves.
- This gate rejected every current candidate against the reference:
  - reference: `top1=91/213`, `top3=158/213`, `sum_gap_cp=98664`.
  - output-signfit lr5e-6: `top1=89/213`, `top3=156/213`,
    `sum_gap_cp=160216`.
  - material-head float 3M: `top1=83/213`, `top3=150/213`,
    `sum_gap_cp=136898`.
  - king-pressure float-head: `top1=90/213`, `top3=157/213`,
    `sum_gap_cp=160664`.
  - king-pressure head: `top1=94/213`, `top3=165/213`,
    `sum_gap_cp=192411`.
  - kingbucket v3: `top1=81/213`, `top3=149/213`,
    `sum_gap_cp=223046`.
  - scratch26 childx10: `top1=54/213`, `top3=128/213`,
    `sum_gap_cp=278856`.
- Conclusion: the older stable-only gate was too narrow and admitted known
  false positives. Do not run SPRT from that gate alone. A candidate must beat
  the reference on this composite gate, or explain why the composite target set
  is invalid, before it earns match time.
- Split-gate diagnosis shows why the remaining existing-weight deltas are still
  not promotable:
  - `output_signfit_lr5e6` mildly improved non-mate targets:
    `top1=65/164`, `top3=118/164`, `cap200=+285`, worst `-78cp`; but it lost
    the mate-like subset: `cap200=-243`, raw `-61837`, worst `-31138cp`.
  - `kingpressure_head` mildly improved non-mate targets:
    `top1=68/164`, `top3=124/164`, `cap200=+114`, worst `-82cp`; but it also
    lost mate-like raw sum badly: `cap200=-26`, raw `-93861`, worst `-31805cp`.
  - Tail diagnosis found `49` mate-like targets. Only `4` are `<=6` pieces and
    `45` are `>7` pieces, so this is not mostly a tablebase/endgame-policy
    artifact. Treat it as real tactical/search-eval tail risk.
  - Decision: no SPRT for output-signfit or king-pressure head. A candidate must
    win both non-mate and mate-like subsets, or there must be a concrete runtime
    guard/search policy that handles the mate-like tails before match testing.
- Existing-weight composite-pairwise repair has not produced a keeper:
  - `w005-lr5e7-e4`: static `mae=135.384`, `sign=92.18%`; composite gate
    `top1=89/213`, `top3=152/213`, `sum_gap_cp=67877`. Raw gap improved, but
    top1/top3 and capped deltas were not better than reference, so no SPRT.
  - `w02-lr1e6-e6`: static `mae=134.056`, `sign=92.17%`; composite gate
    `top1=88/213`, `top3=158/213`, `sum_gap_cp=266454`. Worse than reference.
  - Tiny overfit diagnostics showed the pair objective is wired and learnable
    only when broad scalar loss is removed or heavily reduced:
    pair-only/all-weights reached `95.1%` pair correctness, but broad MAE
    degraded to `356.93`.
  - A high-pressure blended run (`10k-w100-lr3e3-e100`) learned the pair set
    (`92.7%` pair correctness) but destroyed general behavior: static
    `mae=171.905`, `sign=75.42%`; composite gate `top1=53/213`,
    `top3=109/213`, `sum_gap_cp=395397`, with
    `candidate_better=27`, `reference_better=97`.
  - A low-pressure broad-data blend (`1m-w2-lr5e6-e12`) still failed:
    training pair correctness barely moved (`40.2%`), static validation
    regressed to `mae=167.166`, `sign=88.63%`, and composite gate was worse
    than reference: `top1=82/213`, `top3=150/213`,
    `sum_gap_cp=166918`, `candidate_better=33`, `reference_better=43`.
  - A reference-distilled pairwise repair
    (`pairdistill-1m-w10-lr1e6-e12`) preserved the initializer as the broad
    target instead of chasing Stockfish labels, but still failed. Pair
    correctness stayed stuck at `39.0%`, static validation regressed to
    `mae=180.837`, `sign=88.87%`, and the composite gate remained worse than
    reference: `top1=86/213`, `top3=152/213`,
    `candidate_better=34`, `reference_better=37`,
    `cap200_sum_delta_cp=-446`.
  - Decision: no SPRT. Pairwise can move the network only when it is too
    destructive, and reference distillation did not fix that. Close
    composite-pairwise repair as a near-term Elo lane; reopen only with a
    materially different objective or a less fragile target construction.
- Native/scratch composite gate status:
  - scratch quantized-Kaiming 26M: `top1=51/213`, `top3=114/213`,
    `sum_gap_cp=281771`.
  - Bullet-Enyo 28M eval400: `top1=37/213`, `top3=96/213`,
    `sum_gap_cp=366273`.
  - Bullet-Enyo 5M eval400: `top1=36/213`, `top3=105/213`,
    `sum_gap_cp=302408`.
  - Bullet-Enyo 5M eval800: `top1=40/213`, `top3=104/213`,
    `sum_gap_cp=369294`.
  - Clean native Bullet-Enyo scratch 5M eval400
    (`native-bullet-enyo-scratch-5m-eval400-sb4096`) used `--init-net ''`, so
    it did not inherit Berserk weights. It learned scalar labels better than
    earlier native runs, but still failed the promotion gates: static
    `mae=123.064`, `sign=81.50%`; composite gate `top1=47/213`,
    `top3=117/213`, `candidate_better=42`, `reference_better=99`,
    `cap200_sum_delta_cp=-9693`.
  - Full-data native Bullet-Enyo scratch eval400
    (`native-bullet-enyo-scratch-full-eval400-sb16384`) also used
    `--init-net ''` and trained from scratch on the imported d12/d16 labels.
    The final checkpoint failed static and move-choice gates: static
    `mae=503.945`, `sign=84.46%`, slope `2.56`; composite gate
    `top1=26/213`, `top3=89/213`, `candidate_better=30`,
    `reference_better=135`, `cap1000_sum_delta_cp=-38653`.
  - Checkpoint sweep for the same full-data run did not find a keeper:
    - `4096`: `top1=29/213`, `top3=85/213`, `candidate_better=28`,
      `reference_better=131`, `cap1000_sum_delta_cp=-35168`.
    - `8192`: `top1=30/213`, `top3=87/213`, `candidate_better=29`,
      `reference_better=129`, `cap1000_sum_delta_cp=-35193`.
    - `12288`: `top1=33/213`, `top3=82/213`, `candidate_better=30`,
      `reference_better=131`, `cap1000_sum_delta_cp=-35705`.
    - `16384`: `top1=26/213`, `top3=89/213`, `candidate_better=30`,
      `reference_better=135`, `cap1000_sum_delta_cp=-38653`.
    All checkpoints are far below reference on the composite move-choice gate;
    no SPRT.
  - Decision: no native SPRT. Native remains a long-term lane; the current
    scratch/Bullet-Enyo nets are not close enough on move choice.
- Embedded validation note: when `validate.py static` is used inside a custom
  gate against imported data, pass `--run <candidate-run>` or omit
  `--event-command`; otherwise the event hook reports the imported data run
  such as `fresh_d12self18h64_d16_labels_20260519_113826`.

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
- Bullet/Reckless-like 512-hidden scale and targeted-data runs:
  - The Enyo loader/search path is functional: `evalnet check` passes and
    direct eval speed is about `2.7M` eval/s for 512-hidden checkpoints.
  - 1M cp-only checkpoint `sb24` looked good statically
    (`broad5000 mae=71.0`, `corr=0.866`) and was the best early move-choice
    gate (`top1=5/13`, `top3=8/13`, `sum_gap=876`, `worst=301`), but SPRT
    rejected catastrophically at `127/1000`: about `-961 Elo`, `LLR=-2.95`.
  - Scaling cp-only to the full imported d12 pool did not improve the gate.
    Best checkpoint `sb8`: `top1=3/13`, `top3=5/13`, `sum_gap=945`,
    `worst=149`.
  - Existing WDL-weighted Bullet checkpoints also did not beat the cp-only
    gate. Best WDL checkpoint: 512-hidden `sb64`, `top1=5/13`, `top3=7/13`,
    `sum_gap=916`, `worst=291`.
  - Hardcase-child oversampling did not help. Best checkpoint `sb119`:
    `top1=4/13`, `top3=7/13`, `sum_gap=1085`, `worst=301`.
  - Direct move-choice child oversampling improved the narrow gate. Best
    checkpoint `sb112`: `top1=4/13`, `top3=8/13`, `sum_gap=651`,
    `worst=208`; static broad sample was still acceptable but worse than the
    1M cp-only checkpoint (`mae=82.3` vs `71.0`).
  - The move-choice checkpoint still failed SPRT hard at `199/1000`:
    `-211.2 +/- 45.8`, `LLR=-2.95`, `LOS=0.0%`.
  - Conclusion: the Bullet/Reckless-like path is technically integrated, but
    scalar cp/WDL training on the current labels does not produce a playable
    Enyo search eval. Do not spend more SPRT on Bullet checkpoints until a
    broader search-aware gate improves, not just the tiny repeated-tail gate.
- Bullet Enyo-format native training:
  - Bullet can now train and export Enyo `.nn` layout directly. This is useful
    because it gives fast iteration without changing Enyo's runtime evaluator.
  - The native/scratch lane is still not a keeper. The current 5M/28M
    layoutfix2 runs learned scalar labels but remained far below the current
    reference on sign/move-choice behavior.
  - Treat this as the `nnue_native` long-term lane only: useful for proving
    initialization, layout, and architecture ideas, not for immediate SPRT.
- Existing-weight Bullet/PyTorch head deltas:
  - Enyo `.nn` -> Bullet init -> Enyo `.nn` roundtrip is exact enough for this
    purpose: integer tensors are identical and float-head epsilon is tiny.
  - Bullet output-only training initially moved frozen sparse tensors because
    optimizer/clipping still touched frozen layers. That was fixed; the
    freeze-fixed run changed only `output_weights` and `output_bias`.
  - Output-only and float-head scalar Bullet fits improved MAE but reduced sign.
    Example output-only freeze-fix: candidate `mae=66.450`, `sign=91.38%`
    versus reference `mae=136.060`, `sign=92.20%`.
  - Sign-aware PyTorch head-only runs were safer but still not promotable.
    Float-head signfit: `mae=133.047`, `sign=92.15%`, failure-suite
    `candidate_better=71`, `reference_better=59`, `sum_diff_cp=+1116`,
    `worst_regression_cp=-408`.
  - Output-only signfit: `mae=132.432`, `sign=92.16%`, failure-suite
    `candidate_better=61`, `reference_better=66`, `sum_diff_cp=+344`,
    `worst_regression_cp=-294`.
  - Lower-pressure output-only signfit (`lr=5e-6`) changed only 30 output
    weights and improved MAE slightly (`135.145` vs reference `136.060`), but
    sign still lost to reference (`92.18%` vs `92.20%`). It was rejected before
    failure-suite/SPRT.
  - The material-head failed-SPRT mining pass made this candidate look uniquely
    interesting on the mined target set: `top1=28/80`, `top3=55/80`,
    `sum_gap_cp=2561`, `worst_gap_cp=329`, far better than the other checked
    candidates on that narrow diagnostic. Full replay failure-suite was only
    mildly positive: `candidate_better=68`, `reference_better=67`,
    `sum_diff_cp=+990`, `median_nonzero_diff_cp=+1`,
    `worst_regression_cp=-187`.
  - Corrected SPRT with `enyo/build/enyo` rejected it anyway:
    `+1.4 +/- 15.1`, `LLR=-0.13/2.94`, `LOS=57.2%`, `draw=51.0%`.
    It started around `+20 Elo` and collapsed to neutral by 1000 games, matching
    the earlier false-positive smoke pattern.
  - Mining this neutral SPRT produced another 80-target diagnostic:
    `candidate top1=19/80`, `top3=59/80`; reference `top1=27/80`,
    `top3=55/80`; `candidate_better=28`, `reference_better=32`.
    Raw `sum_diff_cp=+64933` was again dominated by mate-scale outliers.
    Capped sums were small and not convincing: `+268cp` at `100cp`,
    `+677cp` at `200cp`, with median nonzero diff `-1cp`.
    This explains the bad selection: the mined material-head target set was too
    narrow and rewarded a net that did not improve aggregate search strength.
  - Decision: no extension. Existing-weight head/output-only fitting is closed
    as a near-term Elo lane. It can be used only as a diagnostic source, not as
    another LR/objective/checkpoint sweep.
- Existing-weight material-bucketed output head:
  - Added an 8-bucket material/output-head net format as an existing-weight
    compatible `nnue_reckless` delta. The sparse input transformer and L1 remain
    unchanged; only the float/output head is replicated per material bucket.
  - Copied-head parity was clean: the expanded 8-bucket net had zero tensor diff
    against the copied source values and zero eval difference on the initial
    smoke FENs. This proves the format can be introduced as a no-op before
    training.
  - Output-only 1M-row fit changed only `232` output weights and failed the
    failure-suite gate: `candidate_better=44`, `reference_better=55`,
    `sum_diff_cp=-529`, `worst_regression_cp=-563`.
  - Float-head 1M-row fit changed `3136` head values and was closer, but still
    not promotable: `candidate_better=22`, `reference_better=25`,
    `sum_diff_cp=+592`, `median_nonzero_diff_cp=-3`,
    `worst_regression_cp=-123`.
  - Full-data/loss-selected float-head fit passed the failure-suite gate:
    `candidate_better=68`, `reference_better=59`, `sum_diff_cp=+1478`,
    `median_nonzero_diff_cp=+3`, `worst_regression_cp=-187`.
  - First SPRT attempt was invalid because it used the old
    `assets/engines/reference` binary against a changed-size bucketed net. That
    old binary effectively treated the bucketed net as a single-head net and
    lost `0-127`. `run_net_sprt_pwa.sh` now refuses changed-size nets with the
    default asset engine unless `--engine` is supplied.
  - Corrected SPRT with `enyo/build/enyo` was neutral-negative:
    `-3.5 +/- 15.1`, `LLR=-0.59/2.94`, `LOS=32.6%`, `draw=51.0%`.
  - A follow-up mask sweep kept trained material heads only for low/material
    buckets. Replay gates looked clean:
    - `mask0to5`: `candidate_better=63`, `reference_better=59`,
      `sum_diff_cp=+1643`, `worst_regression_cp=-187`.
    - `mask0to6`: `candidate_better=70`, `reference_better=59`,
      `sum_diff_cp=+1496`, `worst_regression_cp=-187`.
  - Corrected `mask0to6` SPRT still rejected the idea:
    `-7.0 +/- 14.8`, `LLR=-0.94/2.94`, `LOS=17.9%`, `draw=52.8%`.
  - Failed-SPRT mining produced 80 candidate-loss targets from the match:
    `assets/failure_suite/material_head_mask0to6_sprt_failure_targets_20260522.csv`
    and
    `assets/failure_suite/material_head_mask0to6_sprt_failure_scores_20260522.csv`.
    Raw cp sum was misleading because mate-scale outliers dominated it:
    `candidate_better=31`, `reference_better=42`,
    `sum_diff_cp=+61158`, `worst_regression_cp=-31006`,
    `best_gain_cp=+31082`. After capping deltas at `100cp` or `200cp`,
    the same gate is negative (`-388cp` / `-223cp`) with median `-1.5cp`.
    Future SPRT-failure gates must report capped diff, median, and
    candidate/reference counts, not only raw sum.
  - Decision: no extension. Stop material-head-only variants. The failure
    suite missed this regression, so mine the failed SPRT for new move-choice
    targets before launching another reckless candidate.
- Existing-weight king-bucket refinement:
  - `arch-kingbucket-v1` had the most interesting pre-SPRT signal so far:
    failure-suite `candidate_better=68`, `reference_better=54`,
    `sum_diff_cp=+1491`, but tail veto `worst_regression_cp=-448`.
  - `arch-kingbucket-v2-lr3e7-e8` lowered LR to reduce tail risk, but failed
    the static gate before failure-suite: candidate `sign=92.19%` versus
    reference `92.20%`.
  - Current follow-up: one export-aware/sign-aware king-bucket run
    (`arch-kingbucket-v3-quant-sign-lr7e7-e8`) targeted the observed
    sign/tail failure mode. It changed only 504 values, improved MAE
    (`135.375` vs reference `136.060`), but still lost sign (`92.18%` vs
    `92.20%`) and was rejected before failure-suite/SPRT.
  - Decision: stop this king-bucket split. Do not continue retuning it without
    a new implementation hypothesis.
- Enyo `.nn` pairwise SPRT-failure diagnostics:
  - Weak/current-initialized pairwise barely moved the exported net and failed
    the search gate.
  - Aggressive float/quantized runs improved pairwise training metrics but did
    not produce useful exported/search behavior.
  - Forced quantized run proved exported integer weights can move, but destroyed
    search behavior: `top1=4/60`, `top3=12/60`, `sum_gap_cp=40816`,
    `worst_gap_cp=31252`.
  - Mid-strength quantized run `pairwise-sprtfail-qmid-w50-lr3e3-e30` fixed the
    static margin gate after the Enyo `evalnet` validation path was corrected:
    candidate `43/49` best-preferring vs baseline `21/49`, fixed `24`,
    regressed `2`, worst static regression `-148cp`.
  - The same qmid net still failed the search move-choice gate badly:
    `top1=14/60`, `top3=27/60`, `sum_gap_cp=35866`,
    `worst_gap_cp=31251`, versus reference around `top1=29/60`,
    `top3=47/60`, `sum_gap_cp=953`, `worst_gap_cp=272`.
  - qmid2 targeted qmid's actual search-selected bad moves. It improved the
    search gate dramatically but still failed promotion gates: `top1=16/60`,
    `top3=27/60`, `sum_gap_cp=5259`, `worst_gap_cp=555`. It only moved the
    dense head relative to qmid and introduced a `-301cp` static tail.
  - qmid3 targeted qmid2's actual search-selected bad moves with lower weight,
    lower LR, and capped margins. The narrow search gate improved again:
    `top1=21/60`, `top3=34/60`, `sum_gap_cp=2715`,
    `worst_gap_cp=387`.
  - qmid3 failed broader gates badly. Static validation over 100k rows was weak:
    `mae=125.009`, `sign=73.09%`, `corr=0.700`. Repeated-tail gate regressed:
    `top1=1/13`, `top3=4/13`, `sum_gap_cp=33613`,
    `worst_gap_cp=31311`. Clean replay failure suite was clearly negative:
    `positions=913`, `candidate_better=101`, `reference_better=225`,
    `sum_diff_cp=-12207`, `worst_regression_cp=-881`.
- Broadtail-from-reference attempted that broader target set by combining
  qmid SPRT-failure legal-move scores with repeated-tail legal-move scores,
  starting again from the current reference net instead of a damaged qmid
  checkpoint. It moved only the dense head (`541/25200209` exported values),
  but failed broad validation: static `mae=114.921`, `sign=77.30%`,
  repeated-tail `top1=3/13`, `top3=6/13`, `sum_gap_cp=32941`, and clean
  replay failure suite `candidate_better=79`, `reference_better=332`,
  `sum_diff_cp=-57896`, `worst_regression_cp=-717`.
  - Conclusion: pairwise can create the intended local static/search preference,
    but the current pairwise target construction does not generalize. The qmid
    loop is suspended. Do not SPRT qmid/qmid2/qmid3/broadtail, and do not
    launch another pairwise run without a materially different target design
    and a clean broad gate.
- scratch quantized-Kaiming 26M-row Huber scale check:
  - Scalar/export behavior is the best scratch result so far on its own
    validation slice: exported `.nn` `mae=60.865`, `sign=88.35%`,
    `corr=0.902590`, `slope=0.811315`.
  - Same-slice Berserk-derived reference still wins sign heavily:
    `mae=136.093`, `sign=92.15%`, `corr=0.831691`, `slope=1.400200`.
  - Repeated-tail gate was slightly better than reference on a tiny set:
    scratch `top1=4/13`, `top3=6/13`, `sum_gap_cp=1121`,
    `worst_gap_cp=291`; reference `top1=3/13`, `top3=6/13`,
    `sum_gap_cp=32393`, `worst_gap_cp=31311`.
  - Full replay failure-suite gate was clearly bad versus current reference:
    `positions=913`, `candidate_better=89`, `reference_better=328`,
    `sum_diff_cp=-37297`, `worst_regression_cp=-1008`.
  - Decision: no SPRT. Scratch scalar fit is real, but move-choice/search
    behavior is not close enough to replace the reference.
- broader repeated search-failure target set:
  - Extracted 59 positions that repeated across qmid3, broadtail, and scratch26
    failures:
    `assets/failure_suite/search_failure_targets_20260522.csv`.
  - Scored all legal moves with Stockfish: 1806 move rows in
    `assets/failure_suite/search_failure_move_scores_20260522.csv`.
  - Current reference baseline on this set: `top1=42/59`, `top3=55/59`,
    `sum_gap_cp=595`, `worst_gap_cp=159`.
  - scratch26 baseline on this set: `top1=7/59`, `top3=23/59`,
    `sum_gap_cp=44640`, `worst_gap_cp=31887`.
  - Decision: this is a useful gate. It rejects scratch26 cleanly and is now
    the first gate for any scratch-repair attempt.
- first scratch26 search-failure pairwise repair:
  - `pairwise-searchfail-scratch26-w6-lr1e3-e16` targeted the aggregated
    candidate move from the source CSV. That was wrong for this purpose: it
    often used qmid3/broadtail's selected bad move rather than scratch26's
    actual selected move.
  - It still improved the broad search-failure gate over scratch26:
    `top1=10/59`, `top3=24/59`, `sum_gap_cp=12061`,
    `worst_gap_cp=699`.
  - It damaged broad scalar fit and repeated-tail behavior:
    static `mae=88.760`, `sign=85.63%`; repeated-tail `top1=4/13`,
    `top3=5/13`, `sum_gap_cp=1447`, `worst_gap_cp=291`.
  - Decision: no replay gate and no SPRT. The next diagnostic must pass
    `pairwise_candidate_moves_csv` with scratch26's actual search moves.
- scratch26 override repair:
  - `pairwise-searchfail-scratch26-override-w4-lr5e4-e12` preserved broad
    static fit better than the first repair: static `mae=67.972`,
    `sign=88.16%`.
  - It did not fix the actual search-failure gate: `top1=9/59`,
    `top3=23/59`, `sum_gap_cp=75150`, `worst_gap_cp=31887`.
  - Reason: one selected bad move per target is too narrow; the aggregate run
    fixed the catastrophic `d3e4 -> d3g6` case only by shifting the broader
    local decision boundary.
  - Decision: no replay gate and no SPRT. Use explicit multi-pair rows instead.
- scratch26 multi-pair repair:
  - `pairwise-searchfail-scratch26-multipair-w4-lr5e4-e12` trained on 108
    explicit bad-vs-best pairs over 57 search-failure targets.
  - It still only moved the dense head in the exported net:
    `497/25200209` exported values changed; input, input bias, L1 weights, and
    L1 bias were unchanged.
  - Broad static was acceptable for a scratch repair diagnostic:
    `mae=65.955`, `sign=87.86%`, `corr=0.900056`, `slope=0.697924`.
  - Move-choice gates were still bad: search-failure `top1=11/59`,
    `top3=24/59`, `sum_gap_cp=74341`, `worst_gap_cp=31814`;
    repeated-tail `top1=2/13`, `top3=4/13`, `sum_gap_cp=2265`,
    `worst_gap_cp=647`.
  - Decision: no replay gate and no SPRT. Pairwise margins are not moving the
    sparse/input side of the scratch net enough. Next diagnostic should train
    the same child positions as normal scalar labeled rows.
- scratch26 scalar child-row blend:
  - `scalar-searchfail-scratch26-childx100-lr5e4-e8` blended 100k broad rows
    with 100 repeats of the 216 search-failure child rows.
  - It still only moved the dense head in the exported net:
    `497/25200209` exported values changed; input, input bias, L1 weights, and
    L1 bias were unchanged.
  - It improved the broad search-failure gate over scratch26:
    `top1=12/59`, `top3=22/59`, `sum_gap_cp=11305`,
    `worst_gap_cp=699`.
  - It damaged broad scalar fit and repeated-tail behavior:
    static `mae=91.171`, `sign=82.24%`, `corr=0.750914`,
    `slope=0.593222`; repeated-tail `top1=4/13`, `top3=5/13`,
    `sum_gap_cp=32467`, `worst_gap_cp=31311`.
  - Decision: no replay gate and no SPRT. The signal is real but too
    destructive. Try one lower-pressure scalar blend before stopping this lane.
- scratch26 conservative scalar child-row blend:
  - `scalar-searchfail-scratch26-childx25-lr2e4-e8` reduced the child repeat
    count to 25 and lr to `0.0002`.
  - It still only moved the dense head in the exported net:
    `497/25200209` exported values changed; input, input bias, L1 weights, and
    L1 bias were unchanged.
  - It improved search-failure over scratch26 and avoided the catastrophic
    repeated-tail collapse from `childx100`: search-failure `top1=13/59`,
    `top3=23/59`, `sum_gap_cp=11014`, `worst_gap_cp=699`;
    repeated-tail `top1=4/13`, `top3=6/13`, `sum_gap_cp=1503`,
    `worst_gap_cp=294`.
  - Broad scalar fit was still worse than scratch26:
    `mae=69.222`, `sign=86.67%`, `corr=0.852982`, `slope=0.738942`.
  - Decision: no replay gate and no SPRT. Try one minimal child-row blend; if
    it does not preserve broad fit, stop this scalar child-blend lane.
- scratch26 minimal scalar child-row blend:
  - `scalar-searchfail-scratch26-childx10-lr1e4-e8` reduced the child repeat
    count to 10 and lr to `0.0001`.
  - It preserved broad fit better than the higher-pressure variants:
    static `mae=63.989`, `sign=87.66%`, `corr=0.882222`,
    `slope=0.783920`.
  - It was the best search-failure scalar blend so far:
    search-failure `top1=13/59`, `top3=25/59`, `sum_gap_cp=10484`,
    `worst_gap_cp=699`; repeated-tail `top1=4/13`, `top3=6/13`,
    `sum_gap_cp=1084`, `worst_gap_cp=291`.
  - Full replay failure-suite versus current reference still rejected it hard:
    `positions=913`, `candidate_better=78`, `reference_better=461`,
    `sum_diff_cp=-95130`, `median_nonzero_diff_cp=-95`,
    `worst_regression_cp=-977`, `best_gain_cp=465`.
  - Decision: no SPRT. Stop scalar child-blend repairs from scratch26. The
    search-failure signal is real but still produces a much worse engine than
    the current reference.
- current-reference sparse-LR movement probe:
  - `pairwise-ref-sparseprobe-w2-lr5e6-in100-l120-d02-e8` used per-layer LR
    multipliers to try to move exported sparse/input tensors:
    base lr `5e-6`, input multiplier `100`, L1 multiplier `20`, dense
    multiplier `0.2`.
  - Gradients were nonzero for input and L1, but exported sparse tensors still
    did not cross quantization thresholds:
    `input_weights changed=0/25165824`, `input_biases changed=0/1024`,
    `l1_weights changed=0/32768`, `l1_biases changed=0/16`.
  - Only dense/head floats moved, and only very slightly:
    total `508/25200209`.
  - Repeated-tail top counts improved (`top1=5/13`, `top3=8/13`), but the tail
    remained catastrophic: `sum_gap_cp=32243`, `worst_gap_cp=31311`.
  - Decision: no replay gate and no SPRT. Current-reference quantized
    fine-tunes are dense/head-only at this scale. Do not keep increasing LR
    unless the objective explicitly targets sparse/exported movement.
- current-reference sparse export-threshold diagnosis:
  - Float `.pt` deltas from the sparse-LR probe confirmed why export stayed
    unchanged: max input delta was `0.058586`, max input-bias delta `0.052368`,
    max L1 delta `0.010529`, and max L1-bias delta `0.012695`.
  - None approached the roughly half-integer threshold needed to alter rounded
    int16/int8 exported tensors.
  - `pairwise-ref-sparsecross-w2-lr5e6-in1200-l11200-d0-e12` proved exported
    sparse movement is achievable when dense/head is frozen and sparse LR is
    pushed hard: input weights changed `233232/25165824`, input biases
    `740/1024`, L1 weights `25744/32768`, L1 biases `10/16`, with dense/head
    unchanged.
  - The same candidate failed broad replay badly versus the current reference:
    `positions=913`, `candidate_better=71`, `reference_better=495`,
    `sum_diff_cp=-114245`, `median_nonzero_diff_cp=-151.5`,
    `worst_regression_cp=-923`, `best_gain_cp=465`.
  - Decision: no SPRT. Crossing sparse export thresholds is possible, but the
    repeated-tail pairwise objective is not usable at this pressure.
- current-reference broad sparse refresh probe:
  - `broad-ref-sparse-huber-cp800-lr5e6-in800-l1800-d0-e8` used broad
    Stockfish-d16 labels instead of repeated-tail pairs, with dense/head LR
    frozen and input/L1 multipliers set to `800`.
  - Training gradients were nonzero for input and L1, but the exported `.nn`
    was identical to the initializer:
    `input_weights changed=0/25165824`, `input_biases changed=0/1024`,
    `l1_weights changed=0/32768`, `l1_biases changed=0/16`,
    total `0/25200209`.
  - Train MAE stayed flat at `135.95` across all 8 epochs.
  - Decision: no replay gate and no SPRT. The current-reference sparse
    fine-tune lane is exhausted: broad scalar labels do not cross export
    thresholds, while forced sparse movement damages broad behavior.
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
- scratch Kaiming 100k-row quantized-forward preflight:
  - Float/export matched, but the run was a no-op: both MAE `143.151` and sign
    `50.05%`.
  - Input/L1 gradient norms stayed zero because rounded Kaiming starts with
    zero exported input/L1 tensors.
  - Decision: use a quantization-compatible Kaiming initializer with integer
    scale for input/L1 and Kaiming scale for the dense float head.
- scratch quantized-Kaiming 100k-row quantized-forward preflight:
  - Training moved clearly: validation sign reached `70.84%` at epoch 7.
  - Export parity was good on the 20k validation slice: float `.pt` MAE
    `138.672`, sign `70.58%`; exported `.nn` MAE `138.027`, sign `70.84%`.
  - Decision: this is the first scratch path that both learns and survives
    export. Scale the same method to 1M train rows before changing anything
    else.
- scratch quantized-Kaiming 1M-row Huber scale check:
  - Training selected epoch 11 by validation sign, then early-stopped at epoch
    15.
  - Export parity was good on the 200k validation slice: float `.pt` MAE
    `109.630`, sign `80.26%`; exported `.nn` MAE `108.474`, sign `80.83%`.
  - Same-slice Berserk-derived reference: MAE `135.862`, sign `92.21%`.
  - Decision: scratch now learns and exports correctly, but Huber is producing
    too many sign/ranking errors. Test MPE25 before scaling scratch further.
- scratch quantized-Kaiming 100k-row MPE25 preflight:
  - Export parity was good: exported `.nn` MAE `136.012`, sign `69.82%`,
    correlation `0.354`.
  - The same-size Huber preflight was better on sign/correlation: exported
    `.nn` MAE `138.027`, sign `70.84%`, correlation `0.405`.
  - Decision: do not scale MPE25. Add a small explicit sign auxiliary loss to
    the Huber recipe instead.
- scratch quantized-Kaiming 100k-row Huber plus sign-loss preflight:
  - Peak validation sign was `70.75%`, not better than plain Huber's `70.84%`.
  - MAE also degraded late in training.
  - Decision: do not scale the current-architecture scratch sign-loss lane.
    Return to the Bullet/Reckless-like architecture lane.

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
   - Enyo commit `8fda8d3` fixes `evalnet` for loaded Enyo `.nn` files to use
     the same scaled `Evaluate2` route as search. Older `eval_move_gate` results
     from before this fix understated candidate differences.

## Next Concrete Experiment

Immediate next branch:

- Head-only existing-weight fitting has been tested and rejected. Do not run
  more LR/objective/checkpoint sweeps in this lane.
- Material-head-only fitting has also been tested and rejected. Do not run
  more material-head masks, LR/objective retunes, or checkpoint sweeps.
- Immediate work: inspect a genuinely different `nnue_reckless` structural
  hypothesis before launching more training. The latest diagnostics say
  head/output, material-output buckets, and king-pressure heads can pass narrow
  or non-mate gates while staying neutral or negative overall.
- Do not spend more training or SPRT on any candidate that wins only non-mate
  targets while losing mate-like tactical tails. The latest tail diagnosis found
  that those tails are mostly `>7` pieces, so they are not a simple Syzygy
  fallback issue.
- The next `nnue_reckless` candidate must be a different
  existing-weight-compatible structural idea. Head-only fitting, material-head
  fitting, king-pressure head buckets, and the current king-bucket split are all
  rejected.
- The first candidate to inspect is a Reckless-inspired threat/attack feature
  side branch:
  - Preserve existing Enyo weights exactly at initialization.
  - Add any new threat/attack weights as zero-initialized tensors, so copied
    nets are eval-identical before training.
  - Keep the sparse piece/king accumulator and current search evaluator as the
    primary path; do not replace this lane with a scratch Reckless net.
  - Prove copied-net parity, export roundtrip, and NPS before training.
  - If the feature is disabled at compile/runtime, it must have zero hot-path
    overhead.
  - `threat-zero-parity-20260522_171752` proved copied-net tensor/eval parity:
    all existing tensors unchanged, appended threat weights all zero, and four
    checked FENs evaluated identically.
  - The first active threat implementation failed the NPS gate even with zero
    threat weights: startpos `go nodes 300000` dropped from roughly
    `0.94M-1.63M nps` to `0.28M-0.53M nps`. Do not train or SPRT this
    from-scratch threat recomputation path.
  - `threat-kingzone-fastcheck-20260522_173322` reduced the active feature set
    to occupied targets in the king zone. Copied zero nets still had parity and
    no meaningful cost, but a single active threat row still dropped depth-19
    NPS from roughly `1.43M` to `0.88M`.
  - `threat-rowmask-fastcheck-20260522_173518` added an active-row mask so
    zero-weight rows do not perform 1024-wide accumulator additions. Copied
    zero nets remained fast, but one active row still dropped depth-19 NPS from
    roughly `1.47M` to `1.22M`, which fails the 3-5% NPS gate.
  - Decision: reject the current appended threat-accumulator design. Do not
    train or SPRT it. Threat work is only worth revisiting as a much smaller
    scalar/head feature or a genuinely incremental cached accumulator.
- Current `nnue_reckless` probe: king-pressure output bucket.
  - This keeps the existing sparse input transformer and L1 weights unchanged.
    Only the float head is replicated into the existing 8 output buckets.
  - Bucket selection is based on opponent attack pressure into the side-to-move
    king zone, not total material count.
  - `kingpressure-bucket-fastcheck-20260522_173949` passed copied-net parity:
    base and bucketed-copy nets produced identical checked evals.
  - NPS cost was acceptable for a head-bucket selector: depth-19 startpos
    dropped from roughly `1.51M` to `1.48M` NPS, about `2.1%`, under the
    `3-5%` gate.
  - Active run: `reckless-kingpressure-head-1m-lr1e6-e8` via `build.py`,
    backend `material-head`, bucket mode `king-pressure`, existing Berserk/Enyo
    weights, output-only, 1M rows. This is a cheap probe; no SPRT unless static
    and failure-suite gates are clean.
  - Result: output-only probe is not SPRT-worthy. Static was slightly better
    on MAE and equal on sign (`135.684`, `92.17%`) versus reference
    (`135.868`, `92.17%`), but the failure-suite was only marginal:
    `candidate_better=53`, `reference_better=52`, `sum_diff_cp=+702`,
    `median_nonzero_diff_cp=+1`, `worst_regression_cp=-253`.
  - Decision: reject for SPRT. The structure is close enough for one
    float-head probe, but do not keep sweeping it if the next replay gate is
    also marginal or tail-negative.
  - Float-head follow-up `reckless-kingpressure-floathead-1m-lr2e7-e8` is also
    rejected. Static was again only marginally better than reference
    (`mae=135.807`, `sign=92.17%`, `wrong_sign=7203` versus reference
    `mae=135.868`, `sign=92.17%`, `wrong_sign=7204`). Failure-suite was worse
    on move choice: `candidate_better=34`, `reference_better=46`,
    `sum_diff_cp=+731`, `median_nonzero_diff_cp=-6.5`,
    `worst_regression_cp=-138`.
  - Decision: no SPRT. Close this king-pressure head-bucket lane unless there
    is a new bucket signal that targets the mate-like tails directly.
- Current `nnue_reckless` probe: check-state output bucket.
  - Branches:
    - Enyo: `feature/nnue-reckless-check-bucket`,
      commit `7582378`.
    - NNUE: `feature/nnue-reckless-check-bucket` with the committed
      `build.json` recipe.
  - This is a small existing-weight delta. It keeps the sparse input
    transformer and L1 unchanged, accepts copied single-head nets as a no-op,
    and only trains the replicated float output layer.
  - Bucket selection is side-to-move tactical state:
    no pressure, direct check by non-slider, direct check by slider, double
    check, then king-zone pressure bands.
  - `build.json` is the source of truth:
    `reckless-check-bucket-output-1m-lr5e6-e8`, backend `material-head`,
    bucket mode `check-state`, output-only, 1M train rows plus 100k validation.
  - Preflight already passed before launch:
    normal build with `ENYO_ENABLE_CHECK_BUCKET_NNUE=OFF`, experimental build
    with it `ON`, both test binaries, copied-head parity on startpos,
    evalnet bench, pack smoke, and 1-epoch train smoke.
  - No SPRT until broad static, composite move-choice, mate-like, and mined
    SPRT-failure gates are clean.
- Gate requirement before SPRT:
  - `candidate_better >= reference_better`.
  - capped `sum_diff_cp > 0` at a documented cap such as `200cp`, not only
    raw mate-scale sum.
  - median nonzero diff should not be negative.
  - `worst_regression_cp > -250`.
  - static sign must not be worse than the reference on the same slice.
- Do not keep retuning the same bucket split.

Background lane:

- Keep `nnue_native` separate: scratch/native Enyo-owned nets can run when
  resources allow, but they do not replace the near-term reckless lane and must
  not use Berserk weights.

Reason:

- scratch26 proves an Enyo-owned net can learn scalar labels and survive export,
  but it fails hard on repeated search-choice mistakes.
- Pairwise diagnostics did not produce enough useful exported movement and did
  not fix the broad search-failure gate.
- The first scalar child-row blend improved search failures but destroyed broad
  static fit and repeated-tail behavior.
- The conservative scalar child-row blend reduced the damage, but still
  regressed broad static fit.
- The minimal scalar child-row blend was the best repair, but the replay
  failure-suite remained overwhelmingly worse than the current reference.
- All current-reference and scratch-repair fine-tunes that start near an
  existing exported `.nn` have only changed dense/head tensors after export.
  Input/L1 exported tensors stayed fixed. That makes another ordinary LR sweep
  low value.
- Sparse-LR multipliers did not change that: gradients reached input/L1, but
  exported tensors stayed identical.
- The measured float deltas were far below export rounding thresholds. The
  sparse-cross diagnostic proved that exported sparse movement can be forced,
  but the resulting broad behavior was unusable.
- The broad sparse refresh answered the final sparse-lane question: broad
  scalar labels did not move exported input/L1 tensors at all.
- Bullet Enyo-format training now works technically, but scalar/head-only fits
  still show the same pattern: better MAE does not imply safer search decisions.

Anti-confounding rule:

- Do not change architecture and data source in the same first candidate.
- Reuse the best-understood training source for the first architecture test:
  the current signed-balanced d12 self-play plus Stockfish-d16 labels.
- Use `build.py --labeled-jsonl` so the first architecture test repacks the
  existing labels with the new feature map instead of generating fresh
  self-play or relabeling.
- Because this moves the recipe through the new `build.py` pack/train path,
  run the pack/static/roundtrip sanity checks before training starts.

Gate action:

- Use `search_failure_move_scores_20260522.csv` as the first scratch-repair
  move-choice gate.
- Current reference is strong on this set; scratch26 is not. A scratch-repair
  candidate must improve sharply over scratch26 before any replay gate.
- Run `tools/validate/move_choice_gate.py` before SPRT.
- Add no-op export checks (`validate.py net-diff`) before static/replay gates.

## Candidate Workflow

Normal candidate creation:

```sh
./build.py -c build.json
```

Current `build.json` intent:

- candidate name:
  `native-bullet-enyo-scratch-full-eval400-sb16384`.
- selected branch: `feature/nnue-native-bullet-scratch-full`.
- selected lane: `nnue_native`, a clean Enyo-owned scratch net with no Berserk
  initialization.
- backend: `bullet`, `bullet_mode=enyo`, exporting a normal Enyo `model.nn`.
- initializer: empty `init_net`, so this is scratch/native rather than an
  existing-weight delta.
- broad source:
  `runs/imported/fresh_d12self18h64_d16_labels_20260519_113826/score/labeled.jsonl`.
- schedule: `3M` rows, hidden `1024`, L2 `16`, batch `4096`, batches `64`,
  superbatches `16384`, lr `0.001 -> 0.00005`, eval scale `400`.
- result: rejected by static/composite gates and checkpoint sweep. No SPRT.
- next `build.json` change must name a new branch/lane and should be committed
  before launch.

Rules:

- Treat `build.py` plus committed `build.json` as the public workflow. Files
  under `tools/` are implementation details unless the root `README.md`
  explicitly names them.
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
