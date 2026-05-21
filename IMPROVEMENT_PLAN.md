# Enyo NNUE Improvement Plan

`README.md` defines the current candidate creation command. This file is the
changing plan for the next attempt to produce a stronger net.

Goal: add new signal, not rerun the same old pools or tiny LR/objective
variants.

## Current State, 2026-05-21

No trained Enyo net is currently a keeper.

Tested and rejected:

- deeper teacher labels on old/self-play pools: d16 and d18 improved static
  metrics but did not confirm in SPRT.
- fresh d10 and d12 self-play pools: generated new data, but candidate nets
  were neutral or negative after smoke/screen testing.
- Lichess blends: 15-20% Lichess sometimes improved static validation or early
  smoke direction, but did not hold up in longer SPRT.
- hardcase and failure-suite fine-tunes: moved specific positions, but produced
  too-large tail regressions and no gate-positive candidate.
- instability/disagreement blends from old pools: useful diagnostic signal, but
  not enough to justify SPRT promotion.
- mixed-depth self-play, d12/d8/d6 with Stockfish d16 labels: produced a novel
  pool with only `0.4%` exact overlap against the old d12 pool, but the best
  static candidate still finished smoke at `-0.7 +/- 15.0`, LLR `-0.33/2.94`.
- folded 8-king-bucket shortcut: clearly negative; not usable as a drop-in net.
- thread voting/arbitration search experiments: clearly negative in early SPRT;
  keep branches only for later redesign, not as active strength work.

Current conclusion:

- The current architecture plus Stockfish-labeled Enyo self-play is not
  producing Elo gains by changing depth, LR, objective, or simple source mix.
- Static MAE/sign improvements are now only rejection filters. They are not
  evidence of a stronger engine unless move-choice gates and SPRT agree.
- Do not start another same-pool or same-style matrix unless it tests a new
  hypothesis.

## Next Work

The next attempt must change one of the real bottlenecks:

1. Architecture/features.
   - Prepare one small, auditable NNUE feature or bucket change.
   - Verify feature extraction and export/load roundtrip before training.
   - Measure NPS before any long SPRT; reject changes that are too slow unless
     they show a large strength signal.
   - Do not use folded/drop-in shortcuts as evidence. Train the changed
     architecture properly.

2. Stronger or different teacher data.
   - Stop relying only on Enyo-vs-Enyo self-play labeled by Stockfish d16.
   - Evaluate prepared high-quality datasets or stronger teacher pipelines as a
     separate source, imported under `runs/`/`assets/`, not `~/tmp`.
   - If external formats are used, convert once into the normal Enyo row format
     and keep the source provenance in the run metadata.

3. Targeted move-choice data.
   - Expand the failure-suite and disagreement/PV-instability samplers.
   - Train at most one isolated candidate from this signal at a time.
   - Require pre-SPRT improvement in candidate/reference/oracle CSV gates:
     positive sum diff, no large new tail regression, and candidate-better count
     not worse than reference.

4. Tooling before more long runs.
   - Finish the `build.py -> tools/pipeline -> posgen/score/pack/train` path.
   - Keep new run data under `runs/<run-name>/`.
   - Keep legacy imported data under `runs/imported` and raw old tmp archives
     under `runs/legacy_*` until explicitly deleted.
   - Remove dependence on `~/tmp` for active workflows.

Decision rule for the next candidate:

- 1000-game smoke is only a direction filter.
- Extend only if it is clearly positive, ideally near `+10 Elo` or with a
  convincing positive LLR trend and clean move-choice gates.
- A `+3..+6 Elo` smoke with wide CI is not enough.
- Promote only after a longer screen confirms the signal.

## Archived Rationale For The Failed D12 Pivot

Static metrics have improved several times without producing a stronger engine
in SPRT. The latest d18 experiments tested whether deeper labels alone were the
missing signal. They were not:

- d18 premium/aggressive candidates were negative in 1000-game smoke SPRTs.
- d18 conservative `huber_cp1000_lr5e7_e4` looked promising at 1000 games
  (`+7.0 +/- 15.1`) but collapsed in the add-on run (`-1.7 +/- 9.8` at
  2302/3000).
- Fresh MPE smoke tests were also not useful:
  - self-play MPE: `+0.7 +/- 15.2`
  - fresh+15% Lichess MPE: `-3.5 +/- 14.9`

Later tests confirmed the same pattern:

- d10 self-play d16 matrix:
  - `fresh_huber_cp800_lr7e7_e8`: `+4.5 +/- 15.1` in smoke
  - `fresh_huber_cp1000_lr1e6_e6`: `-4.2 +/- 14.6`
  - `fresh_mpe25_cp1200_lr7e7_e6`: `-1.0 +/- 15.2`
  - `fresh85_lichess15_mpe25_cp1200_lr7e7_e6`: `+5.9 +/- 15.0` in smoke,
    then `-4.4 +/- 7.5` in 4000-game screen
- hardblend instability augments did not produce a keeper:
  - d8/d16 hardblend: `+3.5 +/- 15.1`
  - d6/d12 hardblend: `-1.7 +/- 15.4`
  - d8/d16 on d10 self-play: `-3.1 +/- 14.7`
  - d10/d18 on d10 self-play: `+1.7 +/- 14.7`
- hardcase-only d16 augmentation produced no hard-gate-positive candidate.
- Huber cp800 neighbor sweep produced one attractive smoke:
  `cp800_lr7e7_e8` at `+14.9 +/- 14.7`, LLR `1.19/2.94`; a follow-up run
  restarted from zero and quickly went negative. Treat that as a false positive,
  not a keeper.
- `best_init_cp800_lr2e7_e2` was stopped at 678/1000 after staying negative:
  `-3.6 +/- 18.3`, LLR `-0.41/2.94`.

Conclusion: do not keep spending time on the same data with deeper labels or
tiny LR/objective tweaks. The existing d8/d10/d16 pools are probably close to
tapped out for this architecture and starting net. Progress now requires either
a genuinely different self-play distribution or a feature/architecture change.

## Completed D12/Mixed-Depth Cycle

The previous active plan was to generate a new self-play distribution before
training again.

- Step 1: generate fresh Enyo-vs-Enyo self-play from the current reference at
  fixed depth `12`, not d8/d10. This should create positions from a stronger
  search distribution.
- Step 2: convert PGN to JSONL with the standard filters.
- Step 3: signed-bucket sample unique-FEN positions.
- Step 4: label sampled rows with Stockfish depth `16`.
- Step 5: pack tensors.

Main d12 run:

```text
runs/legacy_tmp_20260521/enyo_teacher/fresh_d12self18h64_d16_labels_20260519_113826
```

Operational note from that run:

- Use a small startup-hash wrapper for self-play:
  `runs/legacy_tmp_20260521/enyo_selfplay_cfg_20260519/enyo_hash64.sh`.
- Reason: Enyo allocates configured hash at startup before fastchess applies
  `option.Hash=...`; using the normal 1024 MB config with high self-play
  concurrency can OOM the host.

Expectation:

- This was a diagnostic run, not a guaranteed breakthrough.
- It tested whether stronger fixed-depth Enyo self-play produced useful new
  signal.
- Result: no keeper. Treat the d12/mixed-depth lane as exhausted unless a new
  architecture or teacher-data source changes the hypothesis.

## Latest Results

The fresh d12 distribution did not produce a keeper through the first static
recipes.

- Fresh d12 + 20% Lichess MPE looked mildly positive in smoke
  (`+8.0 +/- 15.2`, LLR `0.49/2.94`) but failed the 4000-game screen:
  `-0.3 +/- 7.6`, LLR `-1.16/2.94`.
- Failure-suite replay/oracle gate for that same candidate was only mildly
  positive in aggregate:
  - positions `913`
  - candidate better `70`
  - reference better `62`
  - sum diff `+889cp`
  - worst regression `-362cp`
  - conclusion: not a keeper because the tail regressions are too large.
- Failure-hardcase absolute-eval fine-tuning did not produce a gate-positive
  candidate:
  - `d12_l20_hardcase_w12_huber_cp1000_lr3e7_e4`: sum diff `-117cp`,
    worst regression `-511cp`
  - `d12_l20_hardcase_w8_huber_cp1000_lr7e7_e4`: sum diff `+1009cp`, but
    candidate better `62` vs reference better `65`, worst regression `-408cp`
  - conclusion: no SPRT.
- 8-king-bucket folded-net smoke was a valid negative drop-in test:
  the folded net lost every completed game before the run was stopped. This
  does not prove a properly trained 8-bucket net cannot work, but it does prove
  the folded Berserk-derived net is not usable as a shortcut.
- Hardcase pairwise sweep on the failure cases also failed:
  - it improved broad held-out MAE, but did not improve the actual hardcase
    move-choice margins.
  - best static-looking attempt `pair_w30_lr1e5_e6` had strong broad MAE
    improvement, but `delta_avg_margin=-5.0cp` and
    `delta_loss_weighted_margin=-5.5cp`.
  - conclusion: do not spend SPRT time on hardcase pairwise from this small
    43-case suite.

Current follow-up:

- Failure-suite gate the already-trained fresh d12 instability-blend candidate.
  It is a distinct signal from the failed d12/Lichess and hardcase fine-tunes.
  Only run SPRT if it beats the failed d12/Lichess candidate's gate quality, not
  merely if it is slightly positive.
- Result: rejected by failure-suite gate (`candidate_better=69`,
  `reference_better=65`, `sum_diff=-1415cp`, `worst_regression=-511cp`).
- Mixed-depth self-play lane, 2026-05-21:
  60% d12, 25% d8, 15% d6, then signed-bucket sample and relabel with
  Stockfish d16.
  - source run:
    `/home/petter/code/cpp/chess/nnue/runs/legacy_tmp_20260521/enyo_teacher/mixed_depth_selfplay_d16_20260520_165329`
  - rows: `3,183,999`, unique FEN `3,183,999`
  - exact overlap vs old d12 pool: `13,315` rows (`0.4%`)
  - best static candidate: `fresh_huber_cp1000_lr1e6_e6`
  - static deltas: fresh MAE `136.593 -> 130.625`, Lichess MAE
    `180.657 -> 174.996`
  - smoke SPRT: `-0.7 +/- 15.0`, LLR `-0.33/2.94`, LOS `46.4%`,
    draw `51.6%`
  - conclusion: reject. The distribution was novel, but still did not produce
    Elo. Do not spend more SPRT time on the other same-matrix candidates unless
    a separate gate shows a concrete reason.

Next active lane:

- Stop treating self-play distribution changes as the main lever by default.
- Run a sharper architecture/feature investigation and/or a genuinely stronger
  trainer-data path before another large SPRT matrix.
- Candidate data changes should now be justified by a pre-SPRT gate that
  improves actual move-choice/failure-suite behavior, not just broad MAE.

## Archived Parallel Work From The D12 Cycle

This section records what was prepared during the d12 cycle. It is historical,
not the active plan.

Rules:

- Explain the plan before non-trivial implementation.
- Ask before starting new long-running jobs.
- Do not add heavy pwa-5090 search jobs while it is saturated by self-play or
  labeling.
- Prefer preparation work that is useful even if the current d12 candidate
  fails.
- Do not launch 4-8 training variants on the same fresh data unless the first
  candidate shows enough signal to justify it.

Lane A: current d12 pipeline.

- Completed and rejected; see Latest Results.

Lane B: build the fixed failure-suite gate.

- Use historical bot logs and replay candidate/reference/oracle CSV output.
- Summarize candidate-better count, reference-better count, sum diff, median
  diff, worst regression, and best gain.
- Use this as a cheap pre-SPRT screen for every candidate.
- Current prepared tool:

```sh
tools/validate/replay_failure_suite.py
```

Lane C: prepare instability/disagreement sampling.

- Use `sample_search_instability.py` on old pools immediately, or on the new
  d12 JSONL once enough rows exist.
- Focus on rows where low-depth and high-depth search disagree by score or
  bestmove.
- This becomes one isolated candidate later; it should not be mixed with every
  other training knob.
- Pilot result, 2026-05-19:
  - run: `/home/petter/code/cpp/chess/nnue/runs/legacy_tmp_20260521/enyo_teacher/instability_pilot_20260519_142232`
  - source: old d12/d16 labeled pool
  - selected: 20k rows, 2 Stockfish workers, d8 vs d16
  - accepted before reservoir: 11,242 rows
  - written/packed: 10,000 rows
  - timeouts: 0
  - distribution: 55.6% endgame, 22.7% late, 18.6% middlegame, 3.0% opening
  - conclusion: tool works and produces a distinct hard-position slice; use a
    larger run later only when CPU is available or when fresh d12 JSONL exists.
- Larger sample result, 2026-05-19:
  - run: `/home/petter/code/cpp/chess/nnue/runs/legacy_tmp_20260521/enyo_teacher/instability_pilot_20260519_153044`
  - source: old d12/d16 labeled pool
  - selected: 100k rows, 4 Stockfish workers, d8 vs d16
  - written/packed: 50,000 rows
  - unique FEN: 50,000 / 50,000
  - distribution: 55.7% endgame, 22.8% late, 18.5% middlegame, 3.0% opening
  - side-to-move: exactly balanced at 50.0% white / 50.0% black
  - conclusion: useful candidate side dataset exists now, but it is old-pool
    instability signal. Use it as an isolated experiment, not as proof that
    the fresh d12 pool is useful.
- Weighted old-pool instability experiment, 2026-05-19:
  - run: `/home/petter/code/cpp/chess/nnue/runs/legacy_tmp_20260521/enyo_teacher/instability_weighted_train_20260519_171633`
  - mix: 1.95M old d12/d16 rows + 50k instability rows
  - source loss weight: `instability=5.0`
  - recipe: `huber`, clamp `1000`, lr `5e-7`, epochs `6`
  - baseline on held-out tail: MAE `175.887`, sign `90.36%`
  - candidate on held-out tail: MAE `173.398`, sign `90.31%`
  - failure-suite gate: 506 replay/oracle positions, candidate better `42`,
    reference better `37`, sum diff `+1384cp`, median nonzero diff `+4cp`,
    worst regression `-338cp`, best gain `+512cp`
  - conclusion: static MAE improved by `2.489cp`, but sign slightly regressed
    and the source is still the old pool. The failure-suite aggregate is
    positive but includes a worse tail regression than the old-control gate.
    Do not spend SPRT time on this as a keeper attempt; use it only as evidence
    that instability weighting can move static fit and should be retested on
    fresh d12 rows if the main data source looks novel.

Lane D: data audit / novelty report.

- Compare the fresh d12 pool against old pools:
  - unique FEN count
  - duplicate rate
  - side-to-move balance
  - phase/material buckets
  - eval bucket distribution
  - opening/book-exit diversity when available
  - exact-FEN overlap with old pools when practical
- Current prepared tool:

```sh
tools/validate/dataset_novelty_report.py
```

Lane E: architecture/feature branch preparation.

- Keep it isolated from data changes.
- Start with a small auditable feature/bucket change.
- Verify the feature fires correctly and measure NPS before committing to a
  full retune.
- Do not spend long training time on architecture until the branch is verified
  and has a clear test plan.

Do not parallelize:

- Full matrices of tiny LR/objective variants on the same data.
- Multiple SPRTs that compete for the same CPU budget unless the machine is
  otherwise idle and the candidates are already justified.
- Heavy Stockfish labeling on pwa-5090 while current self-play is using the
  host efficiently.

## Archived D12 Matrix Plan

Do not repeat this matrix unless a new architecture or teacher-data source
changes the hypothesis. The d18 and cp800-neighbor tests
showed that attractive 1000-game smokes can collapse, and weak data sources can
waste several candidate runs.

Before training, write a cheap novelty report for the new pool:

- unique FEN count
- side-to-move balance
- phase/material buckets
- eval bucket distribution
- duplicate rate against old pools when practical
- opening/book-exit diversity
- disagreement/instability counts if available

First-pass matrix:

- A: old best-known recipe rerun as a control, so training/script drift is
  visible.
  - active run, 2026-05-19:
    `/home/petter/code/cpp/chess/nnue/runs/legacy_tmp_20260521/enyo_teacher/old_control_d10_20260519_175317`
  - source: existing d10 self-play d16 packed data
    `/home/petter/code/cpp/chess/nnue/runs/legacy_tmp_20260521/enyo_teacher/fresh_d10self_d16_labels_20260517_234525/packed`
  - recipe: `huber`, clamp `800`, beta `200`, lr `7e-7`, epochs `8`
  - result: baseline MAE `138.983`, sign `91.44%`; candidate MAE `133.601`,
    sign `91.36%`
  - failure-suite gate: 506 replay/oracle positions, candidate better `35`,
    reference better `32`, sum diff `+1277cp`, median nonzero diff `+8cp`,
    worst regression `-287cp`, best gain `+545cp`
  - conclusion: static MAE improved by `5.382cp`, but sign regressed slightly.
    The failure-suite aggregate is positive but includes a large regression.
    This is useful as a control result, not a keeper signal; do not spend SPRT
    time while the main d12 self-play is active.
- B: one static fresh d12 candidate:
  self-play-only `huber`, clamp `800`, beta `200`, lr `7e-7`, epochs `8`.
- C: one disagreement/instability sampled fresh d12 candidate:
  `huber`, clamp `800` or `1000`.

Only run the broader matrix if the first-pass data source is not clearly bad:

- self-play-only `huber`, clamp `1000`, beta `200`, lr `1e-6`, epochs `6`
- self-play-only `mpe25`, clamp `1200`, lr `7e-7`, epochs `6`
- fresh+Lichess `mpe25`, clamp `1200`, lr `7e-7`, epochs `6`, with:
  - 80/20 fresh/Lichess
  - 70/30 fresh/Lichess

Do not make "new self-play distribution" one giant experiment. Split it into
falsifiable deltas: old-data control, static d12, Lichess mix, and
instability/disagreement sampling.

For future self-play generation, consider mixed generation instead of pure
deep fixed-depth self-play:

- 70% d12
- 20% d8/d10
- 10% noisy/perturbed generation
- Use SPRT as an early-stopping screen, not as a fixed 1000-game ritual:
  - prefer `elo0=0`, `elo1=5` for smoke screens
  - cap smoke at `1000-2000` games
  - kill clearly negative candidates early
  - only extend candidates with LLR moving convincingly upward
- Extend only if the smoke is clearly positive: ideally near `+10 Elo` or a
  convincing positive LLR trend. A small `+3..+6 Elo` smoke is not enough.
- Interpret `+10 Elo` at 1000 games correctly: it is not close to proof. It is
  only a direction filter that says the candidate is worth more games.
- If a smoke looks good, start the next run at the intended total game count;
  do not assume fastchess recovery will preserve the first 1000 games.

## Result Of The Neutral D12 Cycle

The d12 and mixed-depth cycles landed neutral/negative. Stop treating more
Stockfish-labeled Enyo self-play on this architecture as the main lever. The
next work should be architecture/features, stronger teacher data, and targeted
failure/disagreement data.

## Detailed Next Direction

Change the signal before changing tiny training knobs.

Immediate parallel work:

- Build a fixed Enyo failure suite now.
- Prepare one disagreement-sampled candidate in parallel with the conservative
  d12 matrix, instead of waiting for the d12 matrix to fail.
- Add per-candidate failure notes: where did it regress, e.g. middlegame,
  endgame, tactical swing, sign flip, or timeout-adjacent positions.
- Track why candidates fail, not only whether they fail: score variance,
  largest regressions, sign flips, and position type.

Generate a new, more diverse self-play pool:

- generate a larger fresh self-play pool from the latest reference
- vary openings more aggressively, but with a concrete target:
  - track unique book exits in the generated PGN
  - keep the book close to the SPRT validation book
  - use a slightly wider/reshuffled opening suite, not random garbage openings
- vary self-play depth/time
- add controlled randomness/noise during generation
- keep the opening suite close to the SPRT validation suite; use a slightly
  wider book, not random/unrealistic positions

Sample positions that matter to search decisions, not only static cp buckets:

- search disagreement positions
- PV instability positions
- large eval swings between depths
- Enyo vs Stockfish disagreement
- known Enyo tactical/positional failure cases
- endgames and conversion/defense cases
- positions near move-choice thresholds

Prepared tool:

```sh
tools/posgen/sample_search_instability.py
```

It compares a low-depth and high-depth teacher search and emits high-depth
labels for rows where score or bestmove changes materially.

Run one isolated architecture/feature investigation in parallel:

- keep it separate from data changes
- evaluate with the same fixed gates
- do not mix architecture and data changes in the same candidate
- start with a small, auditable feature/bucket branch before committing to a
  full retune
- measure NPS before full training; reject architecture changes that lose too
  much speed unless they produce a clearly larger Elo gain
- first candidate should be input bucket refinement or another easy-to-audit
  feature/bucket change, not a wide network change

Build a fixed Enyo failure suite:

- lost-game and timeout-adjacent positions
- replay/oracle positions with large regressions
- tactical/positional misses from real bot logs
- use as a non-SPRT regression screen before spending games
- record candidate failures against this suite, even informally, so a failed
  SPRT has an explanation beyond "Elo was negative"

Promotion rule:

- consider a new starting point once Enyo has a clearly positive own net
- only keep a candidate if SPRT is clearly positive, not merely MAE-positive

## Current Lessons

- Stronger labels improve static metrics, but old-pool d16 relabeling has not
  produced a clear replacement net.
- Small objective/LR changes on the same d12 pool mostly retest noise.
- 1000-game smoke SPRT is only a screen; attractive smokes have repeatedly
  failed to confirm.
- The old d16 expansion `lr1e-6` candidate ended around `+1.7 +/- 7.6 Elo`, so
  it is not a keeper.
- The d18 conservative candidate was not a keeper despite a promising smoke
  result; its add-on run went negative.
- The cp800 neighbor candidate was not a keeper despite a `+14.9 +/- 14.7`
  smoke.
- Hardcase-only and pairwise hardcase training moved specific positions but did
  not translate into match strength.
- Binpack-heavy training is not trusted unless isolated and proven by SPRT.
