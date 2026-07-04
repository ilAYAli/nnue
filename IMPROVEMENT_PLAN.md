# NNUE Architecture Improvement Plan

## Current status

Current: **Stage 3 - architecture screening**

- Stage 2 completed all eight short-training runs in `3h 26m 47s`.
- All eight nets passed runtime loading and static validation.
- Static ranges: MAE `302.287-314.086`, sign `81.26-82.25%`,
  correlation `0.704340-0.731247`, and slope `0.276686-0.296967`.
- Results: `runs/logs/stage2-short-training.tsv`.
- First screen: `enyo-16x12-1024-o4` against the
  `enyo-16x12-1024-o8` control over 1,000 fixed games.

Completed stages:

- Stage 0 - plan and protocol, 2026-07-04.
- Stage 1 - architecture support, 2026-07-04.
  - All eight planned layouts passed trainer-to-engine parity.
  - Enyo, Rust/CUDA, and Python support suites passed.
  - Checkpoint resume restored weights, Adam state, data position, and LR
    position. Its maximum final-weight difference was `1.91e-6`, identical to
    the variation between two uninterrupted CUDA runs.
- Stage 2 - short training, 2026-07-04.
  - Eight of eight planned candidates completed and passed the exit gate.

Queued stages and elapsed-time estimates:

1. Stage 3 - architecture screening: 4-5 hours.
2. Stage 4 - full winner training: 1-2.5 hours, or up to 4.5 hours for two finalists.
3. Stage 5 - full validation: 45-60 minutes, plus 20-30 minutes if Berserk qualifies.
4. Stage 6 - incremental improvement: 30-45 minutes per iteration.

Update this section when a stage starts and completes. Record the actual result
and calculate the next stage's expected finish from its real start time.

## Stage 1: architecture support

Support and verify the practical Enyo-native matrix before starting comparison
training:

- Hidden widths: 512, 768, and 1024.
- Enyo-native input bucket layouts: 10, 16, and 32.
- Feature channels: 11 and 12.
- Output buckets: 1, 4, and 8.
- Explicit architecture metadata in exported nets and strict runtime validation.
- Trainer, exporter, scalar runtime, SIMD runtime, and parity-test support.
- True checkpoint resume preserving weights, Adam momentum and velocity, current
  superbatch, and learning-rate schedule position.

Exit gate: every matrix architecture exports, loads, and produces matching
trainer/runtime evaluations on the parity suite. A stopped short run must resume
to final weights within the measured variation between independent uninterrupted
CUDA runs.

## Stage 2: short training

Train all candidates before comparing them:

1. `enyo-16x12-1024-o8` - control.
2. `enyo-16x12-1024-o4`.
3. `enyo-16x12-1024-o1`.
4. `enyo-16x12-768-o8`.
5. `enyo-16x11-768-o8`.
6. `enyo-10x11-768-o8` - Enyo-derived king map, not a copied engine layout.
7. `enyo-32x11-1024-o8`.
8. `enyo-16x12-512-o1`.

Hold these variables fixed for every candidate:

- Random initialization; never use weights from another engine.
- The same 2.8B-position pylon Bullet file and record order.
- The same batch settings, WDL, filters, initial LR, and final LR.
- The same 65,536-superbatch schedule, stopped at 16,384 for screening.
- A complete optimizer checkpoint saved at superbatch 16,384.

Exit gate: all eight short nets and resumable checkpoints exist, pass static
validation, and have recorded training times and hashes.

## Stage 3: architecture screening

- Run 1,000 fixed-protocol games for each candidate against the short-trained
  `enyo-16x12-1024-o8` control.
- Eliminate a candidate after one controlled negative result. Do not immediately
  retry a rejected architecture.
- Advance the strongest three positive or statistically tied candidates.
- Run a three-match, 1,000-game round robin between those finalists.
- Do not run an eight-way all-pairs tournament; the maximum is ten matches.

Exit gate: select one winner. If two finalists remain indistinguishable, advance
both to Stage 4. If no candidate beats the control, the control wins.

## Stage 4: full winner training

- Resume the winner from superbatch 16,384 to 65,536.
- Preserve optimizer state and LR schedule position; do not restart training.
- Resume both finalists only when Stage 3 leaves a genuine tie.

Exit gate: the full net passes export, static, move, and runtime parity checks.

## Stage 5: full validation

Run sequential fixed-size tests:

1. Full winner versus the current fully trained Enyo baseline: 1,000 games.
2. If successful, full winner versus `default.net`: 1,000 games.
3. Only after beating `default.net`, full winner versus
   `~/code/cpp/chess/enyo/net/berserk-9b84c340af7e.nn`: 1,000 games.

The Berserk net is an opponent only. Its weights must never initialize, alter, or
otherwise influence an Enyo net.

Exit gate: establish the validated winner as the new native lineage root, or
record why the existing full Enyo baseline remains champion.

## Stage 6: incremental improvement

- Use `continue_from` for every same-architecture continuation.
- Change one meaningful training variable per rejected experiment.
- After one controlled rejection, move to the next documented hypothesis.
- Preserve a successful regimen and advance only the data slice.
- Run the fixed default-net benchmark periodically to measure absolute progress.

Expected cycle time is 30-45 minutes per normal train/gate/SPRT iteration, plus
20-30 minutes when a default-net benchmark is due.

## Experiment ledger

- `enyo-scratch-calibration-1.0.0-rc1` proved that random-init training can
  recover substantial strength and beat the previous fine-tuned lineage.
- `enyo-scratch-broad-1.0.0-rc1` extended the control recipe to the 2.8B pylon
  corpus and is the full 16x12x1024-o8 comparison baseline.
- `enyo-scratch-32bucket-1.0.0-rc1` changed only 16 to 32 input buckets and was
  rejected at -6.3 +/-15.0 Elo over 1,500 games. Do not promote or retrain that
  exact architecture in Stage 2.

Game results decide promotion. Static and move gates remain rejection filters,
not promotion evidence.
