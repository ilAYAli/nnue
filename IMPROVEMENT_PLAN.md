# NNUE Improvement Plan

## Next experiment

- Candidate: `enyo-4.39.0-rc5`; parent/gate reference: `enyo-4.38.0-rc2` (in
  flight at document update).
- Hypothesis: pylon prefix windows are exhausted at the accepted regimen;
  switch `source_binpack` to `test80-2022-08-aug-16tb7p.v6-dd.min.binpack`
  (offset 0, limit 200M), regimen otherwise unchanged.
- Regimen: float-head, 256 superbatches, LR 2e-5, WDL 0.05, quiet ply>=24.
- Current tip: `enyo-4.38.0-rc2` (+15.3 Elo vs `enyo-4.35.0-rc1` chain:
  +1.9, +3.0, +15.3 at float-head 2e-5 on fresh pylon prefixes).
- Closed from `enyo-4.38.0-rc2`: input block (-1.6 under ideal conditions),
  float-head retreads (0.0), dose-up 512sb (-5.8), pylon prefixes at
  2.8B/3.0B (toxic: -11.8, -43.6 aborted). The 32-bucket migration lane
  below remains closed; see Durable results.
- Fallback if test80 rejects: harvest unread remainders of proven pylon
  windows (e.g. offset 2.7B, limit 100M).

## Durable results

- `enyo-4.27.0-rc2` is the accepted parent at +7.2 Elo and LLR 0.32/2.20.
- Alternating gains and rejections through `enyo-4.28.0-rc1` show the
  16-superbatch 16x12 pylon lane is saturated/noisy.
- `native-4.19.0-rc5` was promoted at +4.4 +/-14.6 Elo and LLR 0.15/2.20 on
  pylon offset 2.2B.
- `native-4.20.0-rc1` was rejected at -1.2 +/-14.3 Elo and LLR -0.09/2.20
  on pylon offset 2.4B.
- `native-4.17.0-rc2` gained +13.9 +/-14.8 Elo over its parent after reducing
  the dose to 64 superbatches on pylon offset 1.2B.
- Fixed 1000-game testing put `native-4.17.0-rc2` at -294.6 +/-35.1 Elo versus
  `default.net`, with a 15.4% draw rate.
- `native-4.18.0-rc4` was initially promoted at +8.3 +/-14.8 Elo, but direct
  replay against `native-4.17.0-rc2` rejected it at -17.6 +/-15.7 Elo after
  1360 games. It is not a valid parent.
- Fixed 1000-game testing put `native-4.18.0-rc4` at -351.1 +/-39.5 Elo versus
  `default.net`, consistent with the direct replay regression.
- `native-4.19.0-rc1` and `native-4.19.0-rc2`, both based on the invalid rc4
  parent, were rejected at -1.4 and -4.2 Elo respectively.
- Pylon offsets 1.4B through 2B are closed after repeated neutral or negative
  results across dose and LR changes.
- `native-4.19.0-rc3`: rejected at -0.7 +/-14.7 Elo and LLR -0.06/2.20 on
  `farseerT76` offset 0.
- `native-4.19.0-rc4`: rejected at -0.2 +/-14.6 Elo and LLR -0.06/2.20 on
  `farseerT76` offset 200M.
- The accepted lineage uses 8 output buckets and Stockfish binpacks. LC0 and
  the tested PV2 data lanes regressed and remain closed.

## Data consumption semantics (2026-07-03)

- The trainer uses bullet's `DirectSequentialDataLoader` and the
  binpack-to-bullet conversion preserves order; nothing shuffles
  (`convert_sfbinpack.rs`, `bullet.py`, `train.rs`).
- A run therefore consumes only a sequential prefix of its
  `data.offset`/`data.limit` window: samples = superbatches x 64 x 2048.
  256 superbatches is ~34M positions, 512 is ~67M, 1024 is ~134M.
- Every ledger conclusion about a "200M slice" applies only to that prefix.
  Example: the +15.3 Elo `enyo-4.38.0-rc2` trained on pylon 2.600B-2.634B
  only; "toxic" pylon 2.8B/3.0B means the ~34M prefixes at those offsets.
- The unread ~166M remainder of each tested 200M window is untested data.
- Do not change the loader or conversion to shuffle mid-lineage: every ledger
  result, including accepted parents, was trained under prefix semantics and
  comparability would be lost.
- Replay validation (2026-07-03): `enyo-4.38.0-rc2` vs `enyo-4.37.0-rc1`
  re-measured at +4.6 +/-15.1, LLR 0.20, LOS 72.7% over 1500 games
  (`enyo-4.38.0-rc2-replay-vs-4.37-20260703-095532`). The original +15.3 was
  winner-curse inflated but the promotion is valid; the parent stands.
  Promotion-gate Elo numbers overstate true gains on average.
- Fine-tune exhaustion (2026-07-03): eleven consecutive rejections from the
  replay-validated `enyo-4.38.0-rc2` closed every configuration axis: blocks
  (input -1.6, float-head 0.0 to -11.8, all -9.5/-3.5), LR (5e-5/2e-5/1e-5),
  dose (256/512/1024/4096 superbatches, including the first multi-epoch run,
  -3.5), data (pylon regions, test80 at two LRs), and objective (wdl 0.2,
  -9.3). The parent is a hard local optimum for one-shot fine-tuning at
  1500-game gate precision. Next gains require a structural change:
  reference-scale from-scratch training or an architecture delta.
