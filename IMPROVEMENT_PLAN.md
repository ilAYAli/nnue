# NNUE Improvement Plan

## Next experiment

- Candidate: `enyo-5.0.0-rc1`; parent/gate reference: `enyo-4.27.0-rc2`.
- Migrate the accepted parent from 16 input buckets / 12 feature channels to 32
  input buckets / 11 feature channels.
- Keep hidden=1024, l2_size=16, output_buckets=8, input factorisation, eval
  scales, pylon source, offset 5.4B, limit 200M, 16 superbatches, LR 0.00005,
  and WDL 0.05 unchanged.
- Judge against `enyo-4.27.0-rc2`; reject unless Elo and LLR are both positive
  at the cap.

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
