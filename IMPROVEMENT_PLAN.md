# NNUE Improvement Plan

## Next experiment

- Candidate: `native-4.8.0-rc2`
- Parent: `native-4.7.0-rc4`
- Hypothesis: a third independent 200M pylon-data slice can reproduce the
  first slice's gain after the second slice was neutral.
- Fixed inputs: 8 output buckets, LR 0.0001, WDL 0.05, 256 superbatches, all
  trainable weights, and the existing architecture.
- Data: `training_data_pylon.binpack`, offset 400M, limit 200M.
- Judge with a 1500-game SPRT against the parent.
- Stop criterion: reject unless Elo and LLR are both positive at the cap.

## Durable results

- `native-4.5.0-rc5`: accepted at +4.2 Elo over 1500 games.
- `native-4.6.0-rc1`: accepted at +2.3 Elo over 1500 games.
- `native-4.7.0-rc1`: rejected at -3.2 Elo; the T60/T70 continuation lane is
  closed.
- `native-4.7.0-rc2` and `native-4.7.0-rc3`: neutral-negative on two
  wrongIsRight slices; that data lane is closed.
- `native-4.7.0-rc4`: accepted at +10.9 Elo on the first pylon slice.
- `native-4.8.0-rc1`: neutral at -0.5 Elo on the second pylon slice.
- The distributed LC0 depth-12 corpus produced catastrophic regressions at
  full and pilot-sized doses; do not reuse it without auditing label
  perspective, ordering, and score distribution.
