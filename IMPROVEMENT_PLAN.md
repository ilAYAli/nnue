# NNUE Improvement Plan

## Next experiment

- Candidate: `native-3.5.0-rc1`
- Parent: `native-3.3.0-rc3`
- Hypothesis: switching from saturated farseerT74 continuation data to
  farseerT75 may improve the accepted 4-output-bucket parent.
- Fixed inputs: 200M positions, LR 0.0001, WDL 0.15, 4096 superbatches, and
  the existing architecture.
- Judge with an 800-game SPRT against the parent.
- Stop criterion: reject if the game test is negative.

## Durable results

- `native-3.3.0-rc3`: accepted at +14.8 Elo over its reference.
- `native-3.4.0-rc1`: neutral at -3.4 Elo over 3000 games.
- `native-3.4.0-rc2`: neutral at -1.7 Elo over 800 games.
- CP-only Enyo self-play D20 fine-tuning was rejected early.
- `native-3.4.0-rc3`: stopped early at 360/800 games, -31.0 Elo.
- The farseerT74 offset-only continuation lane is closed.
