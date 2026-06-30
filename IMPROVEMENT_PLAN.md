# NNUE Improvement Plan

## Next experiment

- Candidate: `native-4.17.0-rc2`; parent: `native-4.16.0-rc1`.
- Retry the neutral 1.2B-offset pylon slice with the dose reduced from 128 to
  64 superbatches.
- Keep 16 input buckets, 8 output buckets, LR 0.00005, WDL 0.05, and all other
  training parameters unchanged.
- Train on `training_data_pylon.binpack`, offset 1.2B, limit 200M.
- Judge with a 1500-game SPRT; reject unless Elo and LLR are both positive at
  the cap.

## Durable results

- `native-4.13.0-rc1`: accepted at +9.5 +/-14.5 Elo and LLR +0.40/2.20 over
  1500 games. It enabled the shared input factoriser on the 200M-offset pylon
  slice.
- `native-4.14.0-rc1`: rejected at -5.3 +/-14.7 Elo and LLR -0.26/2.20 over
  1500 games. It actually used the 400M-offset pylon slice; its replay comment
  was inaccurate.
- `native-4.14.0-rc2`: rejected at -8.1 +/-14.3 Elo and LLR -0.44/2.20 over
  1500 games. Increasing input king buckets from 16 to 32 is closed pending
  evidence that justifies revisiting it.
- `native-4.14.0-rc3`: rejected at -4.9 +/-14.8 Elo and LLR -0.21/2.20 over
  1500 games using 128 superbatches on the 400M-offset pylon slice.
- `native-4.14.0-rc4`: accepted at +5.3 +/-14.6 Elo and LLR +0.17/2.20 over
  1500 games using the same regimen on the 600M-offset pylon slice.
- `native-4.15.0-rc1`: rejected at -2.6 +/-14.7 Elo and LLR -0.14/2.20 over
  1500 games using LR 0.0001 on the 800M-offset pylon slice.
- `native-4.15.0-rc2`: accepted at +2.5 +/-14.9 Elo and LLR +0.09/2.20 over
  1500 games after reducing LR to 0.00005 on the same 800M slice.
- `native-4.16.0-rc1`: accepted at +3.2 +/-14.8 Elo and LLR +0.14/2.20 over
  1500 games using the lower-LR regimen on the 1B-offset pylon slice.
- `native-4.17.0-rc1`: rejected at -0.2 +/-14.3 Elo and LLR -0.04/2.20 over
  1500 games using 128 superbatches on the 1.2B-offset pylon slice.
- The accepted lineage uses 8 output buckets and Stockfish binpacks. LC0,
  nodes5000 PV2 UHO, and PV2 difference-100 nodes-5000 data regressed and are
  closed pending a demonstrated data-quality correction.
