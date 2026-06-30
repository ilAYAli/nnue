# NNUE Improvement Plan

## Next experiment

- Candidate: `native-4.18.0-rc3`; parent: `native-4.17.0-rc2`.
- Double training exposure on the 1.4B-offset pylon slice by increasing the
  dose from 64 to 128 superbatches at LR 0.000025.
- Keep 16 input buckets, 8 output buckets, WDL 0.05, and all other training
  parameters unchanged.
- This restores the nominal LR-times-steps of the 64-superbatch LR 0.00005
  regimen while exposing the net to twice as many examples.
- Judge with a 1500-game SPRT; reject unless Elo and LLR are both positive at
  the cap.

## Durable results

- `native-4.13.0-rc1`: accepted at +9.5 +/-14.5 Elo after enabling the shared
  input factoriser.
- `native-4.14.0-rc4`: accepted at +5.3 +/-14.6 Elo using 128 superbatches on
  the 600M-offset pylon slice.
- `native-4.15.0-rc2`: accepted at +2.5 +/-14.9 Elo after reducing LR to
  0.00005 on the 800M-offset pylon slice.
- `native-4.16.0-rc1`: accepted at +3.2 +/-14.8 Elo on the 1B-offset pylon
  slice.
- `native-4.17.0-rc2`: accepted at +13.9 +/-14.8 Elo after reducing the dose
  to 64 superbatches on the 1.2B-offset pylon slice.
- Fixed 1000-game testing put `native-4.17.0-rc2` at -294.6 +/-35.1 Elo versus
  `default.net`, with a 15.4% draw rate.
- `native-4.18.0-rc1`: rejected at -3.9 +/-15.1 Elo using 64 superbatches and
  LR 0.00005 on the 1.4B-offset pylon slice.
- `native-4.18.0-rc2`: rejected at -6.9 +/-15.1 Elo after reducing LR alone to
  0.000025 on the same slice.
- The accepted lineage uses 8 output buckets and Stockfish binpacks. LC0,
  nodes5000 PV2 UHO, and PV2 difference-100 nodes-5000 data regressed and are
  closed pending a demonstrated data-quality correction.
