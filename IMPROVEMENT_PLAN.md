# NNUE Improvement Plan

## Next experiment

- Benchmark accepted `native-4.18.0-rc4` against `default.net` with the fixed
  1000-game `qsprt.sh` protocol.
- After the benchmark, continue with `native-4.19.0-rc2` from
  `native-4.18.0-rc4` on the next unused pylon slice at offset 2B.
- Keep 16 input buckets, 8 output buckets, 64 superbatches, LR 0.00005, WDL
  0.05, and all other training parameters unchanged.
- Judge training candidates with a 1500-game SPRT; reject unless Elo and LLR
  are both positive at the cap.

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
- The 1.4B-offset pylon slice rejected three dose/LR combinations.
- `native-4.18.0-rc4`: accepted at +8.3 +/-14.8 Elo using the 64-superbatch,
  LR 0.00005 regimen on the 1.6B-offset pylon slice.
- `native-4.19.0-rc1`: rejected at -1.4 +/-14.7 Elo and LLR -0.05/2.20 on the
  1.8B-offset pylon slice.
- The accepted lineage uses 8 output buckets and Stockfish binpacks. LC0,
  nodes5000 PV2 UHO, and PV2 difference-100 nodes-5000 data regressed and are
  closed pending a demonstrated data-quality correction.
