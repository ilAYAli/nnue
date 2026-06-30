# NNUE Improvement Plan

## Next experiment

- Candidate: `native-4.19.0-rc3`; parent: `native-4.17.0-rc2`.
- Correct the lineage after direct replay disproved the `native-4.18.0-rc4`
  promotion.
- Switch the saturated pylon continuation to `farseerT76.binpack`, offset 0,
  limit 200M.
- Keep 16 input buckets, 8 output buckets, 64 superbatches, LR 0.00005, WDL
  0.05, and all other training parameters unchanged.
- Judge with a 1500-game SPRT; reject unless Elo and LLR are both positive at
  the cap.

## Durable results

- `native-4.17.0-rc2` is the accepted parent. It gained +13.9 +/-14.8 Elo over
  its parent after reducing the dose to 64 superbatches on pylon offset 1.2B.
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
- The accepted lineage uses 8 output buckets and Stockfish binpacks. LC0 and
  the tested PV2 data lanes regressed and remain closed.
