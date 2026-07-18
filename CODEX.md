# NNUE Handoff for Next Codex

Current state:

- Repo: `/home/petter/code/cpp/chess/nnue`
- Work only on `main` in `pwa-5090`
- Do not pull
- Do not poll Forge or logs; wait for `AI_stdin` events only
- During an active lane, the tracked diff should stay limited to `build.json` and, only when justified, `architecture.json`

Current candidate:

- `enyo-5.0.0-rc1`
- Hypothesis: add only a zero-initialized material-bucketed PSQT residual while preserving and freezing the protected champion path
- Config at failure:
  - `initialize_from`: `enyo-1.20.0-rc12`
  - `reference`: `enyo-1.20.0-rc12`
  - `trainable`: `psqt`
  - `superbatches`: `256`
  - `lr`: `0.00005`
  - `wdl`: `0.05`
  - `data.source_binpack`: `data/stockfish/master-binpacks/training_data_pylon.binpack`
  - `data.limit`: `200000000`
  - `data.offset`: `600000000`
  - `sfbinpack.min_ply`: `24`

What happened:

- Training completed.
- Export failed with:
  - `checkpoint has unexpected 393248 byte trailer`
  - `FAILED: train/export failed for enyo-5.0.0-rc1 train`
- This is a trainer/export format problem, not an Elo rejection.
- Fix the export/checkpoint path before changing experiment config again.

Useful prior results:

- `enyo-4.29.0-rc7` `output` 1024: `+8.3 Elo`
- `enyo-4.30.0-rc1` `output` 1024: `+6.0 Elo`
- `enyo-4.31.0-rc4` `float-head` 1024: `+3.5 Elo`
- `enyo-4.32.0-rc2` `float-head` 1024: `+4.6 Elo`
- `enyo-4.33.0-rc3` `float-head` 256: `+6.7 Elo`
- `enyo-4.34.0-rc3` `input` 256: `+5.6 Elo`
- `enyo-4.35.0-rc1` `input` 256: `+6.3 Elo`
- `enyo-4.36.0-rc1` `input` 256: rejected at `+0.5 Elo`

Architecture hypotheses worth testing after the export bug is fixed:

1. Reduce `output_buckets` from `8` to `4` while keeping the PSQT residual and frozen champion path.
   - Reason: test whether the residual needs only coarse material conditioning and whether a smaller bucketed head is easier to optimize and export cleanly.

2. Keep `output_buckets = 8` but disable `input_factoriser`.
   - Reason: isolate whether the factoriser is interacting poorly with the additive residual path or contributing unnecessary complexity.

3. Keep the current PSQT residual and `output_buckets = 8`, but reduce `hidden` from `1024` to `768`.
   - Reason: test whether the current architecture is overparameterized for this pylon slice and whether a smaller dense core generalizes better.

Rules for the next instance:

- Do not change architecture until the export bug is understood or fixed.
- Change one meaningful variable per experiment.
- Do not combine data, architecture, learning-rate, WDL, and dose changes.
- If an experiment is positive, keep the regimen and advance only the data slice.
- Relaunch only with the exact NNUE iterate command once the toolchain is healthy again.
