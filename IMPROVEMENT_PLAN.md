# Enyo NNUE Improvement Plan

Durable conclusions and the active hypothesis only. See `AGENTS.md` for
roles, rules, and validation order. Run-by-run logs belong in git history
and run directories, not in this file.

## Status

The `native-1.x`–`native-7.x` lineage is rejected pending re-test. A month
of iteration ran against a validation chain that did not prove the engine
was loading the candidate net: missing `nnue_file` silently fell back to
default/legacy behavior, no startup line confirmed evaluator type or file
hash, and static gates were treated as promotion evidence without verified
game results. The reported `uho52_lr1e4_e64.nn` -730 Elo result is
uninterpretable, not a measurement.

The first simple-architecture candidate, `native-9.0.0-rc1`, proved the
pipeline but failed the game gate. Export and engine loading were correct:
the engine loaded the candidate as native NNUE with `hidden=1024`,
`input_buckets=1`, and `feature_channels=12`. The 100-game smoke vs
`default.net` failed cleanly with no warnings: Elo -381.7 +/- 128.6,
LOS 0.0%, draw 0.0%. Do not promote it and do not run the 1k SPRT.

The simple 1-bucket init-scale family is closed. Three candidate settings
loaded correctly but failed game smoke badly. Do not start another
near-identical 1-bucket init-scale run. Spike-trainer export/eval parity
has now been checked on the 1-bucket architecture: Python `.nn` load and
the Enyo engine eval path agree within 2 cp on the fixed FEN set for
`native-9.2.0-rc1`. The failure is not a byte layout, loader, or eval-path
parity issue.

The first output-bucket-only candidate, `native-10.0.0-rc1`, is also
rejected. It loaded correctly as `hidden=1024`, `input_buckets=1`,
`feature_channels=12`, `output_buckets=8`, and passed Python `.nn` vs
Enyo `evalnet` parity within 2 cp. The 100-game smoke then lost every
game: Elo -inf, LOS 0.0%, draw 0.0%, with no warnings. Do not retry
8 output buckets on the 1-input-bucket base.

## Active Hypothesis

### Iteration 0 — pipeline parity (not a candidate)

The rebuilt validation chain reports a known-good result correctly. Train
a tiny config on a 100k-row slice of `data/bullet/lc0q10m.bullet` to
produce a loadable `.nn`. Run train/export/engine parity. Run a 50-game
SPRT vs `default.net` to verify both sides log the expected evaluator and
file hash. The candidate is expected to lose; the test is whether the
chain reports it correctly.

### Iteration 1 — `native-9.0.0-rc1` rejected

Lc0 test91 Q-targets at 10M rows did not produce a playable simple
Enyo-native baseline from the default spike trainer Enyo init scale.
One variable family changed: architecture. The resulting net loaded and
searched correctly, but game strength was far below `default.net`.

`changed_variables`:
- `architecture`: 1 input bucket, 12 piece-square channels, 1024 hidden,
  1 output bucket
- `source_bullet`: `data/bullet/lc0q10m.bullet`

Result: rejected by the quick smoke. Do not rerun unchanged.

### Iteration 2 — `native-9.1.0-rc1` rejected

Same architecture, data, dose, WDL, and learning-rate family as
`native-9.0.0-rc1`. Mutate only the L0 initialization scale to compensate
for the Enyo quantized path dividing L0 activations by 32:

- set `ENYO_BULLET_ENYO_L0_STD=256`
- keep `ENYO_BULLET_ENYO_L1_STD` at the default unless this candidate is
  rejected
- keep input buckets at 1

Result: rejected by the quick smoke. The net loaded as native NNUE with
`hidden=1024`, `input_buckets=1`, and `feature_channels=12`, and export
trimmed correctly to 1,610,052 bytes. The fixed smoke position produced
a wildly high `+2045cp` depth-1 eval. The 100-game smoke vs `default.net`
finished Elo -798.2, LOS 0.0%, draw 0.0%, with one reference
responsiveness warning. Do not promote it and do not run the 1k SPRT.

Conclusion: full 32x L0 init compensation is too large for this path.
The init-scale theory is not closed, but `l0_std=256` is rejected.

### Iteration 3 — `native-9.2.0-rc1` rejected

Same architecture, data, dose, WDL, and learning-rate family as
`native-9.0.0-rc1`. Mutate only the L0 initialization scale to a smaller
single compensation value after `native-9.1.0-rc1` overshot badly:

- set `training.l0_std=32.0`
- keep `training.l1_std=1.0`
- keep input buckets at 1

Result: rejected by the quick smoke. The net loaded as native NNUE with
`hidden=1024`, `input_buckets=1`, and `feature_channels=12`. The fixed
smoke position produced a sane-looking `+5cp` depth-1 eval, but the game
smoke was catastrophically negative: at 62/100 games, Elo -714.1,
LOS 0.0%, draw 0.0%. The tournament then stopped on a transient reference
engine startup failure; direct startup checks for both `default.net` and
the candidate passed afterward. The partial result is already sufficient
to reject the candidate. Do not promote it and do not run the 1k SPRT.

Conclusion: init-scale tuning did not rescue the simple 1-bucket lane.
Close this family and inspect spike-trainer export/eval parity before
more training.

### Iteration 4 — parity diagnostic passed

The deterministic parity diagnostic for the 1-bucket Enyo-native
spike-trainer path passed on `native-9.2.0-rc1`:

- Python `.nn` load vs Enyo `evalnet` agreed within 2 cp;
- tested FENs included startpos, a normal middlegame, castling-rights
  position, and an en-passant FEN;
- local tooling now supports 1/2/4/8/16/32 input-bucket Enyo `.nn` sizes.

Conclusion: the next candidate should change architecture or objective.
Do not spend more games on 1-bucket init-scale variants.

### Iteration 5 — `native-10.0.0-rc1` rejected

Added output buckets before reintroducing more input buckets. This changed
one architecture family while keeping the known-good 1-input-bucket feature
layout:

- `input_buckets`: 1
- `output_buckets`: 8
- `hidden`: 1024
- `feature_channels`: 12

Result: rejected by the quick smoke. Export trimmed correctly to 1,610,976
bytes. The engine loaded it as native NNUE with the intended 8 output
buckets, the fixed smoke position returned `+5cp`, and Python `.nn` vs
engine `evalnet` parity passed within 2 cp. The 100-game smoke lost every
game: Elo -inf, LOS 0.0%, draw 0.0%, with no warnings.

Conclusion: output buckets alone do not rescue the 1-input-bucket base.
Move to the input-bucket lane.

### Iteration 6 — `native-11.0.0-rc1` input-bucket hypothesis

Reintroduce moderate king conditioning without returning to Enyo's full
16-bucket layout yet. This changes one architecture family from the last
playable-equivalent base:

- `input_buckets`: 8
- `runtime_input_buckets`: 8
- `output_buckets`: 1
- `hidden`: 1024
- `feature_channels`: 12

Keep data, dose, WDL, LR, and `l0_std=32.0` unchanged. Gate in the same
order: export, load/eval metadata, Python `.nn` vs engine parity, then
the 100-game smoke. If this still loses every game, stop architecture
churn and change the data/objective theory instead.

Reference NNUE architectures from top engines:

|                                  | Input buckets | L1   | L2 | L3 | Output buckets | Extras                       |
|----------------------------------|---------------|------|----|----|----------------|------------------------------|
| Enyo (current)                   | 16            | ~1024 | —  | —  | 1 (est.)       | —                            |
| Alexandria 7.0.0 (CCRL Blitz #6) | 16            | 1536 | 16 | 32 | 8              | —                            |
| Reckless (CCRL Blitz #2)         | 10            | 768  | 16 | 32 | 8              | threats (+66864 ft features) |

Iteration 2+ priority, one variable family per iteration:

1. **Drop to ~10 input buckets** (Enyo source change: new
   `KING_BUCKETS_10` table + `IsSupportedFeatureLayout` extension).
   Matches Reckless; reduces data starvation per bucket.
2. **Add L2/L3 hidden layers** (16 → 32 → 1×OUTPUT_BUCKETS). Matches
   both references. Requires runtime extension on the engine side.
3. **Threat features** as a parallel accumulator (Reckless-style).
   Prior local work measured -39% NPS. Defer until 1–3 prove out.
4. **L1 width** is bottom-priority. Alexandria (1536) and Reckless (768)
   are both top-10; width alone is not a strength gate.

Data scale-up — HF Stockfish, Lc0 FENs, or re-extracting the existing
test91 tar — is an independent lever and can run in parallel with any
of these. The preferred ingestion format for non-Lc0 sources is
binpack: Stockfish publishes training data in binpack, Bullet has
native `SfBinpackLoader` / `MontyBinpackLoader` / `ViriBinpackLoader`,
and binpack is lossless relative to BulletFormat (which discards
side-to-move, halfmove counter, and castling rights). Adding binpack
ingestion is a small `tools/bullet/spike_trainer` change to select
the loader by source-file extension; do not write a custom binpack
parser, and do not add a binpack-to-BulletFormat conversion step.

## Closed Lanes

Do not restart without a new representation or objective:

- `native-1.x`–`native-7.x` as a lineage. Two representative nets
  (`uho52_lr1e4_e64.nn`, `enyo-native-7.6.0-rc1.nn`) are kept for a
  one-time post-fix re-test with the corrected `nnue_file=...default.net`
  form. The lineage is otherwise rejected and not a baseline.
- Static-only promotion. MAE/sign improvements on heldout data do not
  predict game strength.
- Repo-local SPRT wrappers. Game validation goes through forge/Crucible.
- Move-policy sidecar runtime path.
- Architecture or features that fail export parity, engine-static gate,
  or NPS before games.

## Required Sanity Test Form

Every candidate SPRT uses the explicit `nnue_file=` form on both sides.
The engine logs evaluator type, absolute `nnue_file` path, and the
SHA-256 of the loaded weights on startup; the SPRT runner asserts these
match the expected values before counting any game. Missing or invalid
`nnue_file` is a hard startup failure, never a silent fallback.

```sh
sprt \
  --reference ~/assets/engines/enyo_HASH \
  --reference-uci "nnue_file=$HOME/code/cpp/chess/enyo/net/default.net" \
  --candidate ~/assets/engines/enyo_HASH \
  --candidate-uci "nnue_file=/absolute/path/to/candidate.nn" \
  --games 300
```

## Pre-Training Gates

In order. Stop early on any failure. Do not rescue a failed family with
near-identical reruns.

1. `./nnue-run doctor -c build.json` passes, including the SPRT preflight
   that dry-runs both engines and asserts evaluator type and weight hash
   on both sides.
2. Training and export complete and produce one intended `.nn`.
3. Train/export/engine parity passes on the produced net within
   quantization tolerance.
4. 100-game smoke vs `default.net` is neutral-positive.
5. Longer SPRT only after smoke passes.

## Active Workflow

Use `./nnue-run` driven by `build.json`. The config states `hypothesis`,
`changed_variables`, ordered `stages`, and pass/fail `gates`. One commit
per iteration changing only `build.json` (and a deliberate update to this
file when the hypothesis line changes). Do not start a training run from
a dirty tree otherwise.

Prefer the Bullet/BulletLib path for training. Large data conversion or
mixing goes through compiled C++ tools called from `build.py`.
