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

The active lane remains the simple 1-bucket architecture for now. This
is only the first failed candidate in the family, so do not close it yet.
The next experiment must mutate exactly one non-architecture variable.

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

### Iteration 2 — `native-9.1.0-rc1`

Same architecture, data, dose, WDL, and learning-rate family as
`native-9.0.0-rc1`. Mutate only the L0 initialization scale to compensate
for the Enyo quantized path dividing L0 activations by 32:

- set `ENYO_BULLET_ENYO_L0_STD=256`
- keep `ENYO_BULLET_ENYO_L1_STD` at the default unless this candidate is
  rejected
- keep input buckets at 1

Promotion criterion: 100-game smoke vs `default.net` with LOS >= 50%,
then a longer SPRT. Per `AGENTS.md`, three consecutive rejected
candidates from the same family close the family.

### Iteration 2+ outlook

Conditional on iteration 1 producing a measurable candidate.

Reference NNUE architectures from top engines:

|                                  | Input buckets | L1   | L2 | L3 | Output buckets | Extras                       |
|----------------------------------|---------------|------|----|----|----------------|------------------------------|
| Enyo (current)                   | 16            | ~1024 | —  | —  | 1 (est.)       | —                            |
| Alexandria 7.0.0 (CCRL Blitz #6) | 16            | 1536 | 16 | 32 | 8              | —                            |
| Reckless (CCRL Blitz #2)         | 10            | 768  | 16 | 32 | 8              | threats (+66864 ft features) |

Iteration 2+ priority, one variable family per iteration:

1. **Add 8 output buckets** (Enyo source change, small). Both top
   references use 8; Enyo at 1 is the clearest current gap. Smallest
   effort, largest expected payoff.
2. **Drop to ~10 input buckets** (Enyo source change: new
   `KING_BUCKETS_10` table + `IsSupportedFeatureLayout` extension).
   Matches Reckless; reduces data starvation per bucket.
3. **Add L2/L3 hidden layers** (16 → 32 → 1×OUTPUT_BUCKETS). Matches
   both references. Requires runtime extension on the engine side.
4. **Threat features** as a parallel accumulator (Reckless-style).
   Prior local work measured -39% NPS. Defer until 1–3 prove out.
5. **L1 width** is bottom-priority. Alexandria (1536) and Reckless (768)
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
