# score

`score` attaches training targets to positions.

Input is usually JSONL from `posgen`. Output is scored JSONL suitable for
packing and training.

The command name is deliberately source-neutral: scores can come from
Stockfish, Enyo, another UCI engine, game results, or future blended targets.

## Commands

```sh
tools/score/score.py uci --help
```

Example:

```sh
tools/score/score.py uci \
  --input run/source.jsonl \
  --output run/labeled.jsonl \
  --engine ~/local/bin/stockfish \
  --depth 16 \
  --threads 1 \
  --hash 128
```

LC0 V6 gzip records can be labeled directly without JSONL intermediates:

```sh
python3 tools/score/label.py \
  --input ~/.cache/forge/inputs/DIGEST \
  --inventory ~/.cache/forge/inputs/DIGEST/inventory.json \
  --output shard.bullet \
  --stats shard.stats.json \
  --shard-count 152 \
  --shard-index 0
```

Files are assigned by sorted inventory ordinal modulo shard count. The global
raw-record limit is split exactly across shards, and the Bullet/stat outputs
are validated and atomically renamed.

`--inventory` is required for a reproducible distributed conversion, but an
individual `.tar` archive can be audited directly without one.

## LC0 -> Enyo calibrated Bullet (required pipeline)

`root_q` is only a source signal. It must never be written directly as an
Enyo training score. `label.py --score-source lc0-root` therefore refuses to
run without both `--enyo-runtime-target` and a valid `--lc0-calibration`
artifact.

Make that artifact from paired observations, not a guessed formula:

1. Forge-distribute deterministic samples of the LC0 records.  Each sample
   records the runtime-normalized white LC0-root score and the score from a
   fixed-depth Enyo search using a fixed native Enyo engine/net on the
   identical FEN. The sampler rejects fallback loading, missing engine-reported
   net SHA-256, a failed KQK search preflight, or all-zero sampled targets.

   ```sh
   forge run tools/forge/sample-lc0-calibration.template.json \
     --input ~/assets/training/lc0/test91-forge-input \
     --engine /path/to/enyo-engine \
     --net /path/to/compatible-native-enyo-net \
     --target-depth 1 \
     --shards 1600
   ```

2. Fit only the deterministic `fit` rows and require independent `holdout`
   improvement and scale checks. Sampling and holdout use separate hash
   domains, so both splits are populated. The default minimum is 50k fit and
   10k holdout pairs; failure produces no usable artifact.

   ```sh
   .venv/bin/python tools/score/lc0_calibration.py fit \
     --input /path/to/forge/*.pairs.jsonl \
     --output ~/assets/training/bullet/lc0/test91/test91.enyo-calibration.json
   ```

3. Use that immutable artifact in the distributed conversion. Forge stages
   the exact file on every worker, and every shard records its SHA-256.

   ```sh
   forge run tools/forge/label-lc0-root.template.json \
     --input ~/assets/training/lc0/test91-forge-input \
     --output ~/assets/training/bullet/lc0/test91/test91-enyo.bullet \
     --calibration ~/assets/training/bullet/lc0/test91/test91.enyo-calibration.json \
     --shards 1600
   ```

4. After Forge's atomic merge, attest the exact merged bytes against every
   shard stats file. This creates the required sidecar
   `<corpus>.calibration.json`.

   ```sh
   .venv/bin/python tools/validate/attest_lc0_calibration.py \
     --input ~/assets/training/bullet/lc0/test91/test91-enyo.bullet \
     --calibration ~/assets/training/bullet/lc0/test91/test91.enyo-calibration.json \
     --stats-dir /path/to/forge/shards \
     --manifest ~/assets/training/bullet/lc0/test91/test91-enyo.bullet.calibration.json
   ```

`nnue train` independently verifies that sidecar for every corpus below
`~/assets/training/bullet/lc0/`. A changed corpus, missing sidecar, invalid
artifact, or a shard from another artifact is a hard training failure.
The sampler only accepts fixed-depth search targets; it does not use Enyo's
currently unusable static `eval` extension. Each Forge task independently
performs the target-engine preflight before it writes a pair shard.

## Binpack count

Build the C++ score tools out of tree:

```sh
cmake -S tools/score -B /tmp/nnue-score-tools-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/nnue-score-tools-build --parallel
```

Count usable sfbinpack rows with the same default filters as the Bullet
sfbinpack conversion path: `min_ply=16`, `max_abs_cp=10000`, `quiet_only=1`,
and side to move not in check.

```sh
/tmp/nnue-score-tools-build/count_binpack data/nodes5000pv2_UHO.binpack
```

```sh
/tmp/nnue-score-tools-build/count_binpack --max-seen 1000000 data/nodes5000pv2_UHO.binpack
```
