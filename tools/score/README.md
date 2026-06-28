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
python3 tools/score/label_lc0.py \
  --input ~/.cache/crucible/inputs/DIGEST \
  --inventory ~/.cache/crucible/inputs/DIGEST/inventory.json \
  --output shard.bullet \
  --stats shard.stats.json \
  --shard-count 152 \
  --shard-index 0
```

Files are assigned by sorted inventory ordinal modulo shard count. The global
raw-record limit is split exactly across shards, and the Bullet/stat outputs
are validated and atomically renamed.

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
