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

