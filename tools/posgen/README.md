# posgen

`posgen` is the Enyo NNUE position-generation front end.

It owns the position side of the pipeline:

```text
self-play PGN -> extracted JSONL rows -> sampled/filtered position JSONL
```

It deliberately does not own teacher labeling. Stockfish labeling is a separate
step because the teacher/oracle choice is an experiment variable.

## Commands

```sh
tools/posgen/posgen.py selfplay --help
tools/posgen/posgen.py extract --help
tools/posgen/posgen.py sample --help
tools/posgen/posgen.py instability --help
```

Typical flow:

```sh
tools/posgen/posgen.py selfplay \
  --engine ~/code/cpp/chess/assets/engines/reference \
  --nnue-file ~/code/cpp/chess/enyo/nnue/current-reference.nn \
  --book ~/code/cpp/chess/assets/books/UHO_Lichess_4852_v1.epd \
  --games 160000 \
  --depth 8 \
  --output run/selfplay.pgn

tools/posgen/posgen.py extract run/selfplay.pgn \
  --output run/selfplay.jsonl \
  --skip-plies 8 \
  --min-depth 8 \
  --max-abs-cp 1600

tools/posgen/posgen.py sample \
  --input run/selfplay.jsonl \
  --output run/source_signed.jsonl \
  --preset signed-balanced-v1 \
  --unique-fen \
  --seed 2026051501
```

Then continue with teacher labeling, packing, training, and validation.
