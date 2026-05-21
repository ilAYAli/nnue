# pack

`pack` converts scored JSONL rows into tensor files consumed by training.

## Commands

```sh
tools/pack/pack.py build --help
tools/pack/pack.py inspect --help
```

Example:

```sh
tools/pack/pack.py build \
  --input run/labeled.jsonl \
  --out-dir run/packed \
  --progress 200000
```

