# train

`train` contains implementation modules for Enyo NNUE training.

Use the single public training frontend from the repo root:

## Commands

```sh
./nnue-train supervised --help
./nnue-train pairwise --help
./nnue-train move-policy --help
./nnue-train eval --help
```

Example:

```sh
./nnue-train supervised \
  --data run/packed \
  --init-from-nn ~/code/cpp/chess/enyo/net/berserk-d43206fe90e4.nn \
  --objective huber \
  --target-clamp 800 \
  --lr 7e-7 \
  --epochs 8 \
  --out run/candidate/model.pt \
  --out-nn run/candidate/model.nn
```
