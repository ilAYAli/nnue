# train

`train` trains and evaluates Enyo NNUE candidate nets.

## Commands

```sh
tools/train/train.py run --help
tools/train/train.py eval --help
```

Example:

```sh
tools/train/train.py run \
  --data run/packed \
  --init-from-nn ~/code/cpp/chess/enyo/nnue/berserk-d43206fe90e4.nn \
  --objective huber \
  --target-clamp 800 \
  --lr 7e-7 \
  --epochs 8 \
  --out run/candidate/model.pt \
  --out-nn run/candidate/model.nn
```

