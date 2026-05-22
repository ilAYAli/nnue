# train

`train` trains and evaluates Enyo NNUE candidate nets.

## Commands

```sh
tools/train/train.py run --help
tools/train/train.py eval --help
tools/train/train_pairwise.py --help
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

Pairwise training keeps the broad scalar eval loss, but adds child-position
ranking pairs. Use it when a candidate fails by choosing the wrong move even
though scalar cp fitting looks acceptable.

With `--scores-csv`, the trainer uses the scored legal-move table produced by
the move-choice gate. Add `--candidate-moves-csv` to override which move is
treated as the bad move for each target, for example after a newer candidate
chooses different bad moves than the original scored candidate.
