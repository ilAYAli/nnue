# train

`train` trains and evaluates Enyo NNUE candidate nets.

## Commands

```sh
tools/train/train.py run --help
tools/train/train.py eval --help
tools/train/train_pairwise.py --help
tools/train/train_search_aware.py --help
```

Example:

```sh
tools/train/train.py run \
  --data run/packed \
  --init-from-nn ~/code/cpp/chess/enyo/nnue/current-reference.nn \
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

## Search-Aware Training

`train_search_aware.py` blends the normal broad scalar Huber loss with a
multi-move target loss. Each target row contains one parent FEN and scored legal
moves. The trainer pushes those moves, evaluates child positions, and trains:

- a margin loss: worse children should evaluate worse than the best child by
  approximately the scored gap.
- a soft-policy loss: the net should rank child moves according to the target
  policy distribution.

`--search-tag-weights mate_like=8,non_mate=1` can up-weight specific target
tags without changing the default unweighted behavior.

Use this through `./build.py create --backend search-aware`; direct invocation is
for debugging the backend only.
