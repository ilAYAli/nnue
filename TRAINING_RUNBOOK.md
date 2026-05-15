# Enyo NNUE Training Runbook

Concise guide for starting, choosing, and judging Enyo NNUE runs.

## Environment

Use `/home/petter/.venv/bin/python` for NNUE Python on `pwa-5090`.

## Main Scripts

```sh
cd ~/code/cpp/chess/nnue
git pull --ff-only
```

- `tools/nnue2/train_new_net_pwa.sh`: default self-play + Lichess run.
- `tools/nnue2/run_d16_expansion_pwa.sh`: slower Stockfish depth-16 relabel run.
- `tools/nnue2/run_net_sprt_pwa.sh`: SPRT an existing `.nn`.
- `tools/nnue2/pack_dataset.py`: JSONL positions -> packed tensors.
- `tools/nnue2/train.py`: PyTorch training -> `.pt` and `.nn`.
- `tools/nnue2/eval_dataset.py`: static validation.

## One-Command Training

Run this when you want the default pipeline to try producing a stronger net:

```sh
tools/nnue2/train_new_net_pwa.sh
```

It samples rows from the two default JSONL inputs below, packs them into
tensors, trains a candidate `.nn`, runs static checks, then runs SPRT. The
result is only a keeper if SPRT is clearly positive.

Default inputs:

- self-play teacher rows:
  `~/tmp/enyo_teacher/sf_d12_20m_20260510_115338/labeled.jsonl`
  Enyo self-play positions evaluated by a teacher, e.g. Stockfish.
- Lichess eval rows:
  `~/tmp/enyo_teacher/lichess_eval_d18_standard/lichess_eval.jsonl`
- starting net:
  `~/code/cpp/chess/enyo/nnue/berserk-d43206fe90e4.nn`
- test engine:
  `~/code/cpp/chess/assets/engines/reference`
- opening book:
  `~/code/cpp/chess/assets/books/UHO_Lichess_4852_v1.epd`

Default values:

- `8,000,000` self-play rows
- `2,000,000` Lichess eval rows
- `mpe25` objective
- learning rate `3e-7`
- target clamp `1200`
- `4` epochs
- SPRT `4000` games at `2+0.02`

Running the exact same command again mostly tests random sampling/training
noise. For a meaningful new experiment, change one of:

- row mix: `--selfplay-rows`, `--lichess-rows`
- learning rate: `--lr`
- target clamp: `--target-clamp`
- epochs: `--epochs`
- SPRT length: `--sprt-games`

Example larger run:

```sh
tools/nnue2/train_new_net_pwa.sh \
  --selfplay-rows 12000000 \
  --lichess-rows 3000000 \
  --epochs 4 \
  --lr 3e-7 \
  --target-clamp 1200
```

## Depth-16 Target Upgrade

Run this when the goal is better labels instead of just another random sample
from the existing data:

```sh
tools/nnue2/run_d16_expansion_pwa.sh
```

It samples signed score buckets from the existing 20M self-play pool, relabels
them with Stockfish depth 16, trains Huber/cp800 candidates, static-checks them,
and runs SPRT only for static-positive candidates.

SPRT an existing net:

```sh
NET=/path/to/model.nn TAG=my_candidate tools/nnue2/run_net_sprt_pwa.sh
```

## Choosing Input Data

Use this order of trust:

| Source | Use | Notes |
| --- | --- | --- |
| Stockfish-labeled self-play | Main training signal | Best match to Enyo search distribution. |
| Stockfish depth-16 relabels | Quality upgrade | Slow, but better targets. Use 1M-3M signed-bucket rows first. |
| Lichess eval DB | Diversity | Mix in modestly, usually 10-25% of rows. |
| Binpack data | Controlled experiment only | Earlier broad binpack-heavy runs improved MAE but hurt Elo. |
| Lichess/bug hard cases | Validation/augmentation only | Do not train mostly on hard cases; it overfits. |

Good sampling:

- Keep both positive and negative scores.
- Avoid neutral-only data; include signed buckets like `0-50`, `50-100`,
  `100-300`, `300-800`, and a small `800-1600` tail.
- Use unique FENs where possible.
- Do not let high-CP positions dominate. Clamp targets.

## Choosing Training Values

Broad self-play + Lichess:

- objective: `mpe25`
- `--wdl-lambda 0.95`
- Lichess/eval DB CP-only source: `source-wdl-lambda=1.0`
- learning rate: `3e-7`
- epochs: `4`
- target clamp: `1200`
- batch size: `8192`

Depth-16 relabel runs:

- objective: `huber`
- Huber beta: `200`
- learning rate: `7e-7` or `1e-6`
- epochs: `8`
- target clamp: `800`
- batch size: `8192`

Avoid by default:

- learning rate above `3e-6`
- mostly binpack runs
- mostly hardcase/pairwise runs
- promoting a net based on MAE alone

## Static Gate

A candidate is worth SPRT only if it roughly passes:

- MAE improves on its own validation rows.
- MAE improves on existing d16/self-play validation.
- Lichess MAE does not regress.
- sign rate drop is small, preferably `<= 0.3%`.
- binpack does not show a large sign collapse.

Static metric improvement is not enough. Many Enyo runs improved MAE and were
neutral in SPRT.

## SPRT Gate

Fast triage:

```text
games=4000
tc=2+0.02
concurrency=10
threads=2
Hash=512
elo0=0
elo1=8
```

Interpretation:

- negative early: stop and archive
- `0..+5 Elo`: archive as weak/inconclusive
- `+5..+8 Elo`: rerun only if static metrics are very clean
- `+8 Elo` or SPRT H1: confirm with more games and/or slower TC

Promote only after a confirming run. A keeper should beat the current reference
clearly enough that noise is not the explanation.

## Current Lessons

- Stronger target quality helps static metrics, but has not yet produced a
  clear replacement net.
- The depth-16 expansion run produced static-positive candidates; the tested
  `lr1e-6` candidate ended around `+1.7 +/- 7.6 Elo`, so it is not a keeper.
- Hardcase-only and pairwise hardcase training moved specific positions but did
  not translate into match strength.
- Binpack-heavy training is not trusted unless isolated and proven by SPRT.

## Run Log Template

Add one line per serious run:

```text
YYYY-MM-DD | run dir | data mix | objective/lr/clamp/epochs | static take | SPRT take | decision
```
