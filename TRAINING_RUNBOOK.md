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

Objective choice:

- `mpe25`: convert predicted/target centipawns to a probability-like value,
  optionally blend in WDL/result signal, then minimize that error. Use this
  for broad self-play + Lichess training where search behavior matters more
  than exact centipawn fit.
- `huber`: train directly on centipawns, but reduce the effect of very large
  errors. Use this for clean teacher relabel runs where the target is trusted
  and extreme scores should not dominate.

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
- Small objective/LR changes on the same depth-12 pool are not enough. They
  mostly retest training noise.
- The depth-16 expansion run produced static-positive candidates; the tested
  `lr1e-6` candidate ended around `+1.7 +/- 7.6 Elo`, so it is not a keeper.
- Hardcase-only and pairwise hardcase training moved specific positions but did
  not translate into match strength.
- Binpack-heavy training is not trusted unless isolated and proven by SPRT.

## Next Improvement Plan

Goal: add new signal, not just rerun the same pool.

1. Generate `15-30M` usable fresh Enyo self-play positions from the current
   reference engine. Use current search/eval, varied openings, and enough
   randomness that the positions are not just repeats of the old pool.
2. Filter out bad training rows: duplicate FENs, timeout/emergency moves, mate
   scores, missing scores, and extreme-only buckets.
3. Sample signed score buckets so the data is not mostly neutral positions:
   `0-25`, `25-75`, `75-150`, `150-300`, `300-600`, `600-1200`, both signs
   where applicable.
4. Label the fresh positions with Stockfish:
   - main labels: depth `16`
   - premium subset: depth `18` on `2-4M` high-quality signed-bucket rows
5. Train a small matrix:
   - self-play-only `huber`, clamp `800`, beta `200`, lr `7e-7..1e-6`
   - self-play-only `huber`, clamp `1000`, beta `200`, lr `7e-7..1e-6`
   - self-play-only `mpe25`, clamp `1200`, lr `7e-7..1e-6`
   - `mpe25` with `10-15%` Lichess eval rows
   - one higher-LR probe, `1.2e-6..1.5e-6`, only if static metrics stay clean
6. Reject with held-out validation first. Include validation from the fresh
   distribution and old d16/self-play/Lichess sets. Do not use binpack or
   hardcase data as a main training source.
7. Run replay gates on known Enyo misses/blunders/time losses. Current `replay`
   output is accurate enough to use as a tactical sanity check again.
8. Test promising candidates:
   - `1000` games: smoke test
   - `4000` games: screen
   - `10000-20000` games: confirm likely `+5..+8 Elo` candidates
   - `30000+` games or slower TC: needed before treating `+3..+5 Elo` as real

If this stays neutral, the next likely bottleneck is architecture/features or
starting-net dependency, not another tiny learning-rate change.

## Run Log Template

Add one line per serious run:

```text
YYYY-MM-DD | run dir | data mix | objective/lr/clamp/epochs | static take | SPRT take | decision
```

## Vocabulary

- `mpe25 objective`: training loss that mixes centipawn accuracy with
  WDL-style chess outcome behavior. In `train.py`, scores are passed through a
  sigmoid and compared as probability-like values with exponent `2.5`.
- `huber objective`: centipawn loss that reduces the influence of very large
  eval errors. Small errors behave roughly like squared error; large errors
  become closer to linear, so one huge score miss does not dominate training.
- `learning rate`: update size for each training step. `3e-7` means
  `0.0000003`, a cautious fine-tuning value.
- `target clamp`: maximum absolute teacher score used for training, e.g.
  `1200` means clamp to `[-1200, +1200]`.
- `epoch`: one full pass over the sampled training rows.
- `MAE`: mean absolute error in centipawns. Useful filter, not proof of Elo.
- `sign`: how often the net and target agree on which side is better.
- `SPRT`: match test that decides whether a candidate is stronger than the
  reference.
