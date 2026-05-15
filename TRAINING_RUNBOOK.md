# Enyo NNUE Training Runbook

Concise guide for training, validating, and promoting Enyo NNUE nets.

## Environment

Run long training work on `pwa-5090`.

```sh
cd ~/code/cpp/chess/nnue
git pull --ff-only
```

Use `/home/petter/.venv/bin/python` for NNUE Python on `pwa-5090`.

## How To Train NNUE

These are the canonical steps. A run can start later if an earlier artifact
already exists.

### Step 1: Self-Play

Generate fresh Enyo-vs-Enyo games from the current reference engine. This
creates PGN with engine score comments.

Script:

```sh
tools/nnue2/run_fresh_d16_labels_pwa.sh
```

This wrapper uses:

```sh
~/tmp/enyo-own-net-pipeline/tools/selfplay/run_selfplay.sh
```

Output:

```text
<run>/selfplay.pgn
```

Skip this step if you already have suitable self-play PGN or JSONL.

### Step 2: Row Extraction And Filtering

Convert PGN to JSONL training rows. Each row is one position with FEN, played
move, score, WDL/result, depth, and source metadata. Filter bad rows: missing
scores, mate scores, timeout/emergency rows, duplicate FENs, and extreme-only
data.

Script used by the fresh run:

```sh
~/tmp/enyo-own-net-pipeline/tools/selfplay/pgn_to_jsonl.py
```

Output:

```text
<run>/selfplay.jsonl
```

Skip this step if you already have clean position JSONL.

### Step 3: Signed Bucket Sampling

Sample a balanced score distribution so the net does not mostly see neutral
positions. Use signed buckets such as `0-25`, `25-75`, `75-150`, `150-300`,
`300-600`, and `600-1200`.

Script:

```sh
tools/nnue2/sample_signed_buckets.py
```

Output:

```text
<run>/source_signed.jsonl
```

Skip this step only if the input JSONL is already balanced.

### Step 4: Teacher Labeling

Replace shallow/self-play scores with stronger teacher scores, usually
Stockfish depth `16`. A smaller premium subset can be labeled at depth `18`.

Script:

```sh
tools/nnue2/label_with_uci.py
```

Output:

```text
<run>/labeled.jsonl
```

Skip this step if the JSONL already has trusted teacher labels.

### Step 5: Tensor Packing

Convert labeled JSONL into tensors consumed by PyTorch training.

Script:

```sh
tools/nnue2/pack_dataset.py
```

Output:

```text
<run>/packed/
```

Skip this step if you already have a packed dataset.

### Step 6: Training

Train/fine-tune a candidate `.nn` from a starting net and packed tensors.

Script:

```sh
tools/nnue2/train.py
```

Common objectives:

- `huber`: centipawn training with reduced impact from huge eval errors.
- `mpe25`: probability-like centipawn/WDL training, usually better for broad
  self-play plus Lichess mixes.

Output:

```text
<candidate>/model.pt
<candidate>/model.nn
```

### Step 7: Static Validation

Reject obviously bad candidates before spending SPRT time. Static validation is
only a filter; it does not prove Elo.

Script:

```sh
tools/nnue2/eval_dataset.py
```

A candidate is worth games only if:

- MAE improves on its own held-out validation rows.
- old self-play/d16 validation does not regress badly.
- Lichess validation does not regress badly.
- sign rate drop is small, preferably `<= 0.3%`.

### Step 8: Replay Gates

Use `replay` as a diagnostic on known Enyo miss/blunder/time-loss logs. Do not
confuse the two modes:

- logged-game analysis: what the engine actually played in the log
- candidate replay: what the candidate plays now at the logged node count

```sh
# Analyze logged moves only. This does not run the candidate engine.
replay --log --summary-only ~/code/cpp/chess/enyo/bugs

# Replay a candidate against the logged positions, then analyze its moves.
replay --engine <candidate-wrapper> --summary-only --jobs 4 ~/code/cpp/chess/enyo/bugs
```

Use `--force` after replay changes or when a cached report looks suspicious.
`--pgn` is for inspecting a single game, not normal gate output.

Treat replay gates as sanity checks, not final strength tests. A candidate is
bad if it consistently adds serious misses versus the logged/reference result,
but replay alone should not promote a net.

### Step 9: SPRT

Test candidate strength against the current reference net.

Script:

```sh
NET=/path/to/model.nn TAG=my_candidate tools/nnue2/run_net_sprt_pwa.sh
```

Default screen:

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
- `0..+5 Elo`: weak/inconclusive
- `+5..+8 Elo`: rerun only if static metrics are clean
- `+8 Elo` or SPRT H1: confirm with more games and/or slower TC
- `+3..+5 Elo`: needs `30000+` games or slower TC before treating as real

Promote only after a confirming run.

## Pipeline Shortcuts

### Fresh D16 Label Pipeline

```sh
tools/nnue2/run_fresh_d16_labels_pwa.sh
```

Performs Steps `1-5`:

- fresh self-play
- PGN to JSONL
- signed bucket sampling
- Stockfish d16 labeling
- tensor packing

It does not train or SPRT by itself. Use its `<run>/packed/` output for Step 6.

### Default One-Command Training

```sh
tools/nnue2/train_new_net_pwa.sh
```

This is a shortcut for existing data. It skips Steps `1-4` because it reuses
already labeled inputs:

- self-play teacher rows:
  `~/tmp/enyo_teacher/sf_d12_20m_20260510_115338/labeled.jsonl`
- Lichess eval rows:
  `~/tmp/enyo_teacher/lichess_eval_d18_standard/lichess_eval.jsonl`

Then it performs:

- Step 5: mix/pack rows
- Step 6: train one `mpe25` candidate
- Step 7: static validation
- Step 9: SPRT

It currently does not run Step 8 replay gates.

Default values:

- `8,000,000` self-play rows
- `2,000,000` Lichess rows
- objective `mpe25`
- learning rate `3e-7`
- target clamp `1200`
- `4` epochs
- SPRT `4000` games at `2+0.02`

Running the exact same command again mostly tests sampling/training noise.
Change the data, LR, clamp, epochs, or SPRT length for a meaningful experiment.

### D16 Relabel Pipeline

```sh
tools/nnue2/run_d16_expansion_pwa.sh
```

This starts at Step 3 using the old 20M d12 pool, then runs Steps `4-9`.
Useful as a small quality-upgrade experiment, but it is not fresh signal.

## Next Improvement Plan

Goal: add new signal, not rerun the same old d12 pool.

Current cycle:

- Step 1: generate `160k` fresh self-play games from the current reference.
- Step 2: convert to rows and filter.
- Step 3: sample signed buckets up to roughly `3M` rows.
- Step 4: label with Stockfish depth `16`.
- Step 5: pack tensors.

Next training matrix after Step 5:

- self-play-only `huber`, clamp `800`, beta `200`, lr `7e-7..1e-6`
- self-play-only `huber`, clamp `1000`, beta `200`, lr `7e-7..1e-6`
- self-play-only `mpe25`, clamp `1200`, lr `7e-7..1e-6`
- `mpe25` with `10-15%` Lichess eval rows
- one higher-LR probe, `1.2e-6..1.5e-6`, only if static metrics stay clean

If this stays neutral, the next likely bottleneck is architecture/features or
starting-net dependency, not another tiny learning-rate change.

## Data Choice

Use this order of trust:

| Source | Use | Notes |
| --- | --- | --- |
| Fresh Stockfish-labeled self-play | Main signal | Best match to current Enyo search distribution. |
| Depth-18 subset | Premium signal | Expensive; use on smaller signed-bucket subset. |
| Lichess eval DB | Diversity | Mix modestly, usually `10-25%`. |
| Binpack data | Controlled experiment only | Earlier broad binpack-heavy runs improved MAE but hurt Elo. |
| Lichess/bug hard cases | Validation/augmentation only | Do not train mostly on hard cases; it overfits. |

## Training Values

Broad self-play + Lichess:

- objective: `mpe25`
- `--wdl-lambda 0.95`
- Lichess/eval DB CP-only source: `source-wdl-lambda=1.0`
- learning rate: `3e-7..1e-6`
- epochs: `4`
- target clamp: `1200`
- batch size: `8192`

Clean d16/d18 teacher labels:

- objective: `huber`
- Huber beta: `200`
- learning rate: `7e-7..1e-6`
- epochs: `4-8`
- target clamp: `800` or `1000`
- batch size: `8192`

Avoid by default:

- learning rate above `3e-6`
- mostly binpack runs
- mostly hardcase/pairwise runs
- promoting a net based on MAE alone

## Current Lessons

- Stronger labels improve static metrics, but old-pool d16 relabeling has not
  produced a clear replacement net.
- Small objective/LR changes on the same d12 pool mostly retest noise.
- The old d16 expansion `lr1e-6` candidate ended around `+1.7 +/- 7.6 Elo`, so
  it is not a keeper.
- Hardcase-only and pairwise hardcase training moved specific positions but did
  not translate into match strength.
- Binpack-heavy training is not trusted unless isolated and proven by SPRT.

## Run Log Template

Add one line per serious run:

```text
YYYY-MM-DD | run dir | steps run | data mix | objective/lr/clamp/epochs | static take | SPRT take | decision
```

## Vocabulary

- `JSONL`: one JSON object per line. Used for position rows.
- `packed tensors`: binary tensor dataset created from JSONL for fast PyTorch
  loading.
- `mpe25 objective`: training loss that mixes centipawn accuracy with WDL-style
  behavior. In `train.py`, scores pass through a sigmoid and are compared as
  probability-like values with exponent `2.5`.
- `huber objective`: centipawn loss that reduces the influence of very large
  eval errors. Small errors behave roughly like squared error; large errors are
  closer to linear.
- `learning rate`: update size for each training step. `3e-7` means
  `0.0000003`, a cautious fine-tuning value.
- `target clamp`: maximum absolute teacher score used for training, e.g.
  `1200` means clamp to `[-1200, +1200]`.
- `epoch`: one full pass over the sampled training rows.
- `MAE`: mean absolute error in centipawns. Useful filter, not proof of Elo.
- `sign`: how often the net and target agree on which side is better.
- `SPRT`: match test that decides whether a candidate is stronger than the
  reference.
