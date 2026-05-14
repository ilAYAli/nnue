# Enyo NNUE Training Runbook

This is the short operational guide for starting a new Enyo NNUE run without
Codex.

## Quick Start

Run this on `pwa-5090`:

```sh
cd ~/code/cpp/chess/nnue
git pull --ff-only
tools/nnue2/train_new_net_pwa.sh
```

The launcher starts the job in tmux session `nnue_test`, writes all large
outputs under `~/tmp/enyo_teacher/`, and calls `~/scripts/notifai.sh` when
major phases complete.

Monitor it with:

```sh
tmux attach -t nnue_test
```

or:

```sh
tail -f ~/tmp/enyo_teacher/<run-dir>/run.log
```

## What The Default Run Does

The default launcher uses the current conservative recipe:

```text
positions -> packed tensors -> PyTorch training -> .nn export -> static eval -> SPRT
```

It uses:

- `8,000,000` Stockfish-labeled self-play rows from
  `~/tmp/enyo_teacher/sf_d12_20m_20260510_115338/labeled.jsonl`
- `2,000,000` Lichess eval-DB rows from
  `~/tmp/enyo_teacher/lichess_eval_d18_standard/lichess_eval.jsonl`
- no binpack rows by default
- MPE25 loss with source-aware WDL blending
- a low learning rate, starting from the current compatible `.nn`
- SPRT versus the current reference engine using the candidate net as
  `nnue2_file`

Binpack data is not used by default because earlier experiments improved static
MAE while hurting Elo. Add it only as a separate controlled experiment.

## Common Commands

Larger run:

```sh
tools/nnue2/train_new_net_pwa.sh \
  --selfplay-rows 12000000 \
  --lichess-rows 3000000
```

Dry run, to see exactly what would be launched:

```sh
tools/nnue2/train_new_net_pwa.sh --dry-run
```

Run in the current shell instead of tmux:

```sh
tools/nnue2/train_new_net_pwa.sh --foreground
```

Change SPRT length:

```sh
tools/nnue2/train_new_net_pwa.sh --sprt-games 8000
```

## Important Options

| Option | Default | Meaning |
| --- | ---: | --- |
| `--selfplay-rows` | `8000000` | Number of self-play teacher rows to sample. |
| `--lichess-rows` | `2000000` | Number of Lichess eval rows to sample. |
| `--epochs` | `4` | Full passes over the sampled packed data. |
| `--lr` | `3e-7` | AdamW learning rate. Keep this low unless testing deliberately. |
| `--target-clamp` | `1200` | Clamp target centipawns during training. |
| `--sprt-games` | `4000` | Maximum games for the validation match. |
| `--sprt-tc` | `2+0.02` | Fast triage time control. Confirm keepers later at slower TC. |
| `--sprt-elo1` | `8` | SPRT alternative hypothesis. |

The wrapper refuses to start if another NNUE training/SPRT process appears
active. Use `--allow-active` only when you intentionally want overlap.

## Output Layout

Each run has a directory like:

```text
~/tmp/enyo_teacher/selflichess_mix_YYYYMMDD_HHMMSS/
```

Important files:

- `run.log`: full pipeline output
- `source_*.jsonl`: mixed source rows
- `packed/`: NumPy arrays consumed by PyTorch
- `<tag>/model.nn`: candidate net for Enyo
- `<tag>/model.pt`: PyTorch checkpoint
- `<tag>/static_summary.txt`: baseline vs candidate static metrics
- `<tag>/sprt/sprt.log`: match result versus the reference net

## Reading Results

Static metrics are only a sanity check. A candidate can improve MAE and still
lose Elo. Treat SPRT as the real gate.

Rough promotion rules:

- clearly negative early: stop and archive the run
- `0..+5 Elo`: interesting but not a reference by itself
- `+8..+15 Elo` with good LOS: rerun or extend with more games
- H1/pass: promote only after confirming there is no time-management or
  obvious replay regression

## Current Lessons

- Target scores are already side-to-move centipawns for self-play, Lichess
  eval, and the current binpack importer.
- Plain static loss, MAE, or sign rate is not enough to select a net.
- Broad binpack-heavy training has repeatedly improved static loss while
  hurting Elo.
- Hard-case pairwise training can overfit replay misses without improving
  match strength.
- Keep experiments small enough to reject quickly, then spend long SPRT time
  only on candidates already showing a positive trend.

## Manual Building Blocks

The wrapper delegates to:

```sh
tools/nnue2/run_selflichess_mix_pwa.sh
```

Lower-level scripts:

- `tools/nnue2/mix_jsonl.py`: stream-sample and interleave JSONL sources
- `tools/nnue2/pack_dataset.py`: convert JSONL to packed NumPy tensors
- `tools/nnue2/train.py`: PyTorch training and `.nn` export
- `tools/nnue2/eval_dataset.py`: static baseline/candidate metrics
- `~/code/cpp/chess/sprt/sprt`: fastchess SPRT wrapper

Use `/home/petter/.venv/bin/python` for NNUE Python on `pwa-5090`.
