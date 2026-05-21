# Enyo NNUE

This repo contains the NNUE training tools and experiment notes for Enyo.
Engine binaries, source checkouts, books, and reference nets are configured via
command-line arguments and defaults in `tools/lib/defaults.py`.

The high-level build command is:

```sh
./build.py
```

It creates a candidate net by running:

```text
posgen -> score -> pack -> train
```

Validation is separate and lives under `tools/validate/`.

## Build A Net

Current reviewed improvement run:

```sh
./build.py -c build.json
```

`build.json` is the committed active candidate recipe. Update it when the next
experiment changes, so the intended run is visible in one diff. Command-line
arguments override values from the file:

```sh
./build.py -c build.json --name local-smoke --selfplay-games 1000 --device cpu
```

Re-running the same `-c build.json` command resumes the same run by
skipping phases with existing `.done` markers. Use `--force` to rerun phases.
Generated run configs can also be relaunched the same way:

```sh
./build.py -c runs/d12-d16-huber-cp800/config.json
```

Small smoke run:

```sh
./build.py create \
  --name smoke-d8-d12 \
  --selfplay-games 1000 \
  --score-depth 12 \
  --score-shards 4 \
  --epochs 1 \
  --device cpu
```

Depth-12 self-play candidate with event notifications:

```sh
./build.py create \
  --name d12-d16-huber-cp800 \
  --selfplay-depth 12 \
  --selfplay-games 120000 \
  --event-command "$HOME/scripts/nnue_event_ntfy.sh"
```

`--event-command` must point to a script that exists on the machine running the
build command. The repo emits generic JSON events; ntfy and personal
routing stay outside the repo.

Dry-run the generated pipeline without starting work:

```sh
./build.py create \
  --name inspect-config \
  --selfplay-depth 12 \
  --dry-run
```

Inspect a run:

```sh
./build.py status runs/d12-d16-huber-cp800
./build.py status runs/d12-d16-huber-cp800 --tail 20
./build.py report runs/d12-d16-huber-cp800
```

## Common Candidate Arguments

```text
--name NAME
--run-dir DIR
-c, --config FILE
--dry-run
--force
--event-command COMMAND

--engine PATH
--nnue-file PATH
--book PATH
--runner PATH
--python PATH

--selfplay-games N
--selfplay-shard-games N
--selfplay-concurrency N
--selfplay-threads N
--selfplay-hash MB
--selfplay-depth N
--selfplay-seed N

--skip-plies N
--source-max-abs-cp CP
--sample-preset NAME

--score-engine PATH
--score-depth N
--score-shards N
--score-threads N
--score-hash MB
--score-max-abs-cp CP

--init-net PATH
--objective mse|huber|mpe25
--target-clamp CP
--huber-beta CP
--wdl-lambda X
--lr X
--epochs N
--batch-size N
--device cpu|cuda
--workers N
--val-rows N
--patience N
--select-metric loss|mse|mae|sign
--trainable all|float-head|output
```

Defaults live in `tools/lib/defaults.py`. Phase-specific behavior is documented
in each tool subdirectory.

## Validation

Run validation explicitly after a candidate is produced:

```sh
tools/validate/validate.py static \
  --net runs/d12-d16-huber-cp800/train/d12-d16-huber-cp800/model.nn \
  --data runs/d12-d16-huber-cp800/pack/train \
  --rows 100000 \
  --buckets \
  --sources \
  --event-command "$HOME/scripts/nnue_event_ntfy.sh"

tools/validate/validate.py sprt \
  --net runs/d12-d16-huber-cp800/train/d12-d16-huber-cp800/model.nn \
  --run runs/d12-d16-huber-cp800 \
  --games 1000 \
  --tag d12_d16_smoke \
  --event-command "$HOME/scripts/nnue_event_ntfy.sh"
```

Architecture sanity checks used before expensive training:

```sh
~/.venv/bin/python tools/validate/material_phase.py
~/.venv/bin/python tools/validate/roundtrip.py ../enyo/nnue/berserk-d43206fe90e4.nn
```

## Important Docs

```text
NNUE.md              NNUE architecture, weights, accumulators, and training concepts
IMPROVEMENT_PLAN.md  Current experiment plan and lessons from failed runs
tools/README.md      Lower-level phase tool overview
```

## Run Data

Run data is stored under:

```text
runs/<run-name>/
```

Expected layout:

```text
config.json
manifest.json
status.json
events.jsonl
logs/
assets/
posgen/
score/
pack/
train/
validate/
```
