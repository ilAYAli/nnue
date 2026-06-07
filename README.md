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

## Candidate Workflow

The normal candidate pipeline is:

```text
Step 1  self-play
Step 2  row extraction and filtering
Step 3  signed bucket sampling
Step 4  teacher labeling
Step 5  tensor packing
Step 6  training and .nn export
Step 7  static validation
Step 8  replay/failure gates
Step 9  SPRT
```

Short meaning:

```text
self-play          generate Enyo-vs-Enyo games from the configured engine/net
extraction         convert PGN into filtered JSONL position rows
sampling           balance useful score buckets before expensive labeling
teacher labeling   score selected positions with the configured oracle
packing            convert JSONL/FEN rows into numeric training tensors or Bullet data
training           train weights and export model.nn
static validation  check exported net behavior on held-out labeled rows
replay gates       reject known tactical/search regressions before games
SPRT               measure actual playing strength
```

Do not promote a net from static MAE alone. Static validation is a rejection
filter. A release candidate needs game strength: at minimum a clean smoke, and
usually a longer SPRT against the current reference net.

## Build A Net

Use the lifecycle command first:

```sh
./build.py --help
./build.py start recipes/static-bullet.example.json
./build.py status
watch ./build.py status
./build.py status --verbose
```

Stop and continue work:

```sh
./build.py stop
./build.py resume
./build.py retry
```

`start` reads a recipe, materializes `runs/<name>/config.json`, and launches the
pipeline. `status` without a run name shows active or incomplete runs. `resume`
continues a stopped or incomplete run by skipping phases with `.done` markers.
`retry` relaunches the failed run when there is exactly one failed run.

The simple recipe path is:

```text
source positions -> Stockfish-static labels -> .bullet data -> Bullet training -> static gates
```

Minimal recipe:

```json
{
  "name": "native-static-smoke",
  "desc": "Stockfish-static labels written directly to Bullet training data",
  "source": "runs/source/source.jsonl",
  "stockfish_net": "nets/stockfish.nnue",
  "init_net": "runs/parent/train/parent/model.nn",
  "workers": "workers.json",
  "score": {
    "shards": 8,
    "jobs": 4,
    "limit": 200000
  },
  "train": {
    "superbatches": 64,
    "batch_size": 512,
    "lr": 3e-7
  }
}
```

Dry-run the generated pipeline without starting work:

```sh
./build.py start recipes/static-bullet.example.json --dry-run
```

The event hook defaults to `tools/events/nnue_event_ntfy.sh` when it exists.
The repo emits generic JSON events; notification routing stays outside the repo.

Low-level `create` and `report` commands still exist for old configs and
diagnostics, but they are not the normal interface.

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

Useful static metrics:

```text
MAE    average absolute centipawn error
MSE    squared error, sensitive to big misses
sign   how often net and target agree which side is better
bias   average prediction minus target
slope  eval calibration; low means compressed, high means exaggerated
corr   whether prediction ordering tracks target ordering
```

SPRT compares the same Enyo binary with different `nnue_file` values. That
isolates the net change from engine-code changes.

## Important Docs

```text
NNUE.md              NNUE architecture, weights, feature rows, accumulators, and evaluation
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
