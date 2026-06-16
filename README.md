# Enyo NNUE

This repo contains the NNUE training tools and experiment notes for Enyo.

The normal workflow is intentionally small:

```text
edit build.json -> ./nnue-run doctor -c build.json -> ./nnue-run start -c build.json
```

`build.json` is the experiment contract. It must show:

```text
hypothesis          what this run is testing
changed_variables   the knobs changed from the previous run
stages              the ordered commands that will run
gates               the pass/fail checks
```

Do not launch from memory. If a variable matters, put it in `build.json`.

## Commands

```sh
./nnue-run plan -c build.json
./nnue-run doctor -c build.json
./nnue-run start -c build.json
./nnue-run status
./nnue-run resume
./nnue-run stop
```

`doctor` is mandatory before training. It rejects dirty source trees, untracked
scripts, unknown config fields, missing compiled hot-path tools, and configs
that still call Python tools with compiled replacements.

Status output is the user-facing contract:

```text
Run: native-next
State: running
Stage: 2/3 static_gate
Hypothesis: ...
Changed variables:
  lr: 1e-10
  objective: output-only
Log: runs/native-next/logs/02-static_gate.log
```

## Fast Data Tools

Python is only for orchestration, status, and small tests. Hot-path data work
over large row sets must use compiled tools.

Build the compiled tools:

```sh
cmake -S . -B build/fast -G Ninja
cmake --build build/fast
```

Current compiled tools:

```text
nnue-mix-jsonl              streaming JSONL dataset mixer
nnue-jsonl-to-bullet-text   JSONL to Bullet text converter
```

If another Python script becomes part of the normal high-volume chain, replace
it with C++ first or document why it is not hot-path work.

## Candidate Workflow

The expected candidate chain is:

```text
position source -> teacher labeling -> packing -> training/export -> static gates -> smoke/SPRT
```

Short meaning:

```text
position source   Enyo self-play, oracle positions, or curated held-out rows
teacher labeling  score selected positions with the configured oracle
packing           convert labeled rows into Bullet/tensor training input
training/export   train weights and export model.nn
static gates      check exported net behavior on held-out labeled rows
smoke/SPRT        measure actual playing strength
```

Do not promote a net from static MAE alone. Static validation is a rejection
filter. A candidate needs game strength: at minimum a neutral-positive smoke,
and usually a longer SPRT against the current reference net.

## Important Docs

```text
AGENTS.md           Workflow, git, long-run, and hygiene rules
NNUE.md             Runtime layout, feature rows, accumulators, evaluation
IMPROVEMENT_PLAN.md Current experiment plan and durable conclusions
tools/README.md     Lower-level tool overview
```

## Run Data

Run data belongs under:

```text
runs/<run-name>/
```

Do not commit generated run data, temporary configs, caches, logs, exported
nets, or local validation outputs unless explicitly requested.
