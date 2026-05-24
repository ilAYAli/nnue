# Enyo NNUE

This repo contains the NNUE training tools and experiment records for Enyo.

The normal interface is `build.py`. The tools under `tools/` are phase helpers
used by `build.py`; they are not the normal way to create a candidate.

## Create A Candidate

`build.json` is the committed source of truth for the current reviewed run.
Change `build.json`, commit it, then launch:

```sh
./build.py -c build.json
```

Use the tracked event hook when phase notifications are needed:

```sh
./build.py -c build.json \
  --event-command ./tools/events/nnue_event_ntfy.sh
```

Re-running the same command resumes the same run by skipping phases with
existing `.done` markers. Use `--force` only when intentionally rerunning
phases.

For `create` configs, inspect the active run:

```sh
./build.py status
./build.py report
```

For target configs or older runs, pass the run path explicitly:

```sh
./build.py status runs/<run-name>
./build.py report runs/<run-name>
```

## Minimal build.json

```json
{
  "description": "short reason for this run",
  "create": {
    "name": "candidate-name",
    "backend": "pytorch",
    "labeled_jsonl": "runs/imported/fresh_d12self18h64_d16_labels_20260519_113826/score/labeled.jsonl",
    "init_net": "../enyo/nnue/current-reference.nn",
    "objective": "huber",
    "huber_beta": 200,
    "target_clamp": 800,
    "lr": 7e-7,
    "epochs": 8,
    "device": "cpu"
  }
}
```

For an Enyo-owned scratch/native bulk run, prefer direct binpack input. Use
`init_net: null`; Bullet initializes the native Enyo weights from its configured
standard deviations.

```json
{
  "create": {
    "name": "native-scratch",
    "backend": "bullet",
    "bullet_mode": "enyo",
    "bullet_data": "/path/to/training.binpack",
    "bullet_loader": "sfbinpack",
    "bullet_sfbinpack_min_ply": 16,
    "bullet_sfbinpack_quiet_only": true,
    "bullet_enyo_input_buckets": 32,
    "init_net": null,
    "bullet_enyo_l0_std": 8.0,
    "bullet_enyo_l1_std": 1.0
  }
}
```

## Backends

```text
pytorch       normal Enyo .nn trainer
bullet        Bullet trainer; use bullet_mode=enyo for native Enyo .nn export
pairwise      move-choice/ranking fine-tune
search-aware  scalar loss plus search-target move-margin/policy losses
material-head bucketed/material head experiments
```

The Bullet backend is optional. It uses a Cargo git dependency pinned by
`tools/bullet/spike_trainer/Cargo.lock`, so it is not tracked as a git
submodule. Backend-specific details live in `tools/bullet/README.md`.

For `bullet_mode=enyo`, `bullet_enyo_input_buckets=32` matches the current Enyo
runtime layout. `16` is legacy compatibility and must be chosen explicitly.

Large training runs should prefer direct binpack input through
`bullet_loader=sfbinpack`. JSONL is still supported for small generated sets,
target construction, and older workflows, but it is being phased out as the
default bulk-training format.

Candidate creation usually runs:

```text
posgen -> score -> pack -> train
```

If `labeled_jsonl` is set, `posgen` and `score` are skipped and the existing
labels are repacked for the selected feature layout.

## Target Files

Search-aware target construction also goes through `build.py`.

When `build.json` contains `target_build`, this builds a target JSONL from
existing child-move scores:

```sh
./build.py -c build.json
```

When `build.json` contains `target_score`, this first scores legal child moves
and then builds a target JSONL:

```sh
./build.py -c build.json
```

Do not launch training from a new target set until the target distribution has
been inspected.

## Validation

Do not go straight to SPRT. A candidate should normally pass:

```text
net-diff
static validation
search/move-choice gates
failure-suite replay
NPS check for architecture/runtime changes
```

Only run SPRT when the cheap gates are clean.

## Important Files

```text
build.py              public build/status/report interface
build.json            committed active run config
NNUE.md               concise native Enyo NNUE design
NOTES.md              vocabulary and longer personal explanations
IMPROVEMENT_PLAN.md   current experiment plan and durable conclusions
tools/README.md       lower-level helper overview
runs/<run-name>/      generated run data
```

## Useful Help

```sh
./build.py create --help
./build.py target-build --help
./build.py target-score --help
./build.py status --help
```
