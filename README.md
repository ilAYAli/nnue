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

The default backend is the Enyo PyTorch trainer. A pairwise backend trains the
normal Enyo `.nn` format from child-position ranking pairs, for move-choice
experiments where scalar cp fitting is not enough.

An experimental Bullet backend uses the same `posgen -> score` front half, then
switches to:

```text
jsonl -> Bullet text -> BulletFormat -> Bullet trainer
```

The Bullet backend currently produces Bullet checkpoints, not the normal Enyo
`.nn` format. The experimental Enyo branch can load these `quantised.bin`
checkpoints directly for architecture experiments, but the path is not yet fast
enough for candidate SPRT.

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

Architecture experiments can reuse existing teacher labels and skip
self-play/scoring:

```sh
./build.py create \
  --name arch-kingbucket-v1 \
  --labeled-jsonl runs/imported/fresh_d12self18h64_d16_labels_20260519_113826/score/labeled.jsonl
```

This still repacks the data, so feature-index changes are reflected in the
new tensors.

Experimental Bullet/Reckless-like architecture spike:

```sh
./build.py create \
  --name bullet-reckless-spike \
  --backend bullet \
  --labeled-jsonl runs/imported/fresh_d12self18h64_d16_labels_20260519_113826/score/labeled.jsonl \
  --bullet-rows 100000 \
  --bullet-superbatches 2 \
  --event-command "$HOME/code/cpp/chess/nnue/tools/events/nnue_event_ntfy.sh"
```

This verifies Bullet conversion/training under the normal pipeline/event
machinery. It writes Bullet `quantised.bin` checkpoints. The experimental Enyo
loader can load those directly, but they are not normal Enyo `model.nn` files.

Pairwise move-choice fine-tune:

```sh
./build.py create \
  --name pairwise-sprtfail \
  --backend pairwise \
  --labeled-jsonl runs/imported/fresh_d12self18h64_d16_labels_20260519_113826/score/labeled.jsonl \
  --pairwise-scores-csv runs/bullet-sprt-failure-diagnostics-20260522/movechoice_sb112_20260522_015057/out/scores.csv \
  --pairwise-candidate-moves-csv runs/pairwise-sprtfail-qmid-w50-lr3e3-e30/validate/sprtfail_gate_20260522_030000/move_choice_gate.csv \
  --pairwise-pair-weight 0.25 \
  --epochs 4 \
  --event-command "$HOME/code/cpp/chess/nnue/tools/events/nnue_event_ntfy.sh"
```

This exports a normal Enyo `model.nn`. It should be gated on the same
move-choice positions before any SPRT.

Pairwise sparse/input movement probe:

```sh
./build.py create \
  --name pairwise-sparseprobe \
  --backend pairwise \
  --labeled-jsonl runs/imported/fresh_d12self18h64_d16_labels_20260519_113826/score/labeled.jsonl \
  --pairwise-scores-csv assets/failure_suite/pairwise_sprtfail_repeated_tail_scores_20260522.csv \
  --forward quantized \
  --input-lr-mult 100 \
  --l1-lr-mult 20 \
  --dense-lr-mult 0.2
```

Use this when a fine-tune only changes dense/head tensors after export. The
first gate is `net-diff`; if input/L1 exported tensors still do not move, do
not spend time on replay or SPRT.

To train an Enyo-owned net from scratch instead of fine-tuning the current
reference net, set `init_net` to `null` in `build.json` or pass an empty
`--init-net` value and choose an initializer:

```json
{
  "create": {
    "name": "scratch-huber-cp800",
    "init_net": null,
    "init": "kaiming",
    "labeled_jsonl": "runs/imported/fresh_d12self18h64_d16_labels_20260519_113826/score/labeled.jsonl"
  }
}
```

This is the clean path for replacing the Berserk-derived net. It will likely
start weaker and needs its own gates before it can become the reference.

Before training an architecture branch, run its feature-map sanity check:

```sh
tools/validate/king_bucket_features.py --engine ~/code/cpp/chess/enyo/build/enyo
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
  --event-command "$HOME/code/cpp/chess/nnue/tools/events/nnue_event_ntfy.sh"
```

`--event-command` must point to a script that exists on the machine running the
build command. The tracked `tools/events/nnue_event_ntfy.sh` hook sends build
events to `https://ntfy.wahlman.no/nnue` and phase-completion prompts to
`https://ntfy.wahlman.no/AI_stdin`. Override with `NNUE_NTFY_URL` and
`NNUE_AI_STDIN_URL` if needed.

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
--labeled-jsonl PATH
--backend pytorch|pairwise|bullet

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
--init kaiming|berserk-ish
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
--max-rows N
--skip-rows N
--grad-norm-every N
--patience N
--select-metric loss|mse|mae|sign
--input-lr-mult X
--l1-lr-mult X
--dense-lr-mult X
--trainable all|input|float-head|output

--pairwise-scores-csv PATH
--pairwise-pairs-jsonl PATH
--pairwise-candidate-moves-csv PATH
--pairwise-pair-batch-size N
--pairwise-pair-weight X
--pairwise-pair-beta CP
--pairwise-min-target-margin CP
--pairwise-max-target-margin CP
--pairwise-loss-weight-by-cp

--bullet-rows N
--bullet-max-abs-cp CP
--bullet-manifest PATH
--bullet-cuda-path PATH
--bullet-cuda-arch auto|native|compute_90|sm_90|...
--bullet-cargo-target-dir PATH
--bullet-hidden N
--bullet-l2 N
--bullet-batch-size N
--bullet-batches N
--bullet-superbatches N
--bullet-threads N
--bullet-wdl X
--bullet-lr X
--bullet-final-lr X
```

Defaults live in `tools/lib/defaults.py`. Phase-specific behavior is documented
in each tool subdirectory.

## Validation

Run validation explicitly after a candidate is produced:

```sh
tools/validate/validate.py net-diff \
  --candidate runs/d12-d16-huber-cp800/train/d12-d16-huber-cp800/model.nn \
  --reference ~/code/cpp/chess/enyo/nnue/berserk-d43206fe90e4.nn \
  --fail-if-identical \
  --event-command "$HOME/code/cpp/chess/nnue/tools/events/nnue_event_ntfy.sh"

tools/validate/validate.py static \
  --net runs/d12-d16-huber-cp800/train/d12-d16-huber-cp800/model.nn \
  --data runs/d12-d16-huber-cp800/pack/train \
  --rows 100000 \
  --buckets \
  --sources \
  --event-command "$HOME/code/cpp/chess/nnue/tools/events/nnue_event_ntfy.sh"

tools/validate/validate.py sprt \
  --net runs/d12-d16-huber-cp800/train/d12-d16-huber-cp800/model.nn \
  --run runs/d12-d16-huber-cp800 \
  --games 1000 \
  --tag d12_d16_smoke \
  --event-command "$HOME/code/cpp/chess/nnue/tools/events/nnue_event_ntfy.sh"
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
