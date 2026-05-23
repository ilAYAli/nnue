# Enyo NNUE

This repo contains the NNUE training tools and experiment notes for Enyo.
Engine binaries, source checkouts, books, and reference nets are configured via
command-line arguments and defaults in `tools/lib/defaults.py`.

## Public Workflow

The public workflow is intentionally small:

```sh
./build.py -c build.json
./build.py status
./build.py report
```

`build.json` is the committed source of truth for the current reviewed
experiment. Change that file, commit it, then run the command above. The many
files under `tools/` are phase helpers and library modules used by `build.py`;
they are not the normal interface for creating a candidate.

When `build.json` contains `target_build` instead of `create`, the same command
builds the configured search-aware target files:

```sh
./build.py -c build.json
```

When `build.json` contains `target_score`, it first scores legal child moves
for the configured target positions, then builds a search-aware target JSONL:

```sh
./build.py -c build.json
```

`status` and `report` use `build.json` by default. Pass an explicit run path
only when inspecting an older run:

```sh
./build.py status runs/d12-d16-huber-cp800
./build.py report runs/d12-d16-huber-cp800
```

Candidate creation runs:

```text
posgen -> score -> pack -> train
```

The default backend is the Enyo PyTorch trainer. A pairwise backend trains the
normal Enyo `.nn` format from child-position ranking pairs, for move-choice
experiments where scalar cp fitting is not enough. A search-aware backend keeps
the broad scalar loss but adds multi-move margin and soft-policy losses from
search-aware target JSONL.

An experimental Bullet backend uses the same `posgen -> score` front half, then
switches to:

```text
jsonl -> Bullet text -> BulletFormat -> Bullet trainer
```

The Bullet backend currently produces Bullet checkpoints, not the normal Enyo
`.nn` format when `--bullet-mode reckless` is used. With `--bullet-mode enyo`,
it exports a normal Enyo `model.nn` after training. Treat both Bullet modes as
experimental until static and move-choice gates pass.

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

Experimental Bullet-trained Enyo-format smoke:

```sh
./build.py create \
  --name bullet-enyo-format-smoke \
  --backend bullet \
  --bullet-mode enyo \
  --labeled-jsonl runs/imported/fresh_d12self18h64_d16_labels_20260519_113826/score/labeled.jsonl \
  --bullet-rows 100000 \
  --bullet-superbatches 2 \
  --event-command "$HOME/code/cpp/chess/nnue/tools/events/nnue_event_ntfy.sh"
```

This uses Bullet for training but writes a normal Enyo `model.nn` in the
current reference-compatible 16-king-bucket layout. Treat it as a
trainer/export feasibility smoke first; validate it before any replay/SPRT.

Experimental Bullet training directly from Stockfish NNUE binpack:

```sh
./build.py create \
  --name bullet-sfbinpack-legacy-init-parity \
  --backend bullet \
  --bullet-mode enyo \
  --bullet-loader sfbinpack \
  --bullet-data /home/petter/code/cpp/chess/assets/test79-may2022-16tb7p-filter-v6-dd.min-mar2023.unmin.high-simple-eval-1k.min-v2.binpack \
  --init-net /home/petter/code/cpp/chess/enyo/nnue/berserk-d43206fe90e4.nn \
  --bullet-export-init-only \
  --bullet-batch-size 16384 \
  --bullet-superbatches 2 \
  --bullet-lr 1e-6 \
  --bullet-final-lr 2e-7 \
  --event-command "$HOME/code/cpp/chess/nnue/tools/events/nnue_event_ntfy.sh"
```

This skips Enyo self-play, scoring, and JSONL-to-Bullet conversion. Use it when
the intended experiment is direct training from a prepared SF binpack while
preserving the current reference-compatible Enyo export layout. Keep
`--bullet-export-init-only` enabled until the exported init net is proven
search-equivalent to the current reference.

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

Search-aware target fine-tune:

```sh
./build.py create \
  --name search-aware-smoke \
  --backend search-aware \
  --labeled-jsonl runs/imported/fresh_d12self18h64_d16_labels_20260519_113826/score/labeled.jsonl \
  --search-targets-jsonl runs/search-aware-targets/search_aware_targets.jsonl \
  --forward quantized \
  --search-margin-weight 1.0 \
  --search-policy-weight 0.25 \
  --event-command "$HOME/code/cpp/chess/nnue/tools/events/nnue_event_ntfy.sh"
```

Build target sets through committed `build.json` target-build configs rather
than direct helper calls. Start with a small committed `build.json` smoke and
require `net-diff`, static validation, search-gate, and failure-suite gates
before any SPRT.

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
./build.py status
./build.py status --tail 20
./build.py report
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
