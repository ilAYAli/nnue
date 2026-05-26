# Enyo NNUE DIY

Goal: run one NNUE experiment without guessing.

Normal workflow:

```text
edit build.json -> ./build.py -c build.json -> gates -> decision
```

## Where To Run

```sh
ssh pwa-5090
cd ~/code/cpp/chess/nnue_native_hidden
tmux new -As nnue_native
```

Use notifications for long runs:

```sh
EVENT=~/code/cpp/chess/nnue_native_hidden/tools/events/nnue_event_ntfy.sh
```

## What To Change

For normal experiments, change `build.json`.

Do not edit trainer/exporter/validator/engine code unless you are fixing
tooling. Do not launch from memory; make the intended recipe visible in
`build.json`.

Minimum fields to review:

```json
{
  "create": {
    "name": "native-short-name-YYYYMMDD",
    "run_dir": "runs/native-short-name-YYYYMMDD",
    "backend": "search-aware",
    "labeled_jsonl": "/home/petter/code/cpp/chess/nnue/runs/imported/fresh_d12self18h64_d16_labels_20260519_113826/score/labeled.jsonl",
    "init_net": "/path/to/base/model.nn",
    "search_broad_target_net": "/path/to/base/model.nn",
    "search_targets_jsonl": "runs/<target-set>/targets.jsonl"
  }
}
```

Target-set knobs:

```json
{
  "search_targets_jsonl": "runs/<target-set>/targets.jsonl",
  "search_select_primary_tags": "diagnostic:<hard-tag>",
  "search_select_guard_tags": "diagnostic:<preserve-tag>",
  "search_preserve_tags": "diagnostic:<preserve-tag>",
  "search_tag_weights": "diagnostic:<hard-tag>=1,diagnostic:<preserve-tag>=0"
}
```

Training-pressure knobs:

```json
{
  "lr": 0.0000007,
  "epochs": 120,
  "trainable": "all",
  "search_policy_weight": 1.6,
  "search_rank_weight": 3.2,
  "search_margin_weight": 0.22,
  "search_preserve_weight": 8.0,
  "search_preserve_rank_weight": 8.0
}
```

Use `trainable: "float-head"` only for quick probes. Use `trainable: "all"` for
serious candidates.

## What To Try

If gates improve but SPRT loses:

```text
change target set, not LR
```

If hard rows do not move:

```text
raise search_rank_weight or search_policy_weight slightly
```

If preserve/broad rows drift:

```text
raise search_preserve_weight or search_preserve_rank_weight
```

If hard rows learn only when preserve rows collapse:

```text
reject this target/loss shape
```

Target sets should contain both:

```text
hard rows: Enyo gets this wrong or nearly wrong
preserve rows: Enyo currently gets this right; do not regress
```

Useful categories:

```text
forcing win
defensive resource
queen/rook endgame
pawn race
conversion
broad quiet preserve
```

## Build A Target Set

Sample positions:

```sh
~/.venv/bin/python tools/validate/sample_game_positions.py \
  --out-dir runs/<target-set> \
  --per-category 40 \
  --min-ply 20 \
  ~/code/cpp/chess/enyo/bugs/win/*.log \
  ~/code/cpp/chess/enyo/bugs/draw/*.log \
  ~/code/cpp/chess/enyo/bugs/loss/*.log
```

Score moves:

```sh
~/.venv/bin/python tools/validate/score_tail_targets.py \
  --positions-jsonl runs/<target-set>/positions.jsonl \
  --out runs/<target-set>/scores.csv \
  --engine ~/local/bin/stockfish \
  --nodes 300000 \
  --threads 1 \
  --hash 512 \
  --syzygy-path ~/code/cpp/chess/assets/tablebases
```

Create targets:

```sh
~/.venv/bin/python tools/validate/build_search_targets.py \
  --scores runs/<target-set>/scores.csv \
  --output runs/<target-set>/targets.jsonl \
  --summary runs/<target-set>/summary.txt \
  --dedupe-fen \
  --max-moves 8
```

If a gate reports missing selected moves:

```sh
~/.venv/bin/python tools/validate/augment_search_targets_moves.py \
  --targets runs/<target-set>/targets.jsonl \
  --candidate-csv runs/<run>/validate/<gate>/candidate.csv \
  --reference-csv runs/<run>/validate/<gate>/reference.csv \
  --output runs/<target-set>/targets_augmented.jsonl \
  --summary runs/<target-set>/augment_summary.txt \
  --oracle-engine ~/local/bin/stockfish \
  --nodes 300000 \
  --threads 1 \
  --hash 512 \
  --syzygy-path ~/code/cpp/chess/assets/tablebases
```

Then point `build.json` at `targets_augmented.jsonl`.

## Train

Dry run:

```sh
./build.py -c build.json --dry-run
```

Run:

```sh
./build.py -c build.json --event-command "$EVENT"
```

Status:

```sh
./build.py status runs/<run-name> --tail 80
./build.py report runs/<run-name> --tail 120
```

Net:

```text
runs/<run-name>/train/<run-name>/model.nn
```

## Verify

Set paths:

```sh
RUN=runs/<run-name>
NET=$RUN/train/<run-name>/model.nn
REF=/path/to/base/model.nn
TARGETS=/path/to/targets.jsonl
ENGINE=~/code/cpp/chess/assets/engines/reference
```

### 1. Model Gate

```sh
~/.venv/bin/python tools/validate/search_target_model_gate.py \
  --targets "$TARGETS" \
  --net "$NET" \
  --forward quantized \
  --search-score-mode child-low \
  --out-csv "$RUN/validate/model_gate.csv"
```

Interpret:

```text
top1 improves, guard stable -> run engine gate
top1 improves, guard collapses -> reject or strengthen preserve
top1 does not improve -> reject
```

### 2. Engine Target Gate

```sh
~/.venv/bin/python tools/validate/search_target_gate.py \
  --targets "$TARGETS" \
  --engine "$ENGINE" \
  --candidate-net "$NET" \
  --reference-net "$REF" \
  --out-dir "$RUN/validate/search_300k" \
  --nodes 300000 \
  --threads 1 \
  --hash 512 \
  --cap 800 \
  --require-native-net-load
```

Hard requirements:

```text
missing_move = 0
wrong_selected_nonpositive = 0
hard gate does not regress
preserve/broad gate does not collapse
worst_regression_cp > -400
target worst_regression_cp > -300
```

If `missing_move > 0`, augment targets and rerun. Do not interpret the gate.

### 3. Bounded Replay

```sh
~/.venv/bin/python tools/validate/run_failure_suite_net.py \
  --candidate-net "$NET" \
  --reference-net "$REF" \
  --engine "$ENGINE" \
  --run "$RUN" \
  --tag full14-300k \
  --bugs-dir ~/code/cpp/chess/enyo/bugs \
  --count 14 \
  --fixed-nodes 300000 \
  --threads 1 \
  --jobs 1 \
  --event-command "$EVENT"
```

Pass before SPRT:

```text
candidate_better > reference_better
sum_diff_cp > +1000
median_nonzero_diff_cp >= 0
worst_regression_cp > -400
```

If one row fails only with full replay history, do not train on it. Treat it as
search/repetition work.

### 4. SPRT Smoke

Only after all gates above pass:

```sh
~/.venv/bin/python tools/validate/validate.py sprt \
  --net "$NET" \
  --run "$RUN" \
  --games 1000 \
  --tag <run-name>-smoke1k \
  --event-command "$EVENT"
```

Interpret:

```text
negative Elo or LOS < 40% -> reject
near zero with wide CI -> not promotable
positive, no tail risks -> longer confirmation
```

## Decision Table

```text
model gate fails
  reject

model gate passes, engine gate fails
  check export/engine parity or search interaction

missing_move > 0
  augment targets and rerun

hard improves, preserve collapses
  increase preserve or rebuild target set

preserve holds, hard does not move
  increase target pressure slightly

replay positive, SPRT negative
  target distribution is too narrow

same family fails twice
  stop sweeping; update IMPROVEMENT_PLAN.md
```

## Update The Plan

After a meaningful result, add only this to `IMPROVEMENT_PLAN.md`:

```text
run name
build.json change
model gate result
engine gate result
bounded replay result
SPRT result, if run
decision
```
