# validate

`validate` runs checks on candidate nets or engines.

It includes cheap pre-SPRT checks and SPRT itself. This keeps validation under
one abstraction instead of splitting "gates" and Elo testing.

## Commands

```sh
tools/validate/validate.py static --help
tools/validate/validate.py failure-suite --help
tools/validate/validate.py move-gate --help
tools/validate/validate.py sprt --help
tools/validate/run_forge_sprt.py --help
```

Examples:

```sh
tools/validate/validate.py static \
  --net run/candidate/model.nn \
  --data run/packed \
  --rows 100000 \
  --buckets \
  --event-command "$HOME/scripts/nnue_event_ntfy.sh"

tools/validate/validate.py failure-suite \
  --candidate ~/code/cpp/chess/assets/engines/candidate \
  --reference ~/code/cpp/chess/assets/engines/reference \
  --output-dir run/failure-suite \
  --event-command "$HOME/scripts/nnue_event_ntfy.sh" \
  ~/code/cpp/chess/enyo/bugs/*.log

$PYTHON tools/validate/build_fixed_move_gate.py \
  --child-targets lc0=targets/lc0-oracle-1k-n50k-20260528/lc0_oracle_child_targets.jsonl \
  --child-targets loss=targets/replay-loss-latest-fast4-20260529/loss_replay_child_targets.jsonl \
  --output runs/fixed-move-gate/cases.jsonl \
  --summary runs/fixed-move-gate/summary.txt \
  --min-gap-cp 30 \
  --max-per-parent 1 \
  --max-per-source 200

$PYTHON tools/validate/validate.py move-gate \
  --cases runs/fixed-move-gate/cases.jsonl \
  --engine ~/code/cpp/chess/assets/engines/reference \
  --reference-net ~/code/cpp/chess/enyo/net/berserk-d43206fe90e4.nn \
  --candidate-net run/candidate/model.nn \
  --fail-if-candidate-below-baseline \
  --fail-if-regressed-above 0

tools/validate/validate.py sprt \
  --net run/candidate/model.nn \
  --games 1000 \
  --tag candidate_smoke \
  --event-command "$HOME/scripts/nnue_event_ntfy.sh"

tools/validate/run_forge_sprt.py \
  --net run/candidate/model.nn \
  --reference-net reference/native-1.5.0.nn \
  --tag candidate-vs-native15-confirm4000 \
  --games 4000 \
  --chunk-games 100 \
  --workers workers.json
```

With the generic event hook:

```sh
tools/validate/validate.py sprt \
  --net runs/d12-d16-huber-cp800/train/d12-d16-huber-cp800/model.nn \
  --run runs/d12-d16-huber-cp800 \
  --games 1000 \
  --tag d12_d16_smoke \
  --event-command "$HOME/scripts/nnue_event_ntfy.sh"
```

The hook receives JSON on stdin and in `NNUE_RUN_EVENT_JSON`.
