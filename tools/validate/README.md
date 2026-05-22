# validate

`validate` runs checks on candidate nets or engines.

It includes cheap pre-SPRT checks and SPRT itself. This keeps validation under
one abstraction instead of splitting "gates" and Elo testing.

## Commands

```sh
tools/validate/validate.py static --help
tools/validate/validate.py failure-suite --help
tools/validate/validate.py sprt --help
tools/validate/validate.py search-targets --help
tools/validate/validate.py search-gate --help
tools/validate/extract_tail_targets.py --help
tools/validate/score_tail_targets.py --help
tools/validate/move_choice_gate.py --help
```

Examples:

```sh
tools/validate/validate.py static \
  --net run/candidate/model.nn \
  --data run/packed \
  --rows 100000 \
  --buckets \
  --event-command "$HOME/code/cpp/chess/nnue/tools/events/nnue_event_ntfy.sh"

tools/validate/validate.py failure-suite \
  --candidate ~/code/cpp/chess/assets/engines/candidate \
  --reference ~/code/cpp/chess/assets/engines/reference \
  --output-dir run/failure-suite \
  --event-command "$HOME/code/cpp/chess/nnue/tools/events/nnue_event_ntfy.sh" \
  ~/code/cpp/chess/enyo/bugs/*.log

tools/validate/validate.py sprt \
  --net run/candidate/model.nn \
  --games 1000 \
  --tag candidate_smoke \
  --event-command "$HOME/code/cpp/chess/nnue/tools/events/nnue_event_ntfy.sh"

tools/validate/extract_tail_targets.py \
  --bug-dir ~/code/cpp/chess/enyo/bugs \
  --out assets/failure_suite/repeated_tail_targets.csv \
  --csv candidate_a=run-a/replay_failure_suite.csv \
  --csv candidate_b=run-b/replay_failure_suite.csv

tools/validate/score_tail_targets.py \
  --targets assets/failure_suite/repeated_tail_targets.csv \
  --out assets/failure_suite/repeated_tail_move_scores.csv \
  --engine stockfish \
  --syzygy-path ~/code/cpp/chess/assets/tablebases \
  --nodes 200000

tools/validate/move_choice_gate.py \
  --scores assets/failure_suite/repeated_tail_move_scores.csv \
  --engine ~/code/cpp/chess/assets/engines/reference \
  --out run/move_choice_gate.csv \
  --nodes 100000
```

Build the unified search-aware target set from existing legal-move score CSVs:

```sh
tools/validate/validate.py search-targets \
  --scores stable=assets/failure_suite/search_failure_move_scores_20260522.csv \
  --scores repeated=assets/failure_suite/repeated_tail_move_scores_20260521.csv \
  --scores sprt_material=assets/failure_suite/material_head_mask0to6_sprt_failure_scores_20260522.csv \
  --scores sprt_output=assets/failure_suite/output_signfit_lr5e6_sprt_mine_scores_20260522.csv \
  --output runs/search-aware-targets/search_aware_targets.jsonl \
  --summary runs/search-aware-targets/summary.txt
```

Run the reusable search-aware gate:

```sh
tools/validate/validate.py search-gate \
  --targets runs/search-aware-targets/search_aware_targets.jsonl \
  --engine ~/code/cpp/chess/assets/engines/reference \
  --candidate-net runs/candidate/train/candidate/model.nn \
  --reference-net ~/code/cpp/chess/enyo/nnue/berserk-d43206fe90e4.nn \
  --out-dir runs/candidate/validate/search_gate
```

With the generic event hook:

```sh
tools/validate/validate.py sprt \
  --net runs/d12-d16-huber-cp800/train/d12-d16-huber-cp800/model.nn \
  --run runs/d12-d16-huber-cp800 \
  --games 1000 \
  --tag d12_d16_smoke \
  --event-command "$HOME/code/cpp/chess/nnue/tools/events/nnue_event_ntfy.sh"
```

The hook receives JSON on stdin and in `NNUE_RUN_EVENT_JSON`.
