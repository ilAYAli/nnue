# Failure-Suite Baseline: reference b19794a

Generated: `2026-05-21T17:23:46+02:00`

Purpose: fixed pre-SPRT move-choice baseline for NNUE candidate gates.

Command:

```sh
~/.venv/bin/python tools/validate/validate.py failure-suite \
  ~/code/cpp/chess/enyo/bugs \
  --candidate ~/code/cpp/chess/assets/engines/reference \
  --reference ~/code/cpp/chess/assets/engines/reference \
  --oracle ~/local/bin/stockfish \
  --replay ~/local/bin/replay \
  --threads 1 \
  --jobs 4 \
  --fixed-nodes 100000 \
  --oracle-nodes 200000 \
  --output-dir ~/code/cpp/chess/nnue/runs/arch-kingbucket-v1/validate/failure-suite-baseline \
  --stderr ~/code/cpp/chess/nnue/runs/arch-kingbucket-v1/validate/failure-suite-baseline/replay.stderr \
  --run ~/code/cpp/chess/nnue/runs/arch-kingbucket-v1 \
  --event-command "$HOME/scripts/nnue_event_ntfy.sh"
```

Input logs:

```text
~/code/cpp/chess/enyo/bugs
```

Output:

```text
~/code/cpp/chess/nnue/runs/arch-kingbucket-v1/validate/failure-suite-baseline
```

Summary:

```text
positions=913
candidate_better=0
reference_better=0
sum_diff_cp=0
median_nonzero_diff_cp=0
worst_regression_cp=0
best_gain_cp=0
```

This is a same-engine baseline, so all deltas are expected to be zero. Candidate
gate results should be compared against this suite and command shape before any
long SPRT is launched.
