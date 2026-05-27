# NNUE Development Agents

This file defines three separate agent roles. Do not merge their
responsibilities.

## 1. Coding Agent

Owns training/tooling changes only.

### Responsibilities

- Implement NNUE tooling, data preparation, build config support, and tests.
- Keep changes narrow and tied to the active hypothesis in
  `IMPROVEMENT_PLAN.md`.
- Prefer `build.py` and `build.json` as the experiment interface.
- Policy-sidecar work must use `./build.py policy ...`. If a useful helper
  script lacks a wrapper, add the `build.py` wrapper before using it as part of
  the workflow.
- Policy-specific helper scripts should fail direct execution unless invoked
  through `build.py` or an explicit debug escape hatch is set.
- Do not stage, commit, merge, tag, push, or rewrite git history.

### Rules

- Read `IMPROVEMENT_PLAN.md` before choosing the next experiment.
- Change one hypothesis at a time: target construction, loss/objective,
  architecture, or data.
- Prefer changing `build.json` for:
  - experiment name and `run_dir`;
  - input data or target file;
  - trainable scope;
  - loss/objective weights;
  - gate thresholds;
  - seed and row limits.
- Change Python/C++ tools only when `build.py` cannot express the experiment or
  a bug is found.
- Do not let one-off helper scripts become the primary workflow. Promote useful
  helpers into `build.py` before continuing the experiment series.
- If a tool is added, it must be staged intentionally or removed before
  stopping.
- Do not run another scale/sweep variant after the current failure theory is
  rejected.
- The W4-preserved local repair family (head-only, bucket-only, guard-row,
  pvdesc, LR/selector sweeps, Lichess-5k reruns) is fully exhausted. Do not
  reopen any variant without a new mechanistic hypothesis written first.
- The next net attempt must be a real new training setup: broader dataset,
  proper scratch/provenance path, move-choice validation before any long run,
  and a 256-game smoke before deep analysis.
- Before any architecture change: prove no-op parity, measure NPS (reject above
  5% regression), and prove export-visible movement on a small diagnostic.

### Target Classification

Classify every native target row as one of:

- `static`: ordinary bare-FEN eval/move-choice target;
- `search-coupling`: shallow search selected a bad move;
- `history-policy`: full replay differs from bare FEN because of repetition or
  history;
- `broad-preserve`: current reference already behaves well and should not drift.

Do not train ordinary FEN targets from `history-policy` rows.

## 2. Validation Agent

Owns NNUE verification only.

### Responsibilities

- Run parity checks, exported gates, engine-side gates, replay/failure suites,
  and game smokes.
- Report exact numbers and classify the result.
- Do not modify source except for generated validation outputs explicitly
  requested by the Coding Agent.
- Do not commit.

### Validation Order

1. Static/tool parity
   - Python `.pt` vs exported `.nn`.
   - Exported `.nn` vs engine eval path.
   - Missing selected moves must be scored before aggregate gates.

2. Small capability diagnostic
   - Prove the target can move through export on a tiny set.
   - If target-only cannot move, stop and change representation or tooling.

3. Preserved engine gate
   - Primary target rows must improve.
   - Broad-preserve rows must remain stable.
   - `missing_move` must be `0`.

4. Replay/failure suite
   - Useful as a rejection filter.
   - Positive replay is not enough for promotion.

5. Game test
   - Run an early 200-300 game smoke before a full 1000-game SPRT.
   - If three consecutive candidates from the same objective family fail the
     smoke, close that family and change the failure theory.

For architecture changes, run incremental-vs-refresh accumulator tests and NPS
checks before training.

### Interpretation

Reject immediately if:

- selected engine moves are missing from scored child lists;
- `.pt`, `.nn`, and engine disagree beyond expected quantization tolerance;
- primary rows stay flat while broad drift rises;
- target-only improves but broad MAE explodes;
- bounded replay improves but the early game smoke is negative.

Useful but not promotable:

- target-only overfits a hard set;
- bounded replay crosses positive;
- scalar MAE/sign improves without engine move-choice improvement.

Consider a candidate only if:

- exported gates pass;
- engine gates pass with complete move coverage;
- broad-preserve rows remain stable;
- replay has no unexplained tail regression;
- a smoke game test is at least neutral-positive.

## 3. Git / Release Agent

Owns git operations only.

### Responsibilities

- Create branches, stage files, commit, rebase, merge, tag, and push when
  explicitly instructed.
- Enforce clean history and correct commit identity.
- Never resolve source conflicts without handing back to the Coding Agent.

### Rules

- Never commit directly to `main`.
- Never create merge commits unless the user explicitly asks.
- Rebase feature branches onto `main`; merge to `main` with `--ff-only`.
- Never rewrite `main`.
- Never use bot authors or AI co-author trailers.
- Verify commit identity before committing:

```sh
git config user.name "Petter Wahlman"
git config user.email "petter@wahlman.no"
```

- Stage only files that belong to the requested change.
- Squash local fixup churn before merge.
- Do not commit unrelated NNUE run artifacts.
- If a commit message is needed, request it from the Coding Agent.

## Shared Documentation Rules

- Update `IMPROVEMENT_PLAN.md` only for durable conclusions:
  - a lane is closed;
  - a new blocker is identified;
  - a tool/gate was corrected;
  - the next useful action changes.
- Keep documents practical: what to change, what to run, how to judge it, stop
  criteria.
- Do not add machine-specific hostnames, local paths, or private notes to public
  documentation.

## Shared Long-Run Rules

- Run long NNUE jobs in the appropriate tmux session.
- Notifications should report task, ETA, and project state. Avoid phase spam.
- Remove temporary tmux windows when the job is done.
- Do not leave nested shells in tmux panes.
- NNUE training is paused. Do not start a new run without an explicit instruction
  and a written hypothesis in `IMPROVEMENT_PLAN.md`.
- Engine stability work takes priority over NNUE research during the pause. Keep
  them separated: engine bugs go to the engine repo, NNUE tooling stays here.
