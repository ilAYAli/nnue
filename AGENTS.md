# Enyo / NNUE Development Agents

This file defines three separate agent roles. Do not merge their
responsibilities. It is intentionally the same in the Enyo engine repo and the
NNUE training repo so workflow rules do not drift.

## 1. Coding Agent

Owns source changes only.

### Responsibilities

- Implement engine features, fixes, refactors, NNUE tooling, and tests.
- Keep changes narrow and directly tied to the requested bug, feature, or
  experiment hypothesis.
- Follow existing style and local patterns.
- Keep engine fixes and NNUE training-repo work in separate commits and
  preferably separate branches.
- Do not stage, commit, merge, tag, push, or rewrite git history.

### General Rules

- Add or update a focused regression test when practical.
- Change one hypothesis at a time: engine behavior, target construction, data,
  objective, architecture, validation, or performance.
- Keep hot-path code very fast. Do not add Python or shell work to hot paths.
- Do not add unrelated formatting, naming, documentation, or script churn.
- Do not leave generated build directories, run directories, logs, binaries,
  caches, or temporary files as intended source work.
- If a useful helper script becomes part of the normal workflow, add a
  `build.py` or project script wrapper before relying on it.
- If a change needs a commit, hand off to the Git Agent with the exact commit
  message.

### Enyo Engine Rules

- Keep search, UCI, tablebase, time-management, and NNUE runtime changes in
  separate edits unless they are inseparable.
- `scripts/make_candidate.sh` must build and test before promoting a candidate.
- Candidate/reference scripts must update the expected engine asset, not only
  local build outputs.

### Tablebase / Endgame Rules

- Missing tablebase directories mean "do not use them".
- Incomplete tablebase files in a configured directory must fail
  initialization.
- Do not probe staging directories or hidden temporary download directories.
- Root TB result class outranks eval.
- Eval/search may only break ties among moves with the same TB result class.
- Prefer shortest winning TB path and longest drawing/losing path before using
  eval/search as secondary tie-break.

### Eval / NNUE Runtime Rules

- Exported `.nn` and engine eval behavior must agree.
- Do not trust Python NNUE results unless exported and engine-side checks agree.
- If `Evaluate2` should replace `Evaluate`, do that as a dedicated refactor.
- Avoid spreading new `#ifdef` blocks through hot code; keep compile-time
  switch boundaries small.

### NNUE Training Rules

- Read `IMPROVEMENT_PLAN.md` before choosing the next experiment.
- Prefer `build.py` and `build.json` as the experiment interface.
- A normal training iteration should change mostly `build.json`; tool changes
  are for missing capabilities, correctness bugs, or measured speedups.
- Do not run ad hoc training scripts as the process. Wire them through
  `build.py` first.
- Selected engine moves must be scored before aggregate gates. A gate with
  missing selected moves is incomplete.
- Python `.pt`, exported `.nn`, and engine eval gates must agree before trusting
  a result.

## 2. Validation Agent

Owns verification only.

### Responsibilities

- Choose the smallest useful validation set for the touched area.
- Run builds, tests, replays, gates, speed profiles, game smokes, and SPRTs.
- Report exact pass/fail numbers and what they mean.
- Do not modify source except for generated test outputs explicitly requested by
  the Coding Agent.
- Do not commit.

### Enyo Checks

General C++ change:

```sh
cmake --build build --target test -j8
build/test
```

NNUE accumulator/runtime change:

```sh
build/test --gtest_filter='nnue_audit.*:network_audit.*'
```

Tablebase change:

```sh
build/test --gtest_filter='syzygy_*:*tablebase*:uci_root.*'
```

Bug-log replay:

```sh
replay ./bugs/<game>.log
```

SPRT:

```sh
../assets/scripts/sprt --games 1000 --concurrency 8 --reference ../assets/engines/reference --candidate ./build/enyo --ntfy-url https://ntfy.wahlman.no/sprt
```

### NNUE Validation Order

1. Static/tool parity:
   - Python `.pt` vs exported `.nn`.
   - Exported `.nn` vs engine eval path.
   - `missing_move` must be `0`.
2. Small capability diagnostic:
   - Prove the target can move through export on a tiny set.
   - If target-only cannot move, stop and change representation or tooling.
3. Preserved engine gate:
   - Primary target rows improve.
   - Broad-preserve rows stay stable.
4. Replay/failure suite:
   - Use as a rejection filter.
   - Positive replay is not enough for promotion.
5. Game test:
   - Run an early 200-300 game smoke before a full SPRT.
   - If three consecutive candidates from the same objective family fail the
     smoke, close that family and change the failure theory.

### Interpretation

Reject or escalate if:

- any focused regression test fails;
- replay still reproduces the bug;
- replay shows a new blunder or illegal PV/root move;
- `.pt`, `.nn`, and engine disagree beyond expected quantization tolerance;
- selected engine moves are missing from scored child lists;
- target-only improves but broad behavior explodes;
- UCI `info score` is misleading for mate/TB cases;
- SPRT or smoke games are negative for a strength-affecting change.

Consider a candidate only when:

- focused tests and exported gates pass;
- engine gates pass with complete move coverage;
- relevant replays no longer reproduce the bug;
- replay has no unexplained tail regression;
- a smoke game test is at least neutral-positive, or the user explicitly
  accepts the playing-strength risk.

## 3. Git / Release Agent

Owns git operations only.

### Responsibilities

- Create branches, stage files, commit, rebase, merge, tag, push, and create
  reference/candidate engines when explicitly instructed.
- Enforce clean history and correct commit identity.
- Never resolve source conflicts without handing back to the Coding Agent.

### Rules

- Never commit directly to `main`.
- Never create merge commits unless the user explicitly asks.
- Rebase feature branches onto `origin/main`; merge to `main` with `--ff-only`.
- Never rewrite `main` unless the user explicitly instructs it.
- Never use bot authors or AI co-author trailers.
- Verify commit identity before committing:

```sh
git config user.name "Petter Wahlman"
git config user.email "petter@wahlman.no"
```

- Start every new change from current `origin/main`:

```sh
git fetch origin
git switch -c feature/<short-name> origin/main
git rev-list --count origin/main..HEAD
```

- The branch creation ahead count must be `0`.
- Before merge, `git rev-list --count origin/main..HEAD` must equal the number
  of intended local commits.
- Before merge, `git diff --stat origin/main..HEAD` must match the requested
  scope.
- Before pushing a feature branch, verify its upstream and push explicitly:

```sh
git status --short --branch
git push -u origin HEAD:refs/heads/feature/<short-name>
```

- Do not use plain `git push` from a branch that tracks `origin/main`.
- If a branch is hundreds of commits ahead of `main`, it is contaminated. Do
  not merge it; create a clean branch from `origin/main` and cherry-pick or
  reapply only the intended commits.
- Stage only files that belong to the requested change.
- Squash local fixup churn before merge.
- Do not commit unrelated run artifacts.

### Engine References

After a verified Enyo main merge that changes engine behavior, create the
reference engine with:

```sh
scripts/make_reference.sh
```

Create a candidate only after tests pass:

```sh
scripts/make_candidate.sh
```

## Shared Documentation Rules

- Update `IMPROVEMENT_PLAN.md` only for durable NNUE conclusions:
  - a lane is closed;
  - a new blocker is identified;
  - a tool/gate was corrected;
  - the next useful action changes.
- Keep documents practical: what to change, what to run, how to judge it, and
  stop criteria.
- Keep public docs free of machine-specific hostnames and private paths.
- Local agent workflow files may mention local hosts and sessions when needed.

## Shared Long-Run Rules

- Run all long-running tasks on `pwa-5090` in the tmux session `nnue_native`.
- Localhost is for short edits, dry-runs, syntax checks, and git only.
- Before starting a long run, verify the host, branch, and tmux session:

```sh
hostname
git status --short --branch
tmux display-message -p '#S:#W'
```

- Do not start long validation from an uncommitted or unexpected branch unless
  the user explicitly asks for that exact state.
- Prefer the project wrapper for long NNUE jobs:

```sh
./build.py create -c build.json --event-command /home/petter/code/cpp/chess/nnue/tools/events/nnue_event_ntfy.sh
```

- Keep `nnue_native` tidy. Reuse one suitable window for the active task and
  close stale windows instead of accumulating panes.
- Notifications should include task, ETA, and current project state. Avoid
  phase spam.
- Remove temporary tmux windows when the job is done.
- Do not leave nested shells or abandoned processes in tmux panes.
