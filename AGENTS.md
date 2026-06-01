# NNUE Development Agents

This file defines three separate agent roles. Do not merge their
responsibilities.

## 1. Coding Agent

Owns training and tooling changes only.

### Responsibilities

- Implement NNUE tooling, data preparation, build config support, and focused
  tests.
- Keep changes narrow and tied to the active hypothesis in
  `IMPROVEMENT_PLAN.md`.
- Prefer `build.py` and `build.json` as the experiment interface.
- Do not stage, commit, merge, tag, push, or rewrite git history.

### Rules

- Read `IMPROVEMENT_PLAN.md` before choosing the next experiment.
- Change one hypothesis at a time: data, target construction, objective,
  architecture, or validation.
- Prefer changing `build.json` for experiment name, run directory, input data,
  target files, trainable scope, objective weights, thresholds, seeds, and row
  limits.
- Change Python/C++ tools only when `build.py` cannot express the experiment or
  a tooling bug is found.
- If a useful helper script lacks a `build.py` wrapper, add the wrapper before
  making it part of the normal workflow.
- If a tool is added, it must be staged intentionally or removed before
  stopping.
- Do not leave generated runs, temporary configs, caches, or local validation
  outputs as intended source work.

## 2. Validation Agent

Owns verification only.

### Responsibilities

- Run parity checks, exported gates, engine-side gates, replay/failure suites,
  speed profiles, and game smokes.
- Report exact pass/fail numbers and what they mean.
- Do not modify source except generated validation outputs explicitly requested
  by the Coding Agent.
- Do not commit.

### Validation Order

1. Static/tool parity:
   - Python `.pt` vs exported `.nn`.
   - Exported `.nn` vs engine eval path.
   - Selected engine moves must be scored before aggregate gates.
2. Small capability diagnostic:
   - Prove the target can move through export on a tiny set.
   - If target-only cannot move, stop and change representation or tooling.
3. Preserved engine gate:
   - Primary target rows improve.
   - Broad-preserve rows stay stable.
   - `missing_move` is `0`.
4. Replay/failure suite:
   - Use as a rejection filter.
   - Positive replay is not enough for promotion.
5. Game test:
   - Run an early 200-300 game smoke before a full SPRT.
   - If three consecutive candidates from the same objective family fail the
     smoke, close that family and change the failure theory.

### Interpretation

Reject immediately if:

- `.pt`, `.nn`, and engine disagree beyond expected quantization tolerance;
- selected engine moves are missing from scored child lists;
- primary rows stay flat while broad drift rises;
- target-only improves but broad MAE explodes;
- bounded replay improves but the early game smoke is negative.

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
- Never rewrite `main`.
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
- If a branch is hundreds of commits ahead of `main`, it is contaminated. Do
  not merge it; create a clean branch from `origin/main` and cherry-pick or
  reapply only the intended commits.
- Research branches are never merge branches. A branch used for experiments,
  failed candidates, generated targets, run configs, or exploratory tooling must
  be treated as scratch/history only.
- Mergeable feature branches must be born clean from `origin/main` and contain
  only the requested feature/fix plus its tests/docs. If useful code was first
  developed on a research branch, extract it into a fresh branch from
  `origin/main`; do not rebase the whole research branch and do not make it
  `main`.
- Keep durable tooling separate from experiments. Tool capability changes,
  experiment configs, and generated run data belong in separate commits or
  branches unless the user explicitly asks otherwise.
- Before any merge, run a contamination check:

```sh
git rev-list --count origin/main..HEAD
git diff --stat origin/main..HEAD
git diff --name-only origin/main..HEAD
```

  Stop if the count or file list contains unrelated experiment churn.
- Rebase feature branches onto `origin/main`; merge to `main` with `--ff-only`.
- Stage only files that belong to the requested change.
- Squash local fixup churn before merge.
- Do not commit unrelated NNUE run artifacts.

## Shared Documentation Rules

- Update `IMPROVEMENT_PLAN.md` only for durable conclusions:
  - a lane is closed;
  - a new blocker is identified;
  - a tool/gate was corrected;
  - the next useful action changes.
- Keep documents practical: what to change, what to run, how to judge it, and
  stop criteria.
- Keep public docs free of machine-specific hostnames and private paths.

## Shared Long-Run Rules

- Run long NNUE jobs in tmux.
- Notifications should report conclusions, task, ETA, and project state.
- The `nnue` topic is for user-facing conclusions, not phase spam. The hook
  subscribes to `done,fail,test` by default, but suppresses generic `done` and
  `fail` messages unless the event is marked user-worthy, improved,
  promotion-candidate, or critical.
- Always wake the agent for long-running phase completions and failures through
  the event hook. It sends agent control traffic to `AI_stdin` by default and
  also tries `notifai.sh` as a direct tmux wakeup path.
- Set `NNUE_NOTIFAI_TARGET` explicitly for long-running jobs. It must point to
  the active Codex pane, never a worker tmux session such as `nnue_native`,
  `nnue_reckless`, `nnue_training`, or `nnue_test`.
- Do not disable `NNUE_AI_STDIN_NTFY_ENABLE` for long runs. `AI_stdin` is agent
  control traffic, not user-facing status.
- Do not rely on inherited tmux environment. Long-run launches should set the
  event split explicitly:

```sh
NNUE_NTFY_EVENTS=done,fail,test \
NNUE_AI_STDIN_EVENTS=phase_done,done,fail \
NNUE_AI_STDIN_NTFY_ENABLE=1 \
NNUE_AI_STDOUT_EVENTS=done,fail \
NNUE_NOTIFAI_TARGET=<current-codex-pane> \
./build.py create -c build.json \
  --event-command /home/petter/code/cpp/chess/nnue/tools/events/nnue_event_ntfy.sh
```

- `AI_stdout` should receive concise structured status/conclusion output when
  a long run completes.
- Remove temporary tmux windows when the job is done.
- Do not leave nested shells in tmux panes.
- Do not start a new training run without an explicit instruction and a written
  hypothesis in `IMPROVEMENT_PLAN.md`.
