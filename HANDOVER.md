# NNUE Handover

This document is for the next Codex session or human operator taking over the
Enyo NNUE loop.

## Current State

Authoritative machine for NNUE work: `pwa-5090`.

Authoritative repos:

- NNUE: `~/code/cpp/chess/nnue`
- Enyo: `~/code/cpp/chess/enyo`
- Forge: `~/code/cpp/chess/forge`

Current engine support:

- Enyo commit: `ac7ada8 feat: load Enyo FullThreats nets`
- NNUE commit: `f3549bc feat: train Enyo FullThreats nets`
- `ac7ada8` is pushed to Enyo `origin/main`.
- `f3549bc` exists on NNUE `main`; pwa-5090 was ahead of `origin/main` when
  this file was written.
- Worker engines must be built on each worker from source. Do not copy Linux
  binaries to macOS workers.
- Verified worker `~/assets/engines/candidate` targets after deployment:
  `enyo_ac7ada8` on Linux hosts as ELF and on macOS as Mach-O.

Active NNUE run:

- Run: `enyo-2.0.0-rc1`
- Architecture: scratch Enyo FullThreats, `16x12x1024-o8`
- Reference for gates: `enyo-1.16.0-rc3`
- Data: `data/bullet/enyo-scratch-broad-1.0.0-rc1.bullet`
- Dose: `32768` superbatches
- WDL: `0.15`
- LR: `0.001 -> 0.000005`
- Activation L1: `0.00001`
- Trainable: `all`
- Launch command used in `nnue_cmd`:

```sh
NNUE_AI_STDIN_EVENTS=done,fail MIN_SLOPE=0.05 SKIP_SMOKE=1 GAMES=1500 ./nnue iterate
```

Because this is a scratch architecture, the loop asked:

```text
continue_from not found. Start a new scratch net? [y/N]
```

The prompt was answered with `y`.

## Current Lineage

The old `enyo-1.x` line is a HalfKA-style Enyo-native lineage with 16 input
buckets, 12 feature channels, hidden 1024, 8 output buckets, and no FullThreats.

Important recent absolute Stockfish-net checks used
`nn-0ee0657fb25e.nnue` as reference. The best recorded 500-game absolute scores
were around:

- `enyo-1.16.0-rc3`: about `-150.5` Elo versus Stockfish net under `enyo_f9cdc38`
- `enyo-1.17.0-rc1`: about `-156.4` Elo
- `enyo-1.18.0-rc1`: about `-155` to `-156` Elo
- Later same-architecture continuations often regressed toward `-170` to `-190`

Do not blindly trust `~/assets/nets/candidate.net` as the strongest net. At the
time this handover was written, pwa-5090 had pointed it at `enyo-1.19.0-rc2`,
which was weaker in the Stockfish-net ledger than `enyo-1.16.0-rc3`.

The current experiment deliberately starts a new architecture major:

- `enyo-2.x`: Enyo FullThreats architecture
- It is scratch-trained, not initialized from another engine or net.
- It uses the proven broad Bullet corpus and WDL/L1 recipe as the first
  calibration test.

## Iteration Flow

The normal loop is:

1. Edit only `build.json` and, for architecture experiments, `architecture.json`.
2. Validate the resolved plan:

```sh
tools/bullet/train plan --build build.json --arch architecture.json --defaults defaults.json
```

3. Launch from tmux session `nnue_cmd` on `pwa-5090`:

```sh
NNUE_AI_STDIN_EVENTS=done,fail MIN_SLOPE=0.05 SKIP_SMOKE=1 GAMES=1500 ./nnue iterate
```

4. The loop performs:
   - train/export
   - startpos gate
   - static eval gate
   - move gate
   - Stockfish-net absolute gate
   - candidate-vs-reference Forge SPRT
   - commit accepted or rejected config/result
   - advance `build.json`

The loop commits accepted and rejected iterations using `git commit --only` for
the relevant config files and benchmark ledger. It should not commit unrelated
untracked files.

Scratch runs are special: the loop only allows scratch training from an
interactive terminal. If `continue_from` is missing, answer `y` only when the
scratch architecture was intentional.

## Event Loop

Main hook:

```sh
tools/events/nnue_event_ntfy.sh
```

Routing in the hook:

- `fail` goes to `ping`
- `iteration_done` goes to `nnue`
- other phase events go to `AI_stdout`
- `AI_stdin` wakeups are opt-in via `NNUE_AI_STDIN_EVENTS`

For the autonomous NNUE loop, launch with:

```sh
NNUE_AI_STDIN_EVENTS=done,fail
```

That should wake Codex only for done/fail events. Do not poll the loop in place
of events. A single snapshot for diagnosis is fine; repeated status polling is
not.

If notifications stop working:

1. Test from pwa-5090:

```sh
notifai.sh test
```

2. If that does not reach `AI_stdin`, inspect the notification socket and tmux
   target before touching training logic.
3. Do not add Forge or NNUE workarounds to compensate for broken notification
   delivery. Fix the notification path.

## What To Do

- Keep training and validation on `pwa-5090`.
- Use `nnue_cmd` for the NNUE loop.
- Keep active iteration diffs limited to `build.json` and/or
  `architecture.json`, except when the user explicitly asks for documentation
  or tooling fixes.
- Record meaningful plan changes in `IMPROVEMENT_PLAN.md`.
- Use `benchmarks/stockfish-net.jsonl` for absolute progress versus
  `nn-0ee0657fb25e.nnue`.
- Use Forge for distributed tests and deployment.
- Deploy engines from source on each worker:

```sh
forge '
cd ~/code/cpp/chess/enyo
git switch main
git fetch origin main
git reset --hard <engine_commit>
./scripts/deploy.sh
'
```

- Verify worker binaries after deploy:
  - Linux workers should report ELF.
  - macOS workers should report Mach-O.
  - all should print the expected Enyo git hash on startup.

## What Not To Do

- Do not use weights from Stockfish, Berserk, Reckless, or any other engine to
  initialize an Enyo net.
- Do not copy a Linux engine binary to macOS workers.
- Do not start unrelated Forge tests while the user is trying to diagnose a
  specific regression.
- Do not change multiple training variables unless the plan explicitly calls for
  that combination.
- Do not treat small candidate-vs-parent wins as proof of absolute progress.
  Always check the Stockfish-net ledger.
- Do not trust `candidate.net` without checking where it points.
- Do not rewrite Forge behavior from the NNUE repo. If Forge lacks a needed
  feature, report that boundary clearly.
- Do not introduce complex tooling logic without explicit approval.

## Testing Policy

Primary gate:

- Candidate-vs-reference Forge SPRT, normally `1500` games through `./nnue`.

Absolute benchmark:

- Reference net: `nn-0ee0657fb25e.nnue`
- Script: `tools/validate/sprt_net.sh`
- Default games: `500`
- Ledger: `benchmarks/stockfish-net.jsonl`

Use the absolute benchmark as a veto for obvious regressions. A net that looks
positive versus its parent but gets worse versus the Stockfish net is not real
progress toward the current goal.

Do not run `default.net` checks anymore unless the user explicitly asks. The
current target is the Stockfish net.

## Path Forward

Immediate next step:

1. Let `enyo-2.0.0-rc1` finish training and gates.
2. If it fails static/startpos/load gates, fix only the FullThreats
   implementation bug and relaunch the same declared experiment.
3. If it passes gates but fails badly versus `enyo-1.16.0-rc3`, record the
   rejection and decide whether FullThreats needs:
   - lower LR,
   - smaller dose,
   - different WDL,
   - or a smaller hidden width before more expensive training.
4. If it is positive or close versus `enyo-1.16.0-rc3`, run/record the
   Stockfish-net benchmark immediately.

Promising follow-up experiments if `enyo-2.0.0-rc1` is alive:

- Same FullThreats architecture with reduced LR, for example `0.0005`.
- Same architecture with shorter dose, for example `8192` or `16384`
  superbatches, if the first run overtrains.
- Same architecture with hidden 768 only after the 1024-wide version has a clear
  result.
- Consider sparse/activation-L1 continuation only after a FullThreats baseline
  is known to be competitive.

Do not resume incremental `enyo-1.x` same-architecture tuning unless the
FullThreats line clearly fails. The `enyo-1.x` line appeared saturated around
`-150` to `-180` Elo versus the Stockfish net.

## Recovery Notes

If `nnue_cmd` is idle unexpectedly:

```sh
tmux capture-pane -pt nnue_cmd -S -80
```

If it failed because Forge was busy, Forge now supports queueing, but the NNUE
loop may still need to be relaunched after recording the failed attempt.

If a run was intentionally stopped because it was clearly losing, commit the
failure result if enough evidence exists, choose one new hypothesis, edit
`build.json` and/or `architecture.json`, and relaunch the loop from `nnue_cmd`.

If the active run creates stale artifacts, remove only that run's stale
train/export artifacts or use the existing `FORCE=1` path when appropriate. Do
not delete unrelated data.

