# NNUE Handover

This document is for the next Codex session or human operator taking over the
Enyo NNUE loop.

## Current State

Authoritative machine for NNUE work: `pwa-5090`.

Authoritative repos:

- NNUE: `~/code/cpp/chess/nnue`
- Enyo: `~/code/cpp/chess/enyo`
- Forge: `~/code/cpp/chess/forge`

Worker engines must be built on each worker from source. Do not copy Linux
binaries to macOS workers. Linux workers should report ELF, macOS Mach-O, and
all should print the expected Enyo git hash on startup.

Active NNUE line:

- Lineage: `enyo-1.32.x`, the Enyo-native HalfKA-style architecture
  (`16x12x1024-o8`, `full_threats=false`)
- Accepted parent and gate reference: `enyo-1.31.0-rc57`
- Launch command, always from the `nnue_cmd` tmux window:

```sh
NNUE_AI_STDIN_EVENTS=done,fail MIN_SLOPE=0.05 SKIP_SMOKE=1 GAMES=1500 ./nnue iterate
```

There is no active FullThreats or scratch run. The `enyo-2.0.0` FullThreats
experiment described in earlier revisions of this file was abandoned; do not
resume it without a deliberate decision.

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

## What Is Known As Of 2026-07-29

Run naming: an `rc` is consumed only when a run **completes training**. A launch
that dies before training reuses the same number. Violating this produced a
badly tangled `enyo-1.32.0` sequence that had to be renumbered.

Watch for stale artifacts under old run names. `resume_training`
(`tools/bullet/spike_trainer/src/bin/train.rs`) early-returns only when neither
`runs/<run>/model.nn` nor `~/assets/nets/<run>.nn` exists; a leftover net from an
abandoned series makes it demand a `train.provenance.json` that does not exist
and abort with "missing or invalid provenance". The error does not name the file
it tripped on.

Falsified on this self-play generation:

- **Dose.** A sweep at 114/355/710 superbatches over the 46.5M-row gen3 corpus
  converged by one epoch: 710 reproduced 355 almost exactly (candidate-vs-
  reference mae 16.41 vs 16.79). 355 is the saturation point.
- **WDL blend.** Raising `wdl` 0.05 -> 0.20 regressed all three gate bands
  hard (endgame -27.0, 800+ -39.0, 300-799 -12.8). If revisited, probe 0.08-0.10.
- **Syzygy endgame labels.** Correcting every <=6-piece position (13,240,569
  rows, zero misses) turned endgame residuals positive for the first time
  (-6.6 -> +0.6) but came out Elo-neutral at -5.3 +/- 13.0 over 1500 games.

Two measurement traps worth knowing:

- **The residual gate is a weak Elo proxy.** It rejected five candidates, then
  passed the one that measured -5.3. Its required bands (`phase:endgame`,
  `eval:800+`) are the two with the *lowest* relative error - endgame mae looks
  large only because endgame evals are large. Normalized against mean |eval|,
  the opening is the worst phase at 50% and the endgame the best at 29%.
- **`phase:endgame` is not tablebase range.** The bucket is `piece_count < 10`,
  and only 13.1% of it is <=6 pieces. Syzygy 6-man cannot touch the other 87%.

Engine options that cancel in this harness: `use_syzygy` is `false` by default,
and every measurement (candidate-vs-reference, and the Stockfish-net benchmark)
runs the same Enyo binary on both sides, so enabling tablebases changes the
engine's strength but measures ~0 here and does nothing for the net.

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

The self-play corpus is exhausted. Dose, WDL blend, and Syzygy endgame labels
have each been tested against it and none produced Elo. The positions come from
roughly 300k games of a ~-150 Elo engine at 10+0.1, so the labels can be
corrected but the distribution cannot - the standard bootstrap problem.

The untapped resource is `data/stockfish/master-binpacks/`: 282 GB, roughly
**11.9 billion** filtered positions across 12 Stockfish binpacks. Stockfish-
labeled data is permitted; only foreign *weights* are forbidden.

That data has been touched but never actually consumed. Every prior binpack
candidate (`enyo-1.31.0-rc44` through `rc48`) ran 64-256 superbatches, i.e.
8.4M-33.5M positions, and the loader reads sequentially from the start of the
file - so each one retrained on the same sub-0.3% prefix. The "data volume
hypothesis falsified" result in `1abbcbe` was measured on **self-play** data
(`gen1-live-distill.bullet`), not on these binpacks.

Immediate next steps:

1. Train on the binpacks at a dose that actually consumes them. A full pass over
   a single 40 GB binpack is on the order of 13,000 superbatches.
2. Prefer sources whose openings match the SPRT book
   (`nodes5000pv2_UHO.binpack` against `UHO_XXL_2022_+120_+149.epd`). Avoid
   `dfrc_n5000.binpack` unless Fischer-random distribution is wanted.
3. Treat the residual gate as advisory here, not as the decision. Let games
   decide, and check the Stockfish-net ledger for absolute movement.
4. If a large binpack dose moves nothing either, the next lever is a real
   training run rather than continued fine-tuning at `lr=1e-5` from `rc57`,
   which is polish on a saturated net.

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

