# Zero Self-Play NNUE Plan

This document is a handover for starting a clean experimental lineage that learns
from Enyo self-play game outcomes only. It is for fun and methodology, not the
current strongest competitive lineage.

## Goal

Create an Enyo-owned NNUE lineage that does not use Stockfish or any other engine
as a labeling oracle.

Allowed signal:

- Enyo-vs-Enyo self-play games.
- Final game outcomes: win/loss/draw.
- Positions reached by Enyo during self-play.
- Optional terminal/tablebase truth only if explicitly approved later.

Disallowed signal:

- Stockfish-labeled binpacks.
- Stockfish evals, depths, PVs, or WDL values as training labels.
- LC0 data labeled by Stockfish.
- Foreign weights, tensors, or hidden activations.
- Training targets copied from another engine.

Benchmarking against Stockfish or other engines is allowed only as measurement.
It must not feed labels back into this lineage.

## Important Correction

Do not train this lineage on Enyo search eval comments as centipawn targets.
That is self-distillation from the current `candidate.net`, not outcome-only
learning.

For outcome-only learning:

- Generate self-play PGNs.
- Extract positions with final game result.
- Convert to BulletFormat with `score = 0` for every row.
- Preserve the result/WDL field from the PGN result.
- Train with `wdl = 1.0` so the trainer learns only from game outcome.

If tooling does not support `score = 0` conversion cleanly, add the smallest
converter option, e.g. `--zero-score`, to `tools/bullet/jsonl_to_bullet_text.py`
and matching library helpers if needed. Do not rewrite the trainer unless the
converter route is insufficient.

## Methodology

Stage 0: Stop Bad Self-Distillation

- Do not continue `enyo-selfplay-1.0.0-rc1` as originally launched if it uses
  Enyo search eval comments as score targets.
- Remove abandoned fastchess outputs before reusing that run directory.
- Keep the repo diff simple and explicit.

Stage 1: Tooling Smoke

- Generate a tiny Enyo-vs-Enyo PGN using `candidate` engine and `candidate.net`.
- Extract rows from the PGN.
- Convert rows to BulletFormat with zero score and result-only targets.
- Validate the Bullet file.
- Confirm there are no abandoned/disconnect games in the sample.

Stage 2: Data Generation

- Generate a modest first corpus, not a huge overnight run.
- Use diverse openings.
- Prefer stable low concurrency over maximum concurrency; fastchess disconnects
  make the data unusable.
- Filter out abandoned/disconnected games, very short games, and games with
  unknown result.
- Keep metadata: engine SHA, net name, book, depth/time control, games,
  extraction filters, and row counts.

Stage 3: Scratch Training

- Use a new lineage name such as `enyo-zero-1.0.0-rc1`.
- Do not set `continue_from`.
- Use the current factorised architecture unless explicitly testing another
  architecture:
  - 16 input king buckets
  - 12 feature channels
  - hidden 1024
  - L2 size 16
  - 8 output buckets
- Use `wdl = 1.0`.
- Start with a short dose to prove the process, then scale only if the run is
  technically healthy.

Stage 4: Validation

- Do not use Stockfish labels.
- Use game tests only:
  - candidate vs current zero-lineage parent
  - candidate vs current competitive Enyo net for rough scale
  - optional candidate vs Stockfish net as a benchmark only
- Expect early zero-lineage nets to be weak. The first success criterion is
  technical validity and learning movement, not immediate parity.

Stage 5: Scaling

Only scale after the first result-only training run proves it can train and play
legal chess.

Possible improvements:

- More self-play games.
- Stronger Enyo search during self-play.
- Better opening diversity: UHO, DFRC/FRC, anti-draw books.
- Curriculum: regenerate data from improved checkpoints.
- Add a policy head later if the project wants a more AlphaZero-like loop.

## Expected Limitations

This is AlphaZero-like only in the broad sense that it learns from self-play
outcomes. It is not AlphaZero-equivalent:

- Enyo currently has a value NNUE, not a policy+value network.
- There is no MCTS policy improvement loop.
- Final-result labels are sparse and noisy.
- Outcome-only learning may require far more games than supervised oracle data.

Treat this as a separate research lineage, not as the fastest path to Elo.

## Development Hygiene

Use pwa-5090 for NNUE training and data generation.

Use `nnue_cmd` for long-running NNUE work. The user should be able to inspect the
active process there.

Every long-running command must send phase and completion notifications:

- phase start
- phase done
- failure
- final done

Use the existing event tools where possible:

- `tools/events/nnue_event_ntfy.sh` for structured NNUE events.
- `aistdout` for concise Codex-facing status messages.

Do not rely on silent polling. If an ad hoc loop is necessary, it must still emit
notifications and explain why it exists.

During an active experiment, keep the tracked diff minimal:

- `build.json` for the active experiment.
- `architecture.json` only for an explicit architecture experiment.
- Any tooling fix must be the smallest necessary fix, tested, committed, and then
  the active experiment relaunched.

Do not leave unrelated generated files in the repo root. In particular, remove
stray `config.json` files created by fastchess.

Do not start Forge/SPRT/training jobs casually. Each job should answer a clear
question and have a notification path.

Do not stop work silently. If a job fails, either:

- make the smallest necessary fix and continue, or
- leave a precise status message explaining the blocker.

Do not use workarounds for Forge or other infrastructure. If Forge is missing a
required feature, report that clearly instead of hiding it in NNUE scripts.

Do not overwrite user changes. Check `git status --short` before edits. If the
worktree has unrelated changes, leave them alone.

## New Session Kickoff

Recommended first actions in a fresh session:

1. Read this file.
2. Check pwa-5090 state:
   `cd ~/code/cpp/chess/nnue && git status --short`.
3. Check `nnue_cmd`:
   `tmux capture-pane -pt nnue_cmd:1.1 -S -80`.
4. Stop any mistaken self-distillation pipeline if it is still running.
5. Implement result-only Bullet conversion if missing.
6. Run a tiny smoke:
   - self-play PGN
   - extract
   - zero-score Bullet conversion
   - Bullet validation
7. Only then launch a small `enyo-zero-1.0.0-rc1` training run.

