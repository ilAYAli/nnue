# LC0 → Enyo Bullet conversion

## Goal

Convert LC0 V6 archives in
`~/assets/training/lc0/test91-forge-input` to:

`~/assets/training/bullet/lc0-stockfish/test91-stockfish-enyo.bullet`

For every selected position, Forge must:

1. decode the LC0 record;
2. obtain the **Stockfish UCI search label** using
   `~/assets/engines/reference` and
   `~/assets/nets/nn-0ee0657fb25e.nnue` through `EvalFile`;
3. apply `--enyo-runtime-target` (Enyo clamp/phase normalization) to that
   Stockfish score; and
4. serialize the result as BulletFormat.

Enyo is **not** the labeling engine.  Do not use `enyo_e8d285b` for this
conversion and do not run `tools/bullet/enyo_recalibrate_labels.py`.

`--enyo-runtime-target` is a deterministic per-score transform in
`tools/score/label.py`; Forge only distributes that program. It does not fit or
recalibrate Stockfish scores.

## Required wrapper workflow

Use **only** `tools/forge/lc0_to_enyo_bullet.py` as the conversion entry point.
It must:

1. inspect and byte-weight the LC0 archives, then create exact non-overlapping
   Forge tasks;
2. use Forge to run `tools/score/label.py` for each task, which decodes LC0,
   obtains the Stockfish label, applies `--enyo-runtime-target`, and writes a
   shard Bullet file;
3. wait for every checksum-valid task result, then use
   `tools/validate/validate_bullet_results.py` to validate and merge the
   shards; and
4. write final provenance only after the merged file passes validation.

## Required fixes

- Keep byte-weighted batches; archive sizes differ. Treat the task count as a
  **per-batch** target: start with e.g splitting up each batch in 100 MB tasks, not a
  1,600-task corpus-wide budget divided across all batches.
- A missing worker must be quarantined and its tasks requeued on healthy
  workers. Retain task stderr and engine exit status.
- Fix Forge's execution checksum so legitimate host-local path mappings do not
  change it, while engine/net/source/shard identity remain bound. Do not bypass
  finalization.
- Merge only complete, checksum-valid shards. Validate the final Bullet file
  independently: full coverage, Stockfish net loaded without fallback, and
  non-degenerate W/D/L and score distributions. Write provenance with hashes.

The latest failed run, `label-lc0-stockfish-enyo-input-0000-12-20260829-145338`,
produced 80 shards but Forge rejected all of them with `task execution checksum
mismatch`. No final corpus is valid yet.

## Verification and resume contract

Implement one stable work directory:
`~/assets/training/bullet/lc0-stockfish/.test91-stockfish-enyo.work/`.
The current PID-named `.batches.*` directory is deleted on failure; replace
that behavior. A restart must read this work state and reuse only batches
explicitly marked valid.

For every batch, retain and verify:

1. **Plan:** source inventory digest, archive paths/bytes, weighted shard plan,
   and Forge preflight manifest. Every source file must occur exactly once.
2. **Launch:** Forge's materialized manifest must match the preflight task
   inputs exactly.
3. **Shard:** Bullet output, stats JSON, task ID, task stderr, engine exit
   status, and accepted Forge execution checksum. Stats must prove
   `score_source=uci`, expected Stockfish engine/net hashes, and no fallback.
4. **Batch:** independent structural validation plus non-degenerate W/D/L and
   score distributions. Mark it valid atomically only after those checks pass.

On interruption, keep the work state and requeue only unfinished or invalid
shards. Never recompute a completed valid batch. Never merge a batch that did
not pass all four gates.

After all batches are valid, merge to a temporary final path; independently
validate full coverage, structure, W/D/L, score distribution, and SHA-256;
write provenance; then atomically publish the final Bullet file and sidecar.

## Temporary data

Never delete the raw LC0 input or a validated final corpus. After a confirmed
stopped failed run, its temporary data is at:

- Required persistent resume state:
  `~/assets/training/bullet/lc0-stockfish/.test91-stockfish-enyo.work/`
- `~/assets/training/bullet/lc0-stockfish/.test91-stockfish-enyo.batches.*`
- `~/code/chess/forge/runs/label-lc0-stockfish-enyo-*`
- Linux workers: `~/.cache/forge/{unpacked-lc0,task-inputs,worker-state/label-lc0-stockfish-enyo-*}`
- macOS workers: `~/Library/Caches/forge/{unpacked-lc0,task-inputs,worker-state/label-lc0-stockfish-enyo-*}`
- The per-input cache named in that run's manifest:
  `~/.cache/forge/inputs/<inventory-digest>/`

Do not wildcard-delete shared Forge cache roots while another Forge run is
active. Long commands use the existing `nnue_cmd` tmux session only.
