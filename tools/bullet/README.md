# Bullet Backend

Experimental backend for testing Bullet-trained NNUE architectures with Enyo
data.

Normal entry point:

```sh
tools/bullet/train plan --build build.json
tools/bullet/train all --build build.json
```

Pipeline shape:

```text
posgen -> score -> Bullet text -> BulletFormat -> Bullet trainer
```

This path currently writes Bullet checkpoints under:

```text
runs/<run>/train/<name>/checkpoints/
```

It does not export a normal Enyo `.nn`. The experimental Enyo branch can load
these raw `quantised.bin` checkpoints directly for architecture tests, but the
current search path is still too slow for serious SPRT.

The current 1024-hidden spike writes `quantised.bin` as raw tensors with no
header:

```text
l0w  i16  [1024, 7680]  input weights, 10 mirrored king buckets
l0b  i16  [1024]        input biases
l1w  i8   [128, 1024]   first head layer, transposed and quantised
l1b  f32  [128]
l2w  f32  [256, 16]     second head layer, transposed
l2b  f32  [256]
l3w  f32  [8, 32]       output layer, transposed
l3b  f32  [8]
```

That architecture is not the same as Enyo's current exported `.nn` format. A
smaller 768-hidden checkpoint uses the same layout with the first two tensor
dimensions adjusted:

```text
l0w  i16  [768, 7680]
l0b  i16  [768]
l1w  i8   [128, 768]
```

`bullet_mode=enyo` exports normal Enyo `.nn` files. Use
`--enyo-input-buckets 32` for the current native runtime layout. Use `16` only
for explicit legacy compatibility.

On RTX 50-series GPUs with CUDA 12.4, `--bullet-cuda-arch auto` patches Bullet's
cached CUDA runtime to use `compute_90`, because NVRTC rejects `sm_120`.

## Distributed local-SGD training

The spike trainer can run one local-SGD group across a trusted LAN. One machine
is the coordinator and trains normally; every other machine trains on its own
data shard. At each sync point the coordinator averages all model weights,
broadcasts the result, and every member resets its optimiser state. This is
weight averaging, not gradient-parallel training, so the machines must use the
same architecture, seed, schedule, batch size, and Bullet revision.

Prepare one non-overlapping Bullet/sfbinpack shard on each host, then run the
normal `tools/bullet/train run` command in the existing `nnue_cmd` session on
each machine. Set `ENYO_BULLET_DISTRIBUTED_DATA` to that host's already-prepared
shard; it overrides only the training input, so it is safe to share the same
`build.json` and architecture. Do not use `all` for the members: each host must
prepare its local shard before the group starts.

For the two-host `pwa-llm` + `pwa-hak` group:

```sh
# pwa-llm
ENYO_BULLET_DISTRIBUTED_ROLE=coordinator \
ENYO_BULLET_DISTRIBUTED_RUN_ID=enyo-16.0.0-rc2 \
ENYO_BULLET_DISTRIBUTED_NODE_ID=pwa-llm \
ENYO_BULLET_DISTRIBUTED_NUM_PEERS=1 \
ENYO_BULLET_DISTRIBUTED_SYNC_EVERY=512 \
ENYO_BULLET_DISTRIBUTED_DATA=/path/to/llm-shard.bullet \
tools/bullet/train run --build build.json

# pwa-hak
ENYO_BULLET_DISTRIBUTED_ROLE=worker \
ENYO_BULLET_DISTRIBUTED_RUN_ID=enyo-16.0.0-rc2 \
ENYO_BULLET_DISTRIBUTED_NODE_ID=pwa-hak \
ENYO_BULLET_DISTRIBUTED_COORDINATOR_ADDR=pwa-llm:9219 \
ENYO_BULLET_DISTRIBUTED_SYNC_EVERY=512 \
ENYO_BULLET_DISTRIBUTED_DATA=/path/to/hak-shard.bullet \
tools/bullet/train run --build build.json
```

To add `pwa-5090` later, launch it as a second worker and change the
coordinator's `ENYO_BULLET_DISTRIBUTED_NUM_PEERS` to `2`.
Choose the sync interval from a throughput measurement: weights are about
104 MB for the current spike topology, so frequent exchanges can erase the
benefit of a second GPU. Start at `512` and benchmark on the actual link.
`ENYO_BULLET_DISTRIBUTED_TIMEOUT_SECS` defaults to 120 and may be raised for a
slow or wireless link. The coordinator listens on `0.0.0.0:9219` by default;
set `ENYO_BULLET_DISTRIBUTED_LISTEN_ADDR` if that port is unavailable.

All members must start from the same checkpoint superbatch when resuming. The
trainer validates the run id, distinct worker identities, synchronization round,
model tensor schema, and finite weights before averaging. It also performs an
unconditional final sync, so only the coordinator's completed checkpoint should
be exported or promoted.

For a guarded smoke test, use `tools/bullet/distributed_smoke.py`. It checks
that every host has identical trainer sources, refuses to overlap an active
Bullet trainer, builds the test binary, streams host logs into the reserved run,
and requires the final `quantised.bin` SHA-256 to match on every host. It does
not edit `build.json`; reserve and prepare a short test build plus shards first.
The default is dry-run command output. Run it from `pwa-llm`'s existing
`nnue_cmd` session, adding `--launch` only after reviewing the commands:

```sh
tools/bullet/distributed_smoke.py \
  --build build.json \
  --coordinator-data /path/to/llm-shard.bullet \
  --worker pwa-hak \
  --worker-data pwa-hak=/path/to/hak-shard.bullet
```

Add `--worker pwa-5090 --worker-data pwa-5090=/path/to/5090-shard.bullet`
when the third host is ready.
