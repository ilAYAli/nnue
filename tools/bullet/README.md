# Bullet Backend

Experimental backend for testing Bullet-trained NNUE architectures with Enyo
scored JSONL data.

Normal entry point:

```sh
./build.py create --backend bullet ...
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

Inspect a checkpoint:

```sh
tools/bullet/inspect_checkpoint.py \
  runs/bullet-spike-100k/train/bullet-spike-100k/checkpoints/bullet-spike-100k-2
```

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

A real Bullet-trained candidate now needs speed work on the Enyo
evaluator/head path, or a Bullet trainer configured to save exactly Enyo's
current layout if the goal is faster tooling rather than a different
architecture.

On RTX 50-series GPUs with CUDA 12.4, `--bullet-cuda-arch auto` patches Bullet's
cached CUDA runtime to use `compute_90`, because NVRTC rejects `sm_120`.
