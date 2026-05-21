# Bullet Backend

Experimental backend for testing Bullet-trained, Reckless-like NNUE
architectures with Enyo scored JSONL data.

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

It does not yet export an Enyo-loadable `.nn`. Use it to test whether Bullet
training and richer architectures are worth porting into Enyo.

Inspect a checkpoint:

```sh
tools/bullet/inspect_checkpoint.py \
  runs/bullet-reckless-spike-100k/train/bullet-reckless-spike-100k/checkpoints/bullet-reckless-spike-100k-2
```

The current spike writes `quantised.bin` as raw tensors with no header:

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
real Bullet/Reckless-like candidate needs an Enyo evaluator/loader for this
layout, or a Bullet trainer configured to save exactly Enyo's current layout.

On RTX 50-series GPUs with CUDA 12.4, `--bullet-cuda-arch auto` patches Bullet's
cached CUDA runtime to use `compute_90`, because NVRTC rejects `sm_120`.
