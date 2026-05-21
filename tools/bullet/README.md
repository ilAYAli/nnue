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

On RTX 50-series GPUs with CUDA 12.4, `--bullet-cuda-arch auto` patches Bullet's
cached CUDA runtime to use `compute_90`, because NVRTC rejects `sm_120`.
