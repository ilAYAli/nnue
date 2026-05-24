# Native Enyo NNUE

This file documents the normal Enyo `.nn` runtime design.

Source of truth in Enyo is currently `src/nnue_model.hpp`.

## Shape

```text
king buckets                 = 32
legacy king buckets          = 16, accepted by loader and expanded
piece/color types            = 12
squares                      = 64
input feature rows           = 32 * 12 * 64 = 24576
accumulator width            = 1024 per perspective
perspectives                 = 2
dense-head input             = 2048
dense hidden layer 1         = 16
dense hidden layer 2         = 32
output                       = 1 centipawn score
optional output buckets      = 8, only for bucketed-head files
optional threat branch       = compile-time experiment, disabled by default
```

## Data Path

```text
active sparse features
  -> white and black 1024-wide perspective accumulators
  -> side-to-move accumulator + opponent accumulator = 2048 values
  -> 2048 -> 16
  -> 16 -> 32
  -> 32 -> 1
  -> output / 32
  -> phase scale and search-side eval handling in Enyo
```

The NNUE returns a centipawn evaluation. Search still chooses moves.

## Features

A sparse feature is:

```text
king bucket + relative piece/color type + mirrored piece square
```

Runtime feature index:

```text
feature =
    king_bucket * 12 * 64
  + relative_piece_type * 64
  + mirrored_piece_square
```

Each occupied square contributes one active feature per perspective. Empty
squares do not contribute rows.

## Stored Weights

Current 32-king-bucket native `.nn` payload:

```text
input weights        = 24576 * 1024 = 25165824 int16 values
input biases         = 1024 int16 values
L1 weights           = 2048 * 16 = 32768 int8 values
L1 biases            = 16 int32 values
L2 weights           = 16 * 32 = 512 float values
L2 biases            = 32 float values
output weights       = 32 float values
output bias          = 1 float value
total trained values = 25200209
.nn payload size     = 50368836 bytes, about 48.0 MiB
```

Legacy 16-king-bucket files are smaller and are expanded by the loader. Native
training should target the current 32-bucket format.

## Native Lane

`nnue_native` means an Enyo-owned net:

```text
init_net = null or an Enyo-native checkpoint
runtime  = Enyo .nn
goal     = produce and improve Enyo-owned weights over time
```

Bullet may be used as a faster trainer, but native experiments must export a
normal Enyo `model.nn` with `--bullet-mode enyo` and
`--bullet-enyo-input-buckets 32`. The 16-bucket Bullet path is legacy
compatibility only.

## Compatibility Experiments

The engine has experimental support for:

```text
bucketed output heads
threat branch weights
alternate runtime loading
```

Those are not the baseline native design. Treat them as gated architecture
experiments, not assumptions about the normal `.nn` file.
