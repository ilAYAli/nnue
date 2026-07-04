# Enyo Native NNUE

This repository trains Enyo-native NNUE networks. The current native line began
from newly initialized Enyo weights.

The normal experiment interface is `build.json`; shared training defaults live
in `defaults.json`; runtime/trainer shape lives in `architecture.json`.

## Provenance

Training data currently comes from a combination of selfplay and the
`official-stockfish/master-binpacks` dataset:

The dataset is licensed as ODbL-1.0. If a trained net is published, include
provenance/notice that Stockfish master binpacks under ODbL-1.0 were used.

Training uses Bullet through the local `tools/bullet/spike_trainer` wrapper.
The pinned dependency is `bullet_lib` from https://github.com/jw1912/bullet at
commit `d372d487aedfeb8bdc256b9f694dbcd41016bf82`. Bullet is MIT licensed.

## Architecture

`architecture.json` is the trainer/export/runtime contract. Changing it is an
architecture experiment and requires matching engine support and parity checks.

The promoted layout uses a factorised 16-bucket, 12-channel input transformer
with 1,024 hidden units and 8 output buckets. Architecture screening also tested
an Enyo-specific HalfKAv2-style `10x11x768-o8` layout with horizontal mirroring
and a shared king channel. It won the short-training screen but did not beat the
fully trained promoted layout, so it was not adopted.

```json
{
  "name": "native-16bucket-12ch-1024-v3-factorised",
  "lineage": "native",
  "mode": "enyo",
  "hidden": 1024,
  "l2_size": 16,
  "feature_channels": 12,
  "input_buckets": 16,
  "output_buckets": 8,
  "input_factoriser": true,
  "eval_scale": 400.0,
  "l0_std": 8.0,
  "l1_std": 1.0,
  "l1_export_scale": 1.0,
  "export_format": "enyo-native-v1"
}
```

## Defaults

`defaults.json` contains the complete shared training configuration. A value in
`build.json` overrides the same value from `defaults.json`; keep overrides in
`build.json` only when they are part of the active experiment.

```json
{
  "loader": "direct",
  "net_id": "native",
  "batches": 64,
  "batch_size": 2048,
  "superbatches": 7600,
  "threads": 16,
  "wdl": 0.3,
  "lr": 0.001,
  "final_lr": 0.000005,
  "save_rate": 7600,
  "trainable": "all",
  "weight_decay": 0.0,
  "sfbinpack": {
    "buffer_mb": 1024,
    "offset": 0,
    "min_ply": 16,
    "max_abs_cp": 10000,
    "quiet_only": true
  },
  "validation": {
    "static_rows": 50000,
    "engine_threads": 1,
    "engine_hash_mb": 64,
    "sprt_games_smoke": 100,
    "sprt_concurrency": 16
  }
}
```

## Active Build

`build.json` describes the next candidate. It should stay small: run name,
parent, hypothesis, and the few parameters that intentionally differ from
`defaults.json`.

```json
{
  "run": "native-3.0.0-rc7",
  "continue_from": "native-3.0.0-rc5",
  "hypothesis": "farseerT74, 4 output buckets, lr: 0.0002",
  "data": {
    "source_binpack": "data/stockfish/master-binpacks/farseerT74.binpack",
    "limit": 100000000,
    "offset": 0
  }
}
```

## Build Patterns

Examples below show only the origin/reference fields. Add the normal data and
training overrides for the experiment.

Same-architecture iteration resumes the previous run checkpoint and compares
against it by default:

```json
{
  "run": "native-3.1.0-rc2",
  "continue_from": "native-3.1.0-rc1"
}
```

New architecture from existing weights converts an exported net and compares
against the parent:

```json
{
  "run": "native-4.0.0-rc1",
  "reference": "native-3.1.0-rc1",
  "initialize_from": "native-3.1.0-rc1"
}
```

New net with no existing weights omits both training origins. `./nnue` asks for
interactive confirmation before starting scratch training:

```json
{
  "run": "native-5.0.0-rc1"
}
```

## Iteration

Use `./nnue` as the wrapper for planning, training, gates, and SPRT iteration:

```sh
./nnue plan
./nnue iterate
```

nnue also supports several environment variables that affect behavior

`NNUE_AI_STDIN_EVENTS=done,fail MIN_SLOPE=0.05 SKIP_SMOKE=1 GAMES=800 ./nnue iterate`

Game results, not static metrics, decide promotion; static and move gates are
rejection filters.
