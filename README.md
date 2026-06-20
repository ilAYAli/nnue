# Enyo NNUE Training

Current native NNUE iteration is driven by `./nnue-run` and `build.json`.
The normal loop is:

```sh
./nnue-run status
./nnue-run plan
ITERATIONS=1 ./nnue-run iterate
```

`iterate` trains one candidate, runs gates, runs Forge/Crucible SPRT by
default, and advances `build.json` only after an accepted SPRT. It stops on a
failed gate or failed SPRT.

## Files

`build.json` is the experiment interface. For normal automatic iteration, keep
it small: `run`, current hypothesis notes, and only values that intentionally
override lane defaults for this experiment.

`architecture.json` describes the engine/runtime net shape. Changing it is an
architecture experiment, not a normal same-lane iteration.

`defaults.json` holds training defaults that are shared by runs. Do not copy
matching default values into `build.json`; override them there only when that
is the experiment.

## `build.json` Parameters

`run`: Candidate name. Automatic iteration increments names like
`pwa-native-v16` to `pwa-native-v17` after a pass.

`lineage`: Candidate family. Current native runs use `scratch-native`.

`hypothesis`: Short reason for the run. Keep it current, not historical log
spam.

`changed_variables`: Human-readable notes for the one variable family being
tested. Do not put derived reference state here.

`data.source_binpack`: Training source file. Changing this is a data experiment.

`data.limit`: Maximum rows/games consumed from the source. Increasing this adds
more data from the same source.

Optional one-run overrides: `superbatches`, `lr`, and `final_lr`. Do not
keep them in `build.json` when they match `defaults.json`.


`continue_from` optional: Parent run for non-interactive or non-standard starts.
If omitted, `nnue-run` infers the previous version from `run`.

`reference` optional: SPRT reference net/run when it should differ from
`continue_from`. If omitted, the reference is `continue_from`.

For normal iteration, omit `continue_from` and `reference` unless you explicitly
need a non-previous parent or a non-parent comparison.

## `defaults.json` Training Parameters

`loader`: Bullet data loader. Current value is `direct`.

`net_id`: Bullet checkpoint/export id. Current native lane uses
`scratch_native`.

`batches`: Batches per superbatch.

`batch_size`: Positions per batch.

`superbatches`: Default training dose. Override it in `build.json` only for
a deliberate dose experiment.

`threads`: CPU data-loading threads for training.

`wdl`: WDL target mixture. Higher values weight WDL/game-result style signal
more; lower values weight centipawn regression more.

`lr`: Default starting learning rate. Override it in `build.json` only for a
deliberate learning-rate experiment.

`final_lr`: Default final learning rate. Override it in `build.json` only for
a deliberate learning-rate experiment.

`save_rate`: Checkpoint frequency in superbatches.

`trainable`: Which weights may change. Current value `all` trains the full net.

`weight_decay`: Optimizer regularization. Current value `0.0` disables it.

`sfbinpack.buffer_mb`: Read buffer size for binpack sources.

`sfbinpack.min_ply`: Minimum ply kept from binpack data.

`sfbinpack.max_abs_cp`: Maximum absolute centipawn target kept.

`sfbinpack.quiet_only`: Whether to keep only quiet positions.

`validation.static_rows`: Rows used by static eval gate.

`validation.engine_threads`: Engine threads used by validation.

`validation.engine_hash_mb`: Engine hash size for validation.

`validation.sprt_games_smoke`: Smoke game count for non-iteration validation.

`validation.sprt_concurrency`: Default SPRT concurrency for validation helpers.

## `architecture.json` Parameters

`hidden`: L1 hidden width.

`l2_size`: Size of the small post-accumulator layer used by this native layout.

`input_buckets`: Number of input/king buckets.

`feature_channels`: Feature channels per bucket.

`output_buckets`: Number of output buckets.

`input_factoriser`: Whether input factorization is enabled.

`eval_scale`: Scale from network output to centipawns.

`l0_std`, `l1_std`: Initialization scales.

`l1_export_scale`: Export-time scale for L1 weights.

`export_format`: Engine-compatible net format.

Do not change architecture parameters during a normal continuation run. If
same-lane continuation stops gaining, change training/data first. Architecture
changes need matching engine support and parity checks.

## `nnue-run` Validation Controls

These are shell overrides, not persistent training parameters:

`ITERATIONS`: Number of accepted iterations to attempt.

`ROWS`: Rows sampled by static eval gate.

`SMOKE_GAMES`: Cheap SPRT smoke game count before the full gate. Default is
`400`.

`SMOKE_MIN_ELO`: Smoke rejection floor. Default is `-5`; below this, the run
stops before the full SPRT.

`GAMES`: Full SPRT game cap. Reaching this without H1 is a failed SPRT.

`CONCURRENCY`: Game concurrency for local `--solo` SPRT.

`THREADS`: Engine threads per game.

`SPRT_ELO0`, `SPRT_ELO1`, `SPRT_ALPHA`, `SPRT_BETA`: Passed to Forge/SPRT.
Default H1 is `SPRT_ELO1=3.0`; iteration accepts only when LLR reaches
the H1 upper bound.

`MOVE_GATE_STRICT`: Set to `1` to make move-gate regressions fail hard.

`REFERENCE_NET`: Temporary override for manual SPRT/reference testing.

`FORCE`: Set to `1` to rebuild training/export even if outputs already exist.

## When Elo Stops Improving

Change one variable family at a time. Always bump `run`. Keep `continue_from`
and `reference` omitted unless the run intentionally starts from or compares
against a non-previous net.

First same-lane retry after a flat/negative iteration, using temporary
`build.json` overrides:

```json
{
  "run": "pwa-native-v17",
  "lr": 0.0003,
  "final_lr": 0.00003
}
```

Use this when gates pass but SPRT is flat or negative. The theory is that the
current parent is already close to a local optimum and `0.001 -> 0.0001` updates
are too large for fine tuning.

If lower LR is still flat, change dose with a temporary override:

```json
{
  "run": "pwa-native-v18",
  "superbatches": 10000
}
```

Use more dose only when static metrics still look healthy and there is no clear
overfit/regression signal. Do not keep extending dose after repeated flat SPRTs.

If continuation remains flat, change data:

```json
{
  "run": "pwa-native-v19",
  "data": {
    "source_binpack": "data/nodes5000pv2_UHO.binpack",
    "limit": 150000000
  }
}
```

Changing `source_binpack`, `limit`, or binpack filtering is a data experiment.
Record the data reason in `hypothesis`/`changed_variables`.

If several same-lane training/data attempts fail, stop continuation. The next
planned architecture lever is output buckets, then input buckets, then L2/L3.
Those are not build.json-only tweaks; they require engine/runtime parity.

## Rejection Rules

Reject immediately if export or engine parity fails, the engine does not load
the intended `.nn`, move-gate coverage is incomplete, static eval is obviously
broken, smoke SPRT is below the rejection floor, or full SPRT does not pass H1
before the configured game cap.

Static metrics are rejection filters only. Elo comes from games.
