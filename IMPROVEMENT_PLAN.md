# Enyo NNUE Improvement Plan

## Goal

Build a fully traceable Enyo-native lineage that achieves positive Elo against
`nn-0ee0657fb25e.nnue` under fixed Forge conditions.

Historical nets, including `enyo-1.32.0-rc10`, are benchmarks only and are not
ancestors of the new lineage.

## Founding architecture

`enyo-5.0.0-rc1` is the sole lineage root:

- SCReLU Enyo-native architecture
- random initialization
- no `continue_from`
- no `initialize_from`
- identified, preserved training data

Its exact configuration, data identity, net hash, and SF result establish the
reproducible baseline.

## Iteration

After the root, every candidate:

- uses `continue_from` the current proven parent;
- changes exactly one meaningful variable;
- receives a globally reserved run number;
- caps parent-relative SPRT at 5,000 games;
- is committed whether accepted or rejected.

The root and every passing candidate receive an identical-condition benchmark
against the fixed SF net. A 5,000-game parent-relative SPRT selects parents;
shorter matches are screening only. Absolute SF Elo tracks progress toward the
goal. Residual improvement is report-only.

Promotion is simple: H0 rejects and H1 selects early. If neither is reached by
the 5,000-game cap, select the stronger fixed-SF result under identical
conditions; retain the incumbent if neither is better. H1 is sufficient, never
required.

## Parallel training

`pwa-llm` owns canonical history and promotion decisions. `pwa-5090` trains
globally reserved sibling candidates. Completed remote experiments are committed
locally, preserved with complete provenance, and integrated into canonical
`main` only when `pwa-llm` reaches an event boundary. Promotion waits for every
in-flight sibling from the same parent. A winning remote net becomes a parent
only after its artifacts and hashes are verified on `pwa-llm`.
All Forge coordination runs on `pwa-llm`; `pwa-5090` is training and worker-only.

## Priorities

The selected parent is `enyo-7.4.0-rc1`. It already uses cubic loss, AdamW
beta1=0.95, SCReLU, an activated L2 skip, and 32 input buckets. Do not retest
those as new ideas.

Its selected path has consumed roughly 386B examples, predominantly from the
combined corpus and then the 2B recalibrated nodes5000pv2 corpus. Further
changes to dose, WDL, LR, or final LR from this parent have not improved Elo.
The 2B calibration is statistically indistinguishable from the successful
600M Enyo-1.32 subset, so recalibration is not the remaining variable.

1. Run one data-only continuation from `enyo-7.4.0-rc1` on the preserved,
   traceable Stockfish Pylon binpack:
   `~/assets/training/stockfish/master-binpacks/training_data_pylon.binpack`.
   Keep the selected parent architecture and every training setting unchanged;
   change only `data.source_binpack`. Pylon was the broad-data bootstrap of
   the recovered strong Enyo-1 path, so this isolates position distribution.
2. If it wins, benchmark it against fixed SF and use it as the sole parent for
   the next explicit data slice. If it loses, do not spend further runs on
   WDL, LR, dose, final LR, FullThreats, pawn pairs, x-rays, or padded-width
   conversions without new evidence.
3. In parallel with no training, audit Pylon and nodes5000pv2 sampling and
   targets. The goal is to identify a concrete data distinction before another
   data run.
4. Only then consider a true compact-topology project. Rice is a genuine
   16-bucket, 512-wide, direct 1024→1 evaluator; `enyo-14` was not equivalent
   because Enyo pads its trained 512 columns into a fixed 1024-wide runtime and
   retained the Enyo-7 dense tail. A valid Rice-style control needs native
   runtime/export support and parity tests before a training run. Never use
   foreign weights.

## Architecture fallback

After four consecutive valid candidates from one parent fail, stop ordinary
hyperparameter tuning. Void runs and infrastructure failures do not count; an
accepted candidate resets the count.

At that boundary, `pwa-5090` may train the next feature via `initialize_from`
while `pwa-llm` tests one final conventional candidate via `continue_from`.
Both branch from the same accepted parent and must finish before parent selection.

Architecture work must start with an implementation and parity audit, then one
feature/topology at a time. The activated L2 skip and 32 input buckets are
already in the selected architecture. FullThreats, pawn pairs, unfactorised
inputs, independent heads, ordinary-ReLU tail, 16 buckets, and padded 512
training have all lost. FullThreats remains closed unless its added rows show
meaningful exported nonzero coverage.

Each architecture item receives a new architecture number and uses
`initialize_from` only when the conversion is supported and preserves the
intended topology. Never combine features merely to imitate another engine.

Earlier FullThreats and x-ray results are not conclusive feature tests. Their
new rows learned sub-quantization float weights that were almost entirely erased
on export; the FullThreats net retained only 16 nonzero values among 61,243,392
threat weights. Do not spend games on another feature net unless its added rows
receive updates and retain meaningful nonzero coverage after export.

Detailed immutable ancestry and results belong in `LINEAGE.md`, not here.
Procedural rules belong in `AGENTS.md` and the repository skills.
