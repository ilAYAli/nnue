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
- uses parent-relative SPRT for promotion;
- is committed whether accepted or rejected.

The root and every passing candidate receive an identical-condition benchmark
against the fixed SF net. Parent-relative SPRT selects parents; absolute SF Elo
tracks progress toward the goal. Residual improvement is report-only.

## Parallel training

`pwa-llm` owns canonical history and promotion decisions. `pwa-5090` trains
globally reserved sibling candidates. Completed remote experiments are committed
locally, preserved with complete provenance, and integrated into canonical
`main` only when `pwa-llm` reaches an event boundary. Promotion waits for every
in-flight sibling from the same parent. A winning remote net becomes a parent
only after its artifacts and hashes are verified on `pwa-llm`.

## Priorities

1. Establish the SCReLU scratch root.
2. Improve training dose and data while holding architecture fixed.
3. Test objective and optimizer parameters individually.
4. Change architecture only after training variables are understood.
5. Continue until the lineage statistically beats the fixed SF net.

Detailed immutable ancestry and results belong in `LINEAGE.md`, not here.
Procedural rules belong in `AGENTS.md` and the repository skills.
