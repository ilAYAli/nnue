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
- requires a 5,000-game parent-relative SPRT for promotion;
- is committed whether accepted or rejected.

The root and every passing candidate receive an identical-condition benchmark
against the fixed SF net. A 5,000-game parent-relative SPRT selects parents;
shorter matches are screening only. Absolute SF Elo tracks progress toward the
goal. Residual improvement is report-only.

## Parallel training

`pwa-llm` owns canonical history and promotion decisions. `pwa-5090` trains
globally reserved sibling candidates. Completed remote experiments are committed
locally, preserved with complete provenance, and integrated into canonical
`main` only when `pwa-llm` reaches an event boundary. Promotion waits for every
in-flight sibling from the same parent. A winning remote net becomes a parent
only after its artifacts and hashes are verified on `pwa-llm`.
All Forge coordination runs on `pwa-llm`; `pwa-5090` is training and worker-only.

## Priorities

1. Establish the SCReLU scratch root.
2. Improve training dose and data while holding architecture fixed.
3. Test objective and optimizer parameters individually.
4. Change architecture only after training variables are understood.
5. Continue until the lineage statistically beats the fixed SF net.

## Architecture fallback

After four consecutive valid candidates from the same parent fail to gain Elo,
stop tuning ordinary training parameters. Void runs and infrastructure failures
do not count; an accepted candidate resets the count.

At that boundary, `pwa-5090` may train the next feature via `initialize_from`
while `pwa-llm` tests one final conventional candidate via `continue_from`.
Both branch from the same accepted parent and must finish before parent selection.

Test one feature at a time, in this order:

1. Add an activated-L2-to-output skip connection. Skip connections are used by
   Stockfish, Stormphrax, PlentyChess, and viridithas and add little inference
   or parameter cost to Enyo's existing 16→32 dense stack.
2. Add FullThreats inputs with verified trainer/runtime parity and
   export-visible weights, against a matched unfactorised control. Threat
   features are common in the strongest current architectures, but Enyo must
   first prove that the added rows train and survive export.
3. Add pawn-pair inputs under the same parity and export-visibility gates.
   They are used by Stockfish, Stormphrax, PlentyChess, and viridithas and are
   cheaper and more structured than another large bucketed feature expansion.
4. Use independent dense heads per output bucket. This is common, but Enyo's
   earlier full-head experiment failed, so retry only after the higher-priority
   architecture changes have a stable training control.
5. Add slider x-ray threat inputs after ordinary FullThreats has demonstrated
   useful, export-visible learning. X-rays are an extension of the threat
   family, not the first threat experiment.
6. Increase L2 width from 16 to 32. Stockfish uses 32→32, but Alexandria,
   Berserk, Obsidian, PlentyChess, and Reckless use 16→32; widening L2 is not
   the prevailing small-engine architecture.
7. Increase input buckets from 16 to 32. This is principally a Stockfish
   design choice and has a much larger parameter cost than the features above.
8. Increase feature channels from 12 to 16 only with a specific semantic
   channel design. A bare channel-count increase has no strong survey support.

Each item is a separate architecture number and uses `initialize_from` the
current accepted parent when conversion is supported. Never combine features.
The historical architecture race fixed `l2_size=16`, so it did not test item 6.
Its `sf32` candidate changed both input buckets (`16` to `32`) and feature
channels (`12` to `11`); replicated continuation was not significantly better
than `h16`, so another input-bucket experiment is low priority.

Earlier FullThreats and x-ray results are not conclusive feature tests. Their
new rows learned sub-quantization float weights that were almost entirely erased
on export; the FullThreats net retained only 16 nonzero values among 61,243,392
threat weights. Do not spend games on another feature net unless its added rows
receive updates and retain meaningful nonzero coverage after export.

Detailed immutable ancestry and results belong in `LINEAGE.md`, not here.
Procedural rules belong in `AGENTS.md` and the repository skills.
