<p align="center">
  <img src="https://github.com/user-attachments/assets/50d0d944-a763-44c9-8e9c-af4112c2fc14" alt="Enyo logo">
</p>

# Enyo NNUE

This repository trains Enyo-native NNUE networks for the
[Enyo](https://github.com/ilAYAli/enyo) chess engine.

The nets are trained from scratch for Enyo. No foreign weights, tensors, or
NNUE parameters are used.

## Evaluation

Enyo NNUE line uses a HalfKAv2-style factorised network trained with Bullet.
Training data comes from Enyo self-play/generated positions and binpack datasets,
with Stockfish used only as a labeling oracle where the dataset requires it.

## Provenance

Some training datasets come from `official-stockfish/master-binpacks`, licensed
under ODbL-1.0. Published nets trained from those datasets should include an
ODbL provenance notice.

Training uses the local `tools/bullet/spike_trainer` wrapper around
[Bullet](https://github.com/jw1912/bullet), pinned at commit
`d372d487aedfeb8bdc256b9f694dbcd41016bf82`. Bullet is MIT licensed.

## Configuration

- `architecture.json`: promoted network shape and export/runtime contract.
- `defaults.json`: shared training defaults.
- `build.json`: the active experiment only.

`build.json` should stay small: run name, parent/reference, hypothesis, data,
and the few parameters intentionally changed for the next candidate.

## Iteration

Normal training and validation runs through `./nnue`:

```sh
./nnue plan
./nnue iterate
```

The long-running automated loop is normally launched from `nnue_cmd` on
`pwa-llm`:

```sh
NNUE_HOOK_EVENTS=done,fail MIN_SLOPE=0.05 SKIP_SMOKE=1 GAMES=5000 ./nnue iterate
```

Game results decide promotion. Static evaluation, startpos, move, and
Stockfish-net checks are rejection filters.

## Validation

The primary absolute benchmark is the Stockfish net
`nn-1a298aa575a0.nnue`

Use the net SPRT helper for fixed net comparisons:

```sh
tools/validate/sprt_net.py --candidate ~/assets/nets/candidate.net
tools/validate/sprt_net.py --candidate ~/assets/nets/enyo-1.20.0-rc12.nn --reference ~/assets/nets/nn-1a298aa575a0.nnue
```

## Data

Large corpora live under `data/` and are not duplicated casually. Prefer
streaming or existing Bullet files over decompressed intermediates.

Generated run artifacts live under `runs/`; promoted nets are copied to
`~/assets/nets/` by the training loop.

## Utilities

- [Forge](https://github.com/ilAYAli/Forge): distributes SPRT and validation
  jobs across the worker fleet.
- [Replay](https://github.com/ilAYAli/replay): analyzes games and failure
  suites.
- [Fastchess](https://github.com/Disservin/fastchess): game runner used under
  Forge SPRT jobs.
