# Enyo NNUE Training

This repo contains the training pipeline, experiment docs, and helper tools for
building Enyo NNUE networks.

The engine remains in the sibling repo:

```text
../enyo
```

Useful entry points:

- `README_nnue_training.html`: explanation of the full pipeline.
- `TRAINING_RUNBOOK.md`: practical command guide for starting and judging runs.
- `tools/nnue2/train_new_net_pwa.sh`: guarded one-command launcher for pwa-5090.
- `tools/nnue2/`: import, label, pack, train, export, and validation scripts.
- `tools/nnue2/run_selflichess_mix_pwa.sh`: current self-play/Lichess experiment runner.
- `tools/nnue2/run_huber_cp800_neighbors_pwa.sh`: short SPRT screens around
  the best currently observed Huber cp800 recipe.

Large datasets, packed tensors, checkpoints, PGNs, and run directories should
stay outside git under `~/tmp/` or `~/code/cpp/chess/assets/`.
