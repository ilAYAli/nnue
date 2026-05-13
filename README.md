# Enyo NNUE Training

This repo contains the training pipeline, experiment docs, and helper tools for
building Enyo NNUE networks.

The engine remains in the sibling repo:

```text
../enyo
```

Useful entry points:

- `README_nnue_training.html`: explanation of the full pipeline.
- `tools/nnue2/`: import, label, pack, train, export, and validation scripts.
- `tools/nnue2/run_source_mix_1m_pwa.sh`: current source-mix experiment runner.

Large datasets, packed tensors, checkpoints, PGNs, and run directories should
stay outside git under `~/tmp/` or `~/code/cpp/chess/assets/`.
