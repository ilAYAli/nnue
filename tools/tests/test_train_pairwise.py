#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import train_pairwise


class TrainPairwiseMetadataTests(unittest.TestCase):
    def test_metadata_records_init_data_pairs_and_label_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "pack" / "train"
            data.mkdir(parents=True)
            data.joinpath("source_map.json").write_text(
                json.dumps({"stockfish": 0}) + "\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                data=str(data),
                pairs="/runs/replay/replay_full_pairs.jsonl",
                init_from_nn="/runs/base/model.nn",
                init="kaiming",
                epochs=12,
                batch_size=8192,
                pair_batch_size=64,
                lr=3e-5,
                weight_decay=1e-6,
                huber_beta=200.0,
                pair_beta=100.0,
                pair_weight=5.0,
                broad_weight=1.0,
                target_clamp=800.0,
                max_target_margin=800.0,
                min_target_margin=1.0,
                loss_weight_by_cp=True,
                project_export_weights_each_step=True,
                max_rows=200000,
                skip_rows=0,
            )

            metadata = train_pairwise.pairwise_metadata(args, epoch=1)

            self.assertEqual(metadata["schema"], "enyo.train_pairwise.v1")
            self.assertEqual(metadata["epoch"], 1)
            self.assertEqual(metadata["init_from_nn"], "/runs/base/model.nn")
            self.assertEqual(metadata["data"], str(data))
            self.assertEqual(metadata["pairs"], "/runs/replay/replay_full_pairs.jsonl")
            self.assertEqual(metadata["label_refs"], ["enyo", "stockfish"])
            self.assertIn(str(data), metadata["position_refs"])
            self.assertIn("/runs/replay/replay_full_pairs.jsonl", metadata["position_refs"])
            self.assertTrue(metadata["project_export_weights_each_step"])


if __name__ == "__main__":
    unittest.main()
