#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.lib.nnue_dataset import (
    BulletDataScoreDataset,
    collate_packed,
    count_rows,
    load_score_dataset,
)


class NNueDatasetTests(unittest.TestCase):
    def test_collate_packed_reuses_shared_offsets(self) -> None:
        batch = [
            (
                np.array([1, 2, 0, 0], dtype=np.uint16),
                np.array([5, 6, 0, 0], dtype=np.uint16),
                np.uint8(2),
                np.uint8(0),
                np.float32(10.0),
                np.float32(0.5),
                np.float32(1.0),
                np.uint16(0),
            ),
            (
                np.array([3, 0, 0, 0], dtype=np.uint16),
                np.array([7, 0, 0, 0], dtype=np.uint16),
                np.uint8(1),
                np.uint8(1),
                np.float32(-20.0),
                np.float32(0.25),
                np.float32(1.1),
                np.uint16(2),
            ),
        ]

        w, b, w_offsets, b_offsets, *_rest = collate_packed(batch)

        self.assertIs(w_offsets, b_offsets)
        np.testing.assert_array_equal(w_offsets, np.asarray([0, 2]))
        np.testing.assert_array_equal(w, np.asarray([1, 2, 3]))
        np.testing.assert_array_equal(b, np.asarray([5, 6, 7]))


    def test_loads_bullet_records(self) -> None:
        from tools.lib import bullet_format

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "rows.bullet"
            row = {
                "fen": "8/8/8/8/8/8/8/K6k b - - 0 1",
                "score": 42,
                "wdl": 0.25,
                "result": "1-0",
            }
            with data.open("wb") as handle:
                bullet_format.write_row(
                    handle,
                    row,
                    enyo_runtime_target=False,
                )

            dataset, collate = load_score_dataset(data)
            self.assertIsInstance(dataset, BulletDataScoreDataset)
            self.assertEqual(count_rows(data), 1)
            batch = collate([dataset[0]])
            (_w, _b, _w_off, _b_off, counts, stm, scores,
             wdls, _phase, source_ids) = batch

            self.assertEqual(counts.tolist(), [2])
            self.assertEqual(stm.tolist(), [0])
            self.assertEqual(scores.tolist(), [42.0])
            self.assertEqual(wdls.tolist(), [0.0])
            self.assertEqual(source_ids.tolist(), [0])

    def test_rejects_deprecated_data_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "rows.data"
            data.write_bytes(b"")

            with self.assertRaisesRegex(ValueError, "deprecated BulletFormat extension"):
                count_rows(data)
            with self.assertRaisesRegex(ValueError, "deprecated BulletFormat extension"):
                load_score_dataset(data)



if __name__ == "__main__":
    unittest.main()
