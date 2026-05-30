#!/usr/bin/env python3
from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.lib.nnue_dataset import collate_packed


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
        torch.testing.assert_close(w_offsets, torch.tensor([0, 2]))
        torch.testing.assert_close(w, torch.tensor([1, 2, 3]))
        torch.testing.assert_close(b, torch.tensor([5, 6, 7]))


if __name__ == "__main__":
    unittest.main()
