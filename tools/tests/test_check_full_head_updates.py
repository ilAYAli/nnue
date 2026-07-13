from __future__ import annotations

import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools/validate/check_full_head_updates.py"


def write_weights(path: Path, tensors: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        for name, values in tensors.items():
            flat = np.asarray(values, dtype=np.float32).ravel()
            handle.write(name.encode() + b"\n")
            handle.write(struct.pack("<Q", flat.size))
            handle.write(flat.tobytes())


class FullHeadUpdateTests(unittest.TestCase):
    def tensors(self, buckets: int) -> dict[str, np.ndarray]:
        return {
            "l0w": np.zeros(4, dtype=np.float32),
            "l0b": np.zeros(2, dtype=np.float32),
            "l1w": np.zeros((2048, buckets, 16), dtype=np.float32),
            "l2w": np.zeros((16, buckets, 32), dtype=np.float32),
            "l3w": np.zeros((32, buckets), dtype=np.float32),
        }

    def run_check(self, dead_bucket: int | None) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            before = self.tensors(2)
            after = {name: values.copy() for name, values in before.items()}
            after["l1w"] += 1
            after["l2w"] += 1
            after["l3w"] += 1
            if dead_bucket is not None:
                after["l1w"][:, dead_bucket, :] = 0
            write_weights(tmp / "before.bin", before)
            write_weights(tmp / "after.bin", after)
            return subprocess.run(
                [sys.executable, str(SCRIPT), "--initial", str(tmp / "before.bin"),
                 "--trained", str(tmp / "after.bin"), "--output-buckets", "2",
                 "--expect-frozen-input"],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_accepts_updates_in_every_head_with_frozen_input(self) -> None:
        self.assertEqual(0, self.run_check(None).returncode)

    def test_rejects_dead_head(self) -> None:
        proc = self.run_check(1)
        self.assertNotEqual(0, proc.returncode)
        self.assertIn("l1w material bucket 1 did not update", proc.stderr)


if __name__ == "__main__":
    unittest.main()
