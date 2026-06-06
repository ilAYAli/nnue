#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
PACK = REPO / "tools" / "pack" / "pack_dataset.py"
FEN = "8/8/8/8/8/8/8/K6k w - - 0 1"
sys.path.insert(0, str(REPO / "tools"))

from lib import packed_labels  # noqa: E402


def write_rows(path: Path) -> None:
    rows = [
        {"fen": FEN, "score": 10, "wdl": 0.5, "source_type": "test"},
        {},
        {"fen": FEN, "score": -20, "wdl": 0.5, "source_type": "test"},
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(rows[0]) + "\n")
        handle.write("\n")
        handle.write(json.dumps(rows[2]) + "\n")


class PackDatasetTests(unittest.TestCase):
    def test_rows_file_overcount_from_blank_lines_shrinks_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "rows.jsonl"
            rows_file = root / "rows.wc"
            out = root / "pack"
            write_rows(src)
            rows_file.write_text(f"3 {src}\n", encoding="utf-8")

            subprocess.run([
                sys.executable, str(PACK),
                "--input", str(src),
                "--out-dir", str(out),
                "--rows-file", str(rows_file),
                "--progress", "0",
            ], check=True, cwd=REPO)

            meta = json.loads((out / "meta.json").read_text())
            self.assertEqual(meta["rows"], 2)
            self.assertEqual(np.load(out / "counts.npy").shape, (2,))
            self.assertEqual(np.load(out / "white_features.npy").shape, (2, 32))

    def test_rows_file_undercount_still_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "rows.jsonl"
            rows_file = root / "rows.wc"
            out = root / "pack"
            write_rows(src)
            rows_file.write_text(f"1 {src}\n", encoding="utf-8")

            result = subprocess.run([
                sys.executable, str(PACK),
                "--input", str(src),
                "--out-dir", str(out),
                "--rows-file", str(rows_file),
                "--progress", "0",
            ], cwd=REPO, text=True, capture_output=True)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("row count exceeded", result.stderr)

    def test_merges_packed_label_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = root / "shards"
            out = root / "pack"
            packed_labels.write_shard(
                shards / "label.0.npz",
                [{"fen": FEN, "score": 10, "wdl": 0.5, "source_type": "a"}],
                max_features=32,
            )
            packed_labels.write_shard(
                shards / "label.1.npz",
                [{"fen": FEN, "score": -20, "wdl": 0.25, "source_type": "b"}],
                max_features=32,
            )

            subprocess.run([
                sys.executable, str(PACK),
                "--packed-shard-dir", str(shards),
                "--out-dir", str(out),
                "--progress", "0",
            ], check=True, cwd=REPO)

            meta = json.loads((out / "meta.json").read_text())
            self.assertEqual(meta["format"], "enyo-packed-label-shard-v1")
            self.assertEqual(meta["rows"], 2)
            self.assertEqual(meta["source_map"], {"a": 0, "b": 1})
            self.assertEqual(np.load(out / "counts.npy").tolist(), [2, 2])
            self.assertEqual(np.load(out / "score.npy").tolist(), [10.0, -20.0])
            self.assertEqual(np.load(out / "source_id.npy").tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
