#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "score"))

import label_with_uci  # noqa: E402


class FakeEngine:
    def __init__(self, path: str, *, threads: int, hash_mb: int) -> None:
        self.path = path

    def label(self, fen: str, *, depth: int) -> tuple[int | None, str | None]:
        return 42, None

    def close(self) -> None:
        pass


class LabelWithUciTests(unittest.TestCase):
    def run_labeler(self, row: dict) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "in.jsonl"
            dst = root / "out.jsonl"
            src.write_text(json.dumps(row) + "\n", encoding="utf-8")

            old_argv = sys.argv
            old_engine = label_with_uci.UciEngine
            try:
                label_with_uci.UciEngine = FakeEngine  # type: ignore[assignment]
                sys.argv = [
                    "label_with_uci.py",
                    "--input", str(src),
                    "--output", str(dst),
                    "--engine", "/tmp/stockfish",
                    "--depth", "12",
                    "--progress", "0",
                ]
                label_with_uci.main()
            finally:
                sys.argv = old_argv
                label_with_uci.UciEngine = old_engine  # type: ignore[assignment]

            return json.loads(dst.read_text(encoding="utf-8"))

    def test_labels_rows_without_existing_score(self) -> None:
        row = self.run_labeler({
            "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
            "source": "lc0_training_data",
        })

        self.assertEqual(row["score"], 42)
        self.assertNotIn("source_score", row)
        self.assertEqual(row["teacher"], "stockfish")
        self.assertEqual(row["teacher_depth"], 12)

    def test_preserves_existing_score_as_source_score(self) -> None:
        row = self.run_labeler({
            "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
            "score": -17,
            "source": "enyo",
        })

        self.assertEqual(row["score"], 42)
        self.assertEqual(row["source_score"], -17)


