#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "bullet"))

from lib import bullet_text  # noqa: E402
import jsonl_to_bullet_text  # noqa: E402


class BulletTextZeroScoreTests(unittest.TestCase):
    def test_white_score_from_row_zero_score_ignores_search_eval(self) -> None:
        row = {"fen": "8/8/8/8/8/8/8/K6k w - - 0 1", "score": 900}
        self.assertEqual(
            bullet_text.white_score_from_row(row, enyo_runtime_target=False, zero_score=True),
            0,
        )

    def test_white_score_from_row_zero_score_works_without_score_field(self) -> None:
        row = {"fen": "8/8/8/8/8/8/8/K6k w - - 0 1"}
        self.assertEqual(
            bullet_text.white_score_from_row(row, enyo_runtime_target=False, zero_score=True),
            0,
        )

    def test_row_to_text_zero_score_preserves_result(self) -> None:
        row = {"fen": "8/8/8/8/8/8/8/K6k b - - 0 1", "score": -350, "result": "1-0"}
        text = bullet_text.row_to_text(row, enyo_runtime_target=False, zero_score=True)
        self.assertEqual(text, "8/8/8/8/8/8/8/K6k b - - 0 1 | 0 | 1.0")

    def test_convert_zero_score_end_to_end(self) -> None:
        rows = [
            {"fen": "8/8/8/8/8/8/8/K6k w - - 0 1", "score": 900, "result": "1-0"},
            {"fen": "8/8/8/8/8/8/8/K6k b - - 0 1", "score": -900, "result": "0-1"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "in.jsonl"
            dst = root / "out.txt"
            src.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

            stats = jsonl_to_bullet_text.convert(
                src,
                dst,
                limit=0,
                max_abs_cp=1600,
                enyo_runtime_target=False,
                zero_score=True,
            )

            lines = dst.read_text(encoding="utf-8").splitlines()
            self.assertEqual(stats["written"], 2)
            self.assertEqual(stats["target"], "zero-score")
            self.assertEqual(lines[0], "8/8/8/8/8/8/8/K6k w - - 0 1 | 0 | 1.0")
            self.assertEqual(lines[1], "8/8/8/8/8/8/8/K6k b - - 0 1 | 0 | 0.0")


if __name__ == "__main__":
    unittest.main()
