#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "bullet"))

from lib import bullet_text  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
