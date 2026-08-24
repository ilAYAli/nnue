#!/usr/bin/env python3
"""Regression tests for LC0 V6 value-head conversion."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.posgen import lc0_to_jsonl  # noqa: E402


class ResultExpectedScoreTests(unittest.TestCase):
    def test_draw_head_does_not_turn_a_draw_into_a_loss(self) -> None:
        self.assertEqual(lc0_to_jsonl.result_expected_score(0.0, 1.0), 0.5)

    def test_result_q_encodes_expected_game_score(self) -> None:
        self.assertEqual(lc0_to_jsonl.result_expected_score(1.0, 0.0), 1.0)
        self.assertEqual(lc0_to_jsonl.result_expected_score(-1.0, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
