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


class CategoricalResultTests(unittest.TestCase):
    def test_terminal_heads_decode_to_white_result_for_both_sides(self) -> None:
        for side, expected in (("w", "1-0"), ("b", "0-1")):
            self.assertEqual(
                expected,
                lc0_to_jsonl.categorical_result(1.0, 0.0, side_to_move=side),
            )
        for side, expected in (("w", "0-1"), ("b", "1-0")):
            self.assertEqual(
                expected,
                lc0_to_jsonl.categorical_result(-1.0, 0.0, side_to_move=side),
            )
        for side in ("w", "b"):
            self.assertEqual(
                "1/2-1/2",
                lc0_to_jsonl.categorical_result(0.0, 1.0, side_to_move=side),
            )

    def test_non_categorical_terminal_heads_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not categorical"):
            lc0_to_jsonl.categorical_result(0.2, 0.3, side_to_move="w")


if __name__ == "__main__":
    unittest.main()
