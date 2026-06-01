#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "posgen"))

import lc0_value_to_score  # noqa: E402


class Lc0ValueToScoreTests(unittest.TestCase):
    def test_converts_root_q_to_score(self) -> None:
        row = {
            "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
            "lc0": {"root_q": 0.5},
            "tags": ["source:lc0"],
        }

        out = lc0_value_to_score.convert_row(
            row, q_field="root_q", scale=300.0, clamp=0.999,
            invert=False)

        self.assertEqual(out["score"], int(round(300.0 * math.atanh(0.5))))
        self.assertEqual(out["wdl"], 0.75)
        self.assertEqual(out["teacher"], "lc0_root_q")
        self.assertIn("label:lc0_root_q", out["tags"])
        self.assertIn("score:lc0_value", out["tags"])

    def test_preserves_existing_score_as_source_score(self) -> None:
        row = {
            "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
            "score": -12,
            "lc0": {"root_q": 0.0},
        }

        out = lc0_value_to_score.convert_row(
            row, q_field="root_q", scale=300.0, clamp=0.999,
            invert=False)

        self.assertEqual(out["score"], 0)
        self.assertEqual(out["source_score"], -12)

    def test_clamps_q_before_atanh(self) -> None:
        score = lc0_value_to_score.q_to_cp(1.0, scale=100.0, clamp=0.9)
        self.assertEqual(score, int(round(100.0 * math.atanh(0.9))))

    def test_stream_converter_skips_large_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "in.jsonl"
            dst = root / "out.jsonl"
            rows = [
                {"fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                 "lc0": {"root_q": 0.1}},
                {"fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                 "lc0": {"root_q": 0.99}},
            ]
            src.write_text("".join(json.dumps(row) + "\n" for row in rows),
                           encoding="utf-8")

            stats = lc0_value_to_score.convert(
                src, dst, q_field="root_q", scale=300.0, clamp=0.999,
                invert=False, max_abs_cp=100, limit=0, progress=0)

            self.assertEqual(stats["written"], 1)
            self.assertEqual(stats["skipped_cp"], 1)
            self.assertEqual(len(dst.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
