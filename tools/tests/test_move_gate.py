#!/usr/bin/env python3
from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
BUILDER = REPO / "tools" / "validate" / "build_fixed_move_gate.py"


class MoveGateTests(unittest.TestCase):
    def test_child_targets_build_hardest_legal_case(self) -> None:
        if importlib.util.find_spec("chess") is None:
            self.skipTest("python-chess is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "child.jsonl"
            out = root / "gate.jsonl"
            summary = root / "summary.txt"
            src.write_text(json.dumps({
                "id": "row1",
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "best_move": "e2e4",
                "moves": [
                    {"move": "e2e4", "score_cp": 20, "role": "best", "rank": 1},
                    {"move": "d2d4", "score_cp": 10, "role": "neighbor", "rank": 2},
                    {"move": "e2e3", "score_cp": -30, "role": "neighbor", "rank": 3},
                ],
                "source": "unit",
                "score_pov": "parent",
                "tags": ["unit"],
            }, sort_keys=True) + "\n", encoding="utf-8")

            subprocess.run([
                sys.executable, str(BUILDER),
                "--child-targets", f"unit={src}",
                "--output", str(out),
                "--summary", str(summary),
                "--min-gap-cp", "20",
                "--max-per-parent", "1",
                "--min-cases", "1",
            ], cwd=REPO, check=True)

            rows = [
                json.loads(line)
                for line in out.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["schema"], "enyo.move_choice_gate.v1")
            self.assertEqual(rows[0]["best"], "e2e4")
            self.assertEqual(rows[0]["played"], "e2e3")
            self.assertEqual(rows[0]["loss_cp"], 50)
            self.assertIn("cases=1", summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
