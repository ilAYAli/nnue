#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


class CppJsonlToBulletTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.build_dir = Path(tempfile.mkdtemp(prefix="nnue_cpp_tools_"))
        subprocess.run(
            ["cmake", "-S", str(REPO), "-B", str(cls.build_dir), "-G", "Ninja"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            ["cmake", "--build", str(cls.build_dir), "--target", "nnue-jsonl-to-bullet-text"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        cls.tool = cls.build_dir / "nnue-jsonl-to-bullet-text"

    def test_converts_side_to_move_scores_and_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "rows.jsonl"
            out = root / "bullet.txt"
            src.write_text(
                "\n".join([
                    '{"fen":"8/8/8/8/8/8/8/K6k w - - 0 1","score":25,"wdl":0.8}',
                    '{"fen":"8/8/8/8/8/8/8/K6k b - - 0 1","score":30,"wdl":0.8}',
                    '{"fen":"8/8/8/8/8/8/8/K6k w - - 0 1","score":5000,"wdl":0.5}',
                    '{"bad": true}',
                ]) + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    str(self.tool),
                    "--input", str(src),
                    "--output", str(out),
                    "--max-abs-cp", "100",
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )

            self.assertEqual(out.read_text(encoding="utf-8").splitlines(), [
                "8/8/8/8/8/8/8/K6k w - - 0 1 | 25 | 1.0",
                "8/8/8/8/8/8/8/K6k b - - 0 1 | -30 | 0.0",
            ])
            stats = json.loads(result.stdout)
            self.assertEqual(stats["read"], 4)
            self.assertEqual(stats["written"], 2)
            self.assertEqual(stats["skipped_cp"], 1)
            self.assertEqual(stats["skipped_bad"], 1)


if __name__ == "__main__":
    unittest.main()
