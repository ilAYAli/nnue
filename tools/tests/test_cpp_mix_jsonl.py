#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


class CppMixJsonlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.build_dir = Path(tempfile.mkdtemp(prefix="nnue_cpp_tools_"))
        subprocess.run(
            ["cmake", "-S", str(REPO), "-B", str(cls.build_dir), "-G", "Ninja"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            ["cmake", "--build", str(cls.build_dir), "--target", "nnue-mix-jsonl"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        cls.tool = cls.build_dir / "nnue-mix-jsonl"

    def test_mixes_requested_rows_and_filters_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.jsonl"
            b = root / "b.jsonl"
            out = root / "mixed.jsonl"
            a.write_text(
                "\n".join([
                    '{"fen":"8/8/8/8/8/8/8/K6k w - - 0 1","score":10}',
                    '{"fen":"8/8/8/8/8/8/8/K6k w - - 0 1","score":500}',
                    '{"fen":"8/8/8/8/8/8/8/K6k b - - 0 1","score":-20}',
                ]) + "\n",
                encoding="utf-8",
            )
            b.write_text(
                "\n".join([
                    '{"fen":"8/8/8/8/8/8/8/K6k w - - 0 1","score":30}',
                    '{"fen":"8/8/8/8/8/8/8/K6k w - - 0 1","score":40}',
                ]) + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    str(self.tool),
                    "--output", str(out),
                    "--source", f"{a}:3:100",
                    "--source", f"{b}:1",
                    "--seed", "7",
                    "--progress", "0",
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )

            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(abs(float(row["score"])) <= 100 for row in rows))

            stats = json.loads(result.stdout)
            self.assertEqual(stats["written_rows"], 3)
            self.assertEqual(stats["sources"][0]["eligible_rows"], 2)
            self.assertEqual(stats["sources"][0]["written"], 2)
            self.assertEqual(stats["sources"][1]["written"], 1)


if __name__ == "__main__":
    unittest.main()
