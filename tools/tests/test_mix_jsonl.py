#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "posgen" / "mix_jsonl.py"


class MixJsonlTests(unittest.TestCase):
    def test_source_spec_can_filter_by_abs_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            output = root / "mixed.jsonl"
            source.write_text(
                "\n".join([
                    json.dumps({"fen": "a", "score": 10}),
                    json.dumps({"fen": "b", "score": 900}),
                    json.dumps({"fen": "c", "score": -800}),
                    json.dumps({"fen": "d", "score": 801}),
                    json.dumps({"fen": "e"}),
                    "not json",
                    "",
                ]) + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable, str(TOOL),
                    "--output", str(output),
                    "--source", f"{source}:10:800",
                    "--seed", "7",
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(["a", "c"], [row["fen"] for row in rows])

            stats = json.loads(result.stdout)
            source_stats = stats["sources"][0]
            self.assertEqual(6, source_stats["total_rows"])
            self.assertEqual(2, source_stats["eligible_rows"])
            self.assertEqual(4, source_stats["filtered_rows"])
            self.assertEqual(2, source_stats["written"])


if __name__ == "__main__":
    unittest.main()
