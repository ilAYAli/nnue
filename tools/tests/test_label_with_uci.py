#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


class LabelWithUciTests(unittest.TestCase):
    def test_scores_rows_with_multiple_uci_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = root / "fake_engine.py"
            engine.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env python3
                    import sys

                    for raw in sys.stdin:
                        line = raw.strip()
                        if line == "uci":
                            print("uciok", flush=True)
                        elif line == "isready":
                            print("readyok", flush=True)
                        elif line.startswith("go "):
                            print("info depth 1 score cp 42 nodes 1", flush=True)
                            print("bestmove e2e4", flush=True)
                        elif line == "quit":
                            break
                    """
                ),
                encoding="utf-8",
            )
            engine.chmod(0o755)

            input_path = root / "input.jsonl"
            rows = [
                {
                    "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                    "score": 0,
                    "id": idx,
                }
                for idx in range(5)
            ]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            output_path = root / "output.jsonl"

            subprocess.run(
                [
                    "python3",
                    str(Path(__file__).resolve().parents[1] / "score" / "label_with_uci.py"),
                    "--input", str(input_path),
                    "--output", str(output_path),
                    "--engine", str(engine),
                    "--depth", "1",
                    "--concurrency", "2",
                    "--progress", "0",
                ],
                check=True,
            )

            out_rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(5, len(out_rows))
            self.assertEqual([row["id"] for row in rows], [row["id"] for row in out_rows])
            self.assertTrue(all(row["score"] == 42 for row in out_rows))
            self.assertTrue(all(row["source_score"] == 0 for row in out_rows))

            stats = json.loads(
                (root / "output.jsonl.stats.json").read_text(encoding="utf-8")
            )
            self.assertEqual(2, stats["concurrency"])
            self.assertEqual(5, stats["selected"])
            self.assertEqual(5, stats["processed"])
            self.assertEqual(5, stats["written"])


if __name__ == "__main__":
    unittest.main()
