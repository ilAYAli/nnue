#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PIPELINE_PATH = REPO / "tools" / "pipeline" / "pipeline.py"
SPEC = importlib.util.spec_from_file_location("pipeline", PIPELINE_PATH)
assert SPEC is not None
pipeline = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pipeline)


class PipelineStatusTests(unittest.TestCase):
    def test_score_progress_uses_score_limit_as_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            (run / "posgen").mkdir(parents=True)
            (run / "score" / "shards").mkdir(parents=True)
            (run / "posgen" / "source.jsonl").write_text("x\n" * 10)
            (run / "score" / "shards" / "label.0.jsonl").write_text("a\n")
            (run / "score" / "shards" / "label.1.jsonl").write_text("b\n")
            config = {
                "create_args": {
                    "score_shards": 2,
                    "score_limit": 3,
                }
            }

            written, source_rows, completed = pipeline.score_progress(run, config)

            self.assertEqual(2, written)
            self.assertEqual(3, source_rows)
            self.assertEqual(0, completed)

    def test_score_progress_reads_completed_shard_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            (run / "posgen").mkdir(parents=True)
            (run / "score" / "shards").mkdir(parents=True)
            (run / "posgen" / "source.jsonl").write_text("x\n" * 10)
            stats = {"written": 7}
            (run / "score" / "shards" / "label.0.jsonl.stats.json").write_text(
                json.dumps(stats) + "\n"
            )
            config = {
                "create_args": {
                    "score_shards": 1,
                    "score_limit": 0,
                }
            }

            written, source_rows, completed = pipeline.score_progress(run, config)

            self.assertEqual(7, written)
            self.assertEqual(10, source_rows)
            self.assertEqual(1, completed)


if __name__ == "__main__":
    unittest.main()
