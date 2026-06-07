#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
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

    def test_score_progress_reads_bullet_shard_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source.jsonl"
            source.write_text("x\n" * 20)
            run = tmp_path / "run"
            (run / "score" / "shards").mkdir(parents=True)
            (run / "score" / "shards" / "label.0.bullet.stats.json").write_text(
                json.dumps({"written": 11}) + "\n",
                encoding="utf-8",
            )
            config = {
                "create_args": {
                    "score_shards": 1,
                    "score_limit": 0,
                    "score_source_jsonl": str(source),
                }
            }

            written, source_rows, completed = pipeline.score_progress(run, config)

            self.assertEqual(11, written)
            self.assertEqual(20, source_rows)
            self.assertEqual(1, completed)

    def test_launch_streams_output_to_stdout_and_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run = tmp_path / "run"
            config_path = tmp_path / "config.json"
            config = {
                "run": str(run),
                "repo": str(REPO),
                "steps": [
                    {
                        "name": "visible_test",
                        "why": "Prove command output is visible.",
                        "command": [
                            sys.executable,
                            "-c",
                            "print('visible-output')",
                        ],
                    }
                ],
            }
            config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")

            result = subprocess.run(
                [sys.executable, str(PIPELINE_PATH), "launch", str(config_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn("phase: visible_test", result.stdout)
            self.assertIn("why: Prove command output is visible.", result.stdout)
            self.assertIn("command:", result.stdout)
            self.assertIn("visible-output", result.stdout)

            log = (run / "runs.log").read_text(encoding="utf-8")
            self.assertIn("why visible_test: Prove command output is visible.", log)
            self.assertIn("visible-output", log)

            events = pipeline.read_events(run)
            phase_done = [
                event for event in events if event.get("event") == "phase_done"
            ][0]
            self.assertEqual(
                "Prove command output is visible.",
                phase_done.get("why"),
            )

    def test_launch_honors_existing_stop_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run = tmp_path / "run"
            run.mkdir()
            (run / "stop.json").write_text("{}\n", encoding="utf-8")
            config_path = tmp_path / "config.json"
            output = tmp_path / "should-not-exist"
            config = {
                "run": str(run),
                "repo": str(REPO),
                "steps": [
                    {
                        "name": "blocked",
                        "command": [
                            sys.executable,
                            "-c",
                            f"from pathlib import Path; Path({str(output)!r}).write_text('bad')",
                        ],
                    }
                ],
            }
            config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")

            result = subprocess.run(
                [sys.executable, str(PIPELINE_PATH), "launch", str(config_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, result.stderr)
            self.assertFalse(output.exists())
            events = pipeline.read_events(run)
            self.assertIn("stop", [event.get("event") for event in events])

    def test_step_reason_has_known_defaults(self) -> None:
        self.assertEqual(
            "Train and export the candidate network with Bullet.",
            pipeline.step_reason({"name": "bullet_train"}, "bullet_train"),
        )


if __name__ == "__main__":
    unittest.main()
