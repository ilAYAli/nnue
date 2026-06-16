#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import simple_runner  # noqa: E402


class SimpleRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_active_path = os.environ.get("NNUE_RUN_ACTIVE_PATH")

    def tearDown(self) -> None:
        if self._old_active_path is None:
            os.environ.pop("NNUE_RUN_ACTIVE_PATH", None)
        else:
            os.environ["NNUE_RUN_ACTIVE_PATH"] = self._old_active_path

    def write_config(self, root: Path, *, run: str = "unit-simple") -> Path:
        out = root / "out.txt"
        config = {
            "run": run,
            "hypothesis": "unit runner proves stage visibility",
            "changed_variables": {
                "lr": "1e-10",
                "objective": "output-only",
            },
            "run_dir": str(root / "runs" / "{run}"),
            "stages": [
                {
                    "name": "train",
                    "why": "write marker",
                    "command": [
                        sys.executable,
                        "-c",
                        (
                            "from pathlib import Path; "
                            f"Path({str(out)!r}).write_text('ok', encoding='utf-8')"
                        ),
                    ],
                },
                {
                    "name": "static_gate",
                    "command": [sys.executable, "-c", "print('gate ok')"],
                },
            ],
        }
        path = root / "build.json"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        return path

    def test_plan_prints_changed_variables_and_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = self.write_config(Path(tmp))
            path, config = simple_runner.load_config(config_path)
            self.assertEqual(path, config_path.resolve())
            buf = io.StringIO()
            with redirect_stdout(buf):
                simple_runner.print_plan(config)
            text = buf.getvalue()
            self.assertIn("Run: unit-simple", text)
            self.assertIn("Changed variables:", text)
            self.assertIn("lr: 1e-10", text)
            self.assertIn("1. train", text)
            self.assertIn("2. static_gate", text)

    def test_start_runs_stages_and_status_is_readable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["NNUE_RUN_ACTIVE_PATH"] = str(root / "active.json")
            config_path = self.write_config(root)
            rc = simple_runner.main([
                "start",
                "-c",
                str(config_path),
                "--unsafe-skip-doctor",
            ])
            self.assertEqual(rc, 0)
            self.assertEqual((root / "out.txt").read_text(encoding="utf-8"), "ok")

            run_dir = root / "runs" / "unit-simple"
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "done")
            self.assertEqual(status["current_stage"], "done")
            self.assertTrue((run_dir / "stages" / "train.done").exists())
            self.assertTrue((run_dir / "stages" / "static_gate.done").exists())

            buf = io.StringIO()
            with redirect_stdout(buf):
                simple_runner.print_status(run_dir)
            text = buf.getvalue()
            self.assertIn("Run: unit-simple", text)
            self.assertIn("State: done", text)
            self.assertIn("objective: output-only", text)

    def test_rejects_unknown_top_level_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "build.json"
            path.write_text(json.dumps({
                "run": "bad",
                "hypothesis": "x",
                "changed_variables": {"lr": 1},
                "stages": [{"name": "x", "command": "true"}],
                "surprise": True,
            }), encoding="utf-8")
            with self.assertRaises(SystemExit) as ctx:
                simple_runner.load_config(path)
            self.assertIn("unknown top-level", str(ctx.exception))

    def test_doctor_rejects_replaced_python_hot_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.write_config(root)
            config = json.loads(path.read_text(encoding="utf-8"))
            config["stages"][0]["command"] = [
                "python3",
                "tools/posgen/mix_jsonl.py",
                "--output",
                "x",
            ]
            path.write_text(json.dumps(config), encoding="utf-8")
            _, loaded = simple_runner.load_config(path)
            failures = simple_runner.run_doctor_checks(
                path,
                loaded,
                skip_git=True,
                require_tools=False,
            )
            self.assertTrue(any("nnue-mix-jsonl" in failure for failure in failures))

    def test_doctor_rejects_direct_training_entrypoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.write_config(root)
            config = json.loads(path.read_text(encoding="utf-8"))
            config["stages"][0]["command"] = [
                "python3",
                "tools/train/train_pairwise.py",
                "--data",
                "x",
            ]
            path.write_text(json.dumps(config), encoding="utf-8")
            _, loaded = simple_runner.load_config(path)
            failures = simple_runner.run_doctor_checks(
                path,
                loaded,
                skip_git=True,
                require_tools=False,
            )
            self.assertTrue(any("./nnue-train pairwise" in failure for failure in failures))

    def test_doctor_reports_missing_compiled_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_config(Path(tmp))
            _, config = simple_runner.load_config(path)
            old_func = simple_runner.compiled_tool_path
            simple_runner.compiled_tool_path = lambda _name: None
            try:
                failures = simple_runner.run_doctor_checks(
                    path,
                    config,
                    skip_git=True,
                    require_tools=True,
                )
            finally:
                simple_runner.compiled_tool_path = old_func
            self.assertTrue(any("compiled hot-path tool" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
