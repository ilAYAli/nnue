#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from contextlib import redirect_stdout
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
loader = SourceFileLoader("nnue_run", str(REPO / "nnue-run"))
spec = spec_from_loader("nnue_run", loader)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load ./nnue-run")
nnue_run = module_from_spec(spec)
spec.loader.exec_module(nnue_run)


class NnueRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_active_path = os.environ.get("NNUE_RUN_ACTIVE_PATH")
        self._old_require_event = os.environ.get("NNUE_REQUIRE_EVENT_COMMAND")

    def tearDown(self) -> None:
        if self._old_active_path is None:
            os.environ.pop("NNUE_RUN_ACTIVE_PATH", None)
        else:
            os.environ["NNUE_RUN_ACTIVE_PATH"] = self._old_active_path
        if self._old_require_event is None:
            os.environ.pop("NNUE_REQUIRE_EVENT_COMMAND", None)
        else:
            os.environ["NNUE_REQUIRE_EVENT_COMMAND"] = self._old_require_event

    def write_config(self, root: Path, *, run: str = "unit-simple") -> Path:
        out = root / "out.txt"
        config = {
            "run": run,
            "hypothesis": "unit runner proves stage visibility",
            "changed_variables": {
                "lr": "1e-10",
                "objective": "output-only",
            },
            "hooks": {
                "event_command": "true",
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
            path, config = nnue_run.load_config(config_path)
            self.assertEqual(path, config_path.resolve())
            buf = io.StringIO()
            with redirect_stdout(buf):
                nnue_run.print_plan(config)
            text = buf.getvalue()
            self.assertIn("Run: unit-simple", text)
            self.assertIn("Changed variables:", text)
            self.assertIn("lr: 1e-10", text)
            self.assertIn("1. train", text)
            self.assertIn("2. static_gate", text)

    def test_changed_variable_templates_are_rendered_in_stage_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = self.write_config(root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["changed_variables"]["candidate_nnue"] = "{run_dir}/train/model.nn"
            config["stages"][0]["command"] = ["echo", "{var_candidate_nnue}"]
            config_path.write_text(json.dumps(config), encoding="utf-8")
            _, loaded = nnue_run.load_config(config_path)
            text = nnue_run.render_plan(loaded)
            self.assertIn(str(root / "runs" / "unit-simple" / "train" / "model.nn"), text)

    def write_architecture(self, root: Path) -> Path:
        path = root / "architecture.json"
        path.write_text(json.dumps({
            "name": "enyo-native-1bucket-v1",
            "mode": "enyo",
            "hidden": 1024,
            "l2_size": 16,
            "input_buckets": 1,
            "runtime_input_buckets": 1,
            "feature_channels": 12,
            "input_factoriser": False,
            "output_buckets": 1,
            "eval_scale": 400.0,
            "l1_export_scale": 1.0,
            "export_format": "enyo-native-v1",
        }), encoding="utf-8")
        return path

    def test_compact_config_generates_native_bullet_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_architecture(root)
            path = root / "build.json"
            path.write_text(json.dumps({
                "run": "native-unit",
                "architecture": "architecture.json",
                "hypothesis": "unit compact config",
                "changed_variables": {
                    "init_scale": "l0_std is the only changed training knob",
                },
                "training": {
                    "l0_std": 256.0,
                },
            }), encoding="utf-8")

            _, loaded = nnue_run.load_config(path)
            self.assertEqual(
                [stage["name"] for stage in loaded["stages"]],
                [
                    "train",
                    "export_and_copy",
                    "smoke_test",
                    "sprt_100_game_smoke",
                    "sprt_1k_main",
                ],
            )
            train_env = loaded["stages"][0]["env"]
            self.assertEqual(train_env["ENYO_BULLET_MODE"], "enyo")
            self.assertEqual(train_env["ENYO_BULLET_HIDDEN"], 1024)
            self.assertEqual(train_env["ENYO_BULLET_ENYO_INPUT_BUCKETS"], 1)
            self.assertEqual(train_env["ENYO_BULLET_ENYO_FEATURE_CHANNELS"], 12)
            self.assertEqual(train_env["ENYO_BULLET_ENYO_L0_STD"], 256.0)
            self.assertIn("expected_size=1610052", loaded["stages"][1]["command"])
            self.assertIn("~/assets/nets/{run}.nn", loaded["validation"]["candidate_net"])

    def test_compact_config_rejects_unknown_training_knobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_architecture(root)
            path = root / "build.json"
            path.write_text(json.dumps({
                "run": "native-unit",
                "architecture": "architecture.json",
                "hypothesis": "unit compact config",
                "changed_variables": {"x": "y"},
                "training": {
                    "l0_std": 256.0,
                    "mystery": True,
                },
            }), encoding="utf-8")
            with self.assertRaises(SystemExit) as ctx:
                nnue_run.load_config(path)
            self.assertIn("training unknown field", str(ctx.exception))

    def test_compact_config_applies_architecture_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_architecture(root)
            path = root / "build.json"
            path.write_text(json.dumps({
                "run": "native-unit",
                "architecture": "architecture.json",
                "architecture_overrides": {
                    "output_buckets": 8,
                },
                "hypothesis": "unit compact config",
                "changed_variables": {
                    "architecture": "output buckets only",
                },
            }), encoding="utf-8")

            _, loaded = nnue_run.load_config(path)
            train_env = loaded["stages"][0]["env"]
            self.assertEqual(loaded["architecture"]["output_buckets"], 8)
            self.assertEqual(train_env["ENYO_BULLET_ENYO_OUTPUT_BUCKETS"], 8)
            self.assertIn("expected_size=1610976", loaded["stages"][1]["command"])

    def test_enyo_network_size_matches_known_layouts(self) -> None:
        architecture = {
            "input_buckets": 1,
            "feature_channels": 12,
            "hidden": 1024,
            "l2_size": 16,
            "output_buckets": 1,
        }
        self.assertEqual(nnue_run.enyo_network_size(architecture), 1610052)
        architecture["input_buckets"] = 16
        self.assertEqual(nnue_run.enyo_network_size(architecture), 25203012)

    def test_plan_publish_uses_generic_event_hook(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "hook-events.jsonl"
            hook = root / "hook.sh"
            hook.write_text(
                "#!/bin/sh\n"
                f"printf '%s\\n' \"$NNUE_RUN_EVENT_JSON\" >> {shlex.quote(str(events))}\n",
                encoding="utf-8",
            )
            hook.chmod(0o755)
            config_path = self.write_config(root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["hooks"]["event_command"] = str(hook)
            config_path.write_text(json.dumps(config), encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                rc = nnue_run.main(["plan", "-c", str(config_path), "--publish"])
            self.assertEqual(rc, 0)
            lines = [
                json.loads(line)
                for line in events.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(lines[-1]["event"], "plan")
            self.assertIn("Run: unit-simple", lines[-1]["message"])

    def test_start_runs_stages_and_status_is_readable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["NNUE_RUN_ACTIVE_PATH"] = str(root / "active.json")
            config_path = self.write_config(root)
            rc = nnue_run.main([
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
                nnue_run.print_status(run_dir)
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
                nnue_run.load_config(path)
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
            _, loaded = nnue_run.load_config(path)
            failures = nnue_run.run_doctor_checks(
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
            _, loaded = nnue_run.load_config(path)
            failures = nnue_run.run_doctor_checks(
                path,
                loaded,
                skip_git=True,
                require_tools=False,
            )
            self.assertTrue(any("./nnue-train pairwise" in failure for failure in failures))

    def test_doctor_does_not_require_unused_compiled_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_config(Path(tmp))
            _, config = nnue_run.load_config(path)
            old_func = nnue_run.compiled_tool_path
            nnue_run.compiled_tool_path = lambda _name: None
            try:
                failures = nnue_run.run_doctor_checks(
                    path,
                    config,
                    skip_git=True,
                    require_tools=True,
                )
            finally:
                nnue_run.compiled_tool_path = old_func
            self.assertFalse(any("compiled hot-path tool" in failure for failure in failures))

    def test_doctor_reports_missing_compiled_tools_when_config_uses_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_config(Path(tmp))
            config = json.loads(path.read_text(encoding="utf-8"))
            config["stages"][0]["command"] = ["nnue-mix-jsonl", "--output", "x"]
            path.write_text(json.dumps(config), encoding="utf-8")
            _, loaded = nnue_run.load_config(path)
            old_func = nnue_run.compiled_tool_path
            nnue_run.compiled_tool_path = lambda _name: None
            try:
                failures = nnue_run.run_doctor_checks(
                    path,
                    loaded,
                    skip_git=True,
                    require_tools=True,
                )
            finally:
                nnue_run.compiled_tool_path = old_func
            self.assertTrue(any("compiled hot-path tool" in failure for failure in failures))

    def test_doctor_reports_missing_forge_when_config_uses_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.write_config(root)
            config = json.loads(path.read_text(encoding="utf-8"))
            config["changed_variables"]["forge"] = str(root / "missing" / "forge")
            config["stages"][0]["command"] = [
                "{var_forge}",
                "nnue",
                "abc",
                "--candidate-nnue",
                "x",
            ]
            path.write_text(json.dumps(config), encoding="utf-8")
            _, loaded = nnue_run.load_config(path)
            failures = nnue_run.run_doctor_checks(
                path,
                loaded,
                skip_git=True,
                require_tools=False,
            )
            self.assertTrue(any("missing forge executable" in failure for failure in failures))

    def test_forge_path_resolves_from_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_repo = root / "nnue" / ".worktrees" / "active"
            fake_repo.mkdir(parents=True)
            forge = root / "crucible" / "scripts" / "forge"
            forge.parent.mkdir(parents=True)
            forge.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            forge.chmod(0o644)

            old_repo_root = nnue_run.repo_root
            nnue_run.repo_root = lambda: fake_repo
            try:
                self.assertEqual(
                    nnue_run.resolve_forge_command("../crucible/scripts/forge"),
                    str(forge.resolve()),
                )
                self.assertEqual(
                    nnue_run.normalize_command(["../crucible/scripts/forge", "nnue", "abc"]),
                    ["bash", str(forge.resolve()), "nnue", "abc"],
                )
            finally:
                nnue_run.repo_root = old_repo_root

    def test_doctor_allows_missing_event_hook_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_config(Path(tmp))
            config = json.loads(path.read_text(encoding="utf-8"))
            config.pop("hooks")
            path.write_text(json.dumps(config), encoding="utf-8")
            _, loaded = nnue_run.load_config(path)
            failures = nnue_run.run_doctor_checks(
                path,
                loaded,
                skip_git=True,
                require_tools=False,
            )
            self.assertFalse(any("event hook is required" in failure for failure in failures))

    def test_doctor_can_require_event_hook(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_config(Path(tmp))
            config = json.loads(path.read_text(encoding="utf-8"))
            config["hooks"] = {"require_event_command": True}
            path.write_text(json.dumps(config), encoding="utf-8")
            _, loaded = nnue_run.load_config(path)
            failures = nnue_run.run_doctor_checks(
                path,
                loaded,
                skip_git=True,
                require_tools=False,
            )
            self.assertTrue(any("event hook is required" in failure for failure in failures))

    def test_sigterm_emits_fail_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "hook-events.jsonl"
            hook = root / "hook.sh"
            hook.write_text(
                "#!/bin/sh\n"
                f"printf '%s\\n' \"$NNUE_RUN_EVENT_JSON\" >> {shlex.quote(str(events))}\n",
                encoding="utf-8",
            )
            hook.chmod(0o755)
            config = {
                "run": "unit-signal",
                "hypothesis": "signal notification test",
                "changed_variables": {"phase": "sleep"},
                "hooks": {"event_command": str(hook)},
                "run_dir": str(root / "runs" / "{run}"),
                "stages": [
                    {
                        "name": "sleep",
                        "command": [
                            sys.executable,
                            "-c",
                            "import time; time.sleep(30)",
                        ],
                    }
                ],
            }
            config_path = root / "build.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            env = os.environ.copy()
            env["NNUE_RUN_ACTIVE_PATH"] = str(root / "active.json")
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(REPO / "nnue-run"),
                    "start",
                    "-c",
                    str(config_path),
                    "--unsafe-skip-doctor",
                ],
                cwd=REPO,
                env=env,
            )
            try:
                deadline = time.time() + 10.0
                while time.time() < deadline:
                    if events.exists() and "phase_start" in events.read_text(encoding="utf-8"):
                        break
                    time.sleep(0.05)
                else:
                    self.fail("runner did not enter phase before timeout")

                proc.terminate()
                rc = proc.wait(timeout=10)
                self.assertEqual(rc, 143)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=10)

            lines = [
                json.loads(line)
                for line in events.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            fails = [line for line in lines if line.get("event") == "fail"]
            self.assertTrue(fails)
            self.assertIn("signal=15", str(fails[-1].get("status")))


if __name__ == "__main__":
    unittest.main()
