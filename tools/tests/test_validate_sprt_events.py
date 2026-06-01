#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
VALIDATE_PATH = REPO / "tools" / "validate" / "validate.py"
SPEC = importlib.util.spec_from_file_location("validate", VALIDATE_PATH)
assert SPEC is not None
validate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(validate)


class ValidateSprtEventTests(unittest.TestCase):
    def make_args(self, root: Path, *, log_dir: Path | None = None) -> Namespace:
        net = root / "model.nn"
        net.write_bytes(b"nn")
        return Namespace(
            net=str(net),
            games=16,
            concurrency=2,
            threads=1,
            hash=128,
            tc="2+0.02",
            elo0=0,
            elo1=8,
            restart="off",
            tag="unit-smoke",
            run=str(root / "run"),
            engine=None,
            reference_net=None,
            sprt=None,
            book=None,
            log_dir=str(log_dir) if log_dir else None,
            ntfy_url=None,
            event_command="",
        )

    def test_successful_sprt_emits_done_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events: list[dict] = []
            sprt_log_dir: Path | None = None
            old_run = validate.run
            old_emit = validate.emit_event

            def fake_run(command: list[str], *, env: dict[str, str] | None = None) -> int:
                nonlocal sprt_log_dir
                self.assertIsNotNone(env)
                assert env is not None
                sprt_log_dir = Path(env["LOG_DIR"])
                self.assertEqual(root.resolve() / "run" / "unit-smoke", sprt_log_dir.parent)
                self.assertTrue(sprt_log_dir.name.startswith("sprt_confirm_"))
                return 0

            def fake_emit(run_dir: str | Path, event: str, **kwargs: object) -> dict:
                payload = {"run": str(run_dir), "event": event, **kwargs}
                events.append(payload)
                return payload

            try:
                validate.run = fake_run
                validate.emit_event = fake_emit
                rc = validate.cmd_sprt(self.make_args(root))
            finally:
                validate.run = old_run
                validate.emit_event = old_emit

            self.assertEqual(0, rc)
            self.assertEqual("phase_start", events[0]["event"])
            self.assertEqual("done", events[1]["event"])
            self.assertEqual("validate_sprt", events[1]["stage"])
            self.assertEqual("ok", events[1]["status"])
            self.assertIsNotNone(sprt_log_dir)
            assert sprt_log_dir is not None
            self.assertEqual(sprt_log_dir / "run.log", events[1]["log"])

    def test_failed_sprt_emits_fail_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_dir = root / "custom-log"
            events: list[dict] = []
            old_run = validate.run
            old_emit = validate.emit_event

            def fake_run(command: list[str], *, env: dict[str, str] | None = None) -> int:
                self.assertIsNotNone(env)
                assert env is not None
                self.assertEqual(str(log_dir.resolve()), env["LOG_DIR"])
                return 1

            def fake_emit(run_dir: str | Path, event: str, **kwargs: object) -> dict:
                payload = {"run": str(run_dir), "event": event, **kwargs}
                events.append(payload)
                return payload

            try:
                validate.run = fake_run
                validate.emit_event = fake_emit
                rc = validate.cmd_sprt(self.make_args(root, log_dir=log_dir))
            finally:
                validate.run = old_run
                validate.emit_event = old_emit

            self.assertEqual(1, rc)
            self.assertEqual("fail", events[1]["event"])
            self.assertEqual("failed", events[1]["status"])
            self.assertEqual(log_dir.resolve() / "run.log", events[1]["log"])


if __name__ == "__main__":
    unittest.main()
