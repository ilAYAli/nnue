#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
HOOK = REPO / "tools" / "events" / "nnue_event_ntfy.sh"


class NnueEventNtfyTests(unittest.TestCase):
    def run_hook(
        self,
        env_extra: dict[str, str],
        *,
        payload_extra: dict[str, object] | None = None,
        log_text: str | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], str]:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            run = tmp / "run"
            run.mkdir()
            (run / "build.resolved.json").write_text(
                json.dumps({
                    "run": "run",
                    "hypothesis": "Continue accepted uho-native-1.0.33 using the next UHO data window",
                }),
                encoding="utf-8",
            )
            validate = run / "validate"
            validate.mkdir()
            log_path = run / "deploy.log"
            log_path.write_text(
                log_text
                or "Forge run-sprt tasks=8/8 games=400/400 elo=+5.2 "
                   "ci=27.7 llr=0.04/2.20 (2%) los=52.3% draw=36.5%\n",
                encoding="utf-8",
            )
            (validate / "move_gate.summary.json").write_text(
                json.dumps({
                    "baseline_prefers_best": 1137,
                    "candidate_prefers_best": 1144,
                    "cases": [],
                    "delta_avg_margin": 2.078,
                    "delta_loss_weighted_margin": 1.515,
                    "fixed": 25,
                    "regressed": 18,
                    "by_source": {
                        "loss": {"cases": 2487},
                    },
                }),
                encoding="utf-8",
            )
            hook_log = tmp / "hook.log"
            payload = {
                "event": "done",
                "run": str(run),
                "stage": "iterate",
                "status": "ok",
                "rc": 0,
                "log": str(log_path),
                "host": "test-host",
                "message": "accepted run: Forge run-sprt tasks=8/8 games=400/400 elo=+5.2 ci=27.7 llr=0.04/2.20 (2%) los=52.3% draw=36.5%",
            }
            if payload_extra:
                payload.update(payload_extra)
            env = os.environ.copy()
            env.update({
                "HOME": tmp_name,
                "NNUE_NTFY_DRY_RUN": "1",
                "NNUE_NTFY_LOG": str(hook_log),
            })
            env.update(env_extra)
            proc = subprocess.run(
                [str(HOOK)],
                input=json.dumps(payload),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                check=False,
            )
            return proc, hook_log.read_text(encoding="utf-8") if hook_log.exists() else ""

    def test_done_event_renders_status_to_ai_stdout(self) -> None:
        proc, log = self.run_hook({
            "NNUE_HOOK_EVENTS": "done",
        })

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("NNUE done", proc.stdout)
        self.assertIn("Run: run", proc.stdout)
        self.assertIn(
            "Hypothesis: Continue accepted uho-native-1.0.33 using the next UHO data window",
            proc.stdout,
        )
        self.assertIn(
            "Status: tasks=8/8 games=400/400 elo=+5.2 ci=27.7 "
            "llr=0.04/2.20 (2%) los=52.3% draw=36.5%",
            proc.stdout,
        )
        self.assertNotIn("What ran", proc.stdout)
        self.assertNotIn("Next", proc.stdout)
        self.assertIn("event=done → llmsh ai-in", log)

    def test_iteration_done_can_wake_ai_stdin(self) -> None:
        proc, log = self.run_hook(
            {"NNUE_HOOK_EVENTS": "done,fail"},
            payload_extra={
                "event": "iteration_done",
                "stage": "iterate",
                "status": "iteration_done",
                "message": "accepted run",
            },
        )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("https://ntfy.wahlman.no/llmsh", proc.stdout)
        self.assertNotIn("https://ntfy.wahlman.no/nnue", proc.stdout)
        self.assertIn("event=iteration_done → llmsh ai-in", log)

    def test_training_done_can_wake_ai_stdin(self) -> None:
        proc, log = self.run_hook(
            {"NNUE_HOOK_EVENTS": "done,fail"},
            payload_extra={
                "event": "training_done",
                "stage": "train",
                "status": "training_done",
                "message": "training/export completed for run",
            },
        )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("https://ntfy.wahlman.no/llmsh", proc.stdout)
        self.assertIn("event=training_done → llmsh ai-in", log)

    def test_fail_event_reports_single_error_line(self) -> None:
        proc, log = self.run_hook(
            {
                "NNUE_HOOK_EVENTS": "",
            },
            payload_extra={
                "event": "fail",
                "stage": "train",
                "status": "fail",
                "critical_failure": True,
                "message": "train/export failed for run",
            },
            log_text="thread main panicked\nFAILED: train/export failed for run train\n",
        )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("NNUE fail", proc.stdout)
        self.assertIn("Run: run", proc.stdout)
        self.assertIn("Error: FAILED: train/export failed for run train", proc.stdout)
        self.assertNotIn("Result", proc.stdout)
        self.assertIn("event=fail → ping", log)

    def test_fail_can_wake_ai_stdin(self) -> None:
        proc, log = self.run_hook(
            {"NNUE_HOOK_EVENTS": "fail"},
            payload_extra={
                "event": "fail",
                "stage": "train",
                "status": "fail",
                "message": "train/export failed for run",
            },
            log_text="FAILED: train/export failed for run train\n",
        )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("https://ntfy.wahlman.no/ping", proc.stdout)
        self.assertIn("https://ntfy.wahlman.no/llmsh", proc.stdout)
        self.assertIn("https://ntfy.wahlman.no/ping", proc.stdout)
        self.assertIn("event=fail → ping", log)
        self.assertIn("event=fail → llmsh ai-in", log)


if __name__ == "__main__":
    unittest.main()
