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
                or "Crucible run-sprt tasks=8/8 games=400/400 elo=+5.2 "
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
                "message": "accepted run: Crucible run-sprt tasks=8/8 games=400/400 elo=+5.2 ci=27.7 llr=0.04/2.20 (2%) los=52.3% draw=36.5%",
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
            return proc, hook_log.read_text(encoding="utf-8")

    def test_done_event_renders_status_to_ai_stdout(self) -> None:
        proc, log = self.run_hook({
            "NNUE_NTFY_EVENTS": "",
            "NNUE_AI_STDOUT_EVENTS": "done",
            "NNUE_AI_STDIN_EVENTS": "",
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
        self.assertNotIn("nnue_sent", log)
        self.assertIn("event=done → AI_stdout", log)

    def test_sprt_phase_done_reports_metrics(self) -> None:
        line = (
            "Crucible uho-native-1.1.11-smoke-400-20260624-225641 "
            "games=400/400 elo=-14.8 ci=28.8 llr=-0.21/2.20 (-10%) "
            "los=42.8% draw=34.3%"
        )
        proc, log = self.run_hook(
            {},
            payload_extra={
                "event": "phase_done",
                "stage": "sprt_smoke",
                "status": "phase_done",
                "message": f"SPRT smoke inconclusive for uho-native-1.1.11: {line}",
                "sprt": {
                    "line": line,
                    "elo": "-14.8",
                    "llr": "-0.21/2.20 (-10%)",
                },
            },
        )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("NNUE SPRT", proc.stdout)
        self.assertIn("Stage: sprt_smoke", proc.stdout)
        self.assertIn("Games: 400/400", proc.stdout)
        self.assertIn("Elo: -14.8 +/- 28.8", proc.stdout)
        self.assertIn("LLR: -0.21/2.20 (-10%)", proc.stdout)
        self.assertIn("LOS: 42.8%", proc.stdout)
        self.assertIn("Draw: 34.3%", proc.stdout)
        self.assertIn("Status: inconclusive", proc.stdout)
        self.assertIn("event=phase_done → AI_stdout", log)

    def test_iteration_done_can_wake_ai_stdin(self) -> None:
        proc, log = self.run_hook(
            {"NNUE_AI_STDIN_EVENTS": "done,fail"},
            payload_extra={
                "event": "iteration_done",
                "stage": "iterate",
                "status": "iteration_done",
                "message": "accepted run",
            },
        )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("https://ntfy.wahlman.no/nnue", proc.stdout)
        self.assertNotIn("https://ntfy.wahlman.no/AI_stdin", proc.stdout)
        self.assertIn("notifai:AI_stdin:1.1", proc.stdout)
        self.assertIn("event=iteration_done → nnue", log)
        self.assertNotIn("event=iteration_done → AI_stdin", log)

    def test_fail_event_reports_single_error_line(self) -> None:
        proc, log = self.run_hook(
            {
                "NNUE_NTFY_EVENTS": "fail",
                "NNUE_AI_STDOUT_EVENTS": "",
                "NNUE_AI_STDIN_EVENTS": "",
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
            {"NNUE_AI_STDIN_EVENTS": "fail"},
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
        self.assertIn("notifai:AI_stdin:1.1", proc.stdout)
        self.assertNotIn("https://ntfy.wahlman.no/AI_stdin", proc.stdout)
        self.assertIn("event=fail → ping", log)
        self.assertNotIn("event=fail → AI_stdin", log)


if __name__ == "__main__":
    unittest.main()
