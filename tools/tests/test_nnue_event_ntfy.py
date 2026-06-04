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
    def run_hook(self, env_extra: dict[str, str]) -> tuple[subprocess.CompletedProcess[str], str]:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            run = tmp / "run"
            run.mkdir()
            log_path = run / "deploy.log"
            log_path.write_text(
                "2026-06-04 Enyo NNUE SPRT finished diagnostic rc=0 "
                "[64/64] Elo 1.0 +/- 10.0 | LLR 0.10/2.94 ( 3%) "
                "| LOS 55.0% | draw 25.0% | ETA 0s\n",
                encoding="utf-8",
            )
            hook_log = tmp / "hook.log"
            payload = {
                "event": "done",
                "run": str(run),
                "stage": "validate_crucible_sprt",
                "status": "ok",
                "rc": 0,
                "log": str(log_path),
                "host": "test-host",
            }
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

    def test_empty_event_lists_disable_all_routes(self) -> None:
        proc, log = self.run_hook({
            "NNUE_NTFY_EVENTS": "",
            "NNUE_AI_STDOUT_EVENTS": "",
            "NNUE_AI_STDIN_EVENTS": "",
        })

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertEqual("", proc.stdout)
        self.assertIn("event=done skipped", log)

    def test_empty_nnue_events_do_not_disable_ai_stdout(self) -> None:
        proc, log = self.run_hook({
            "NNUE_NTFY_EVENTS": "",
            "NNUE_AI_STDOUT_EVENTS": "done",
            "NNUE_AI_STDIN_EVENTS": "",
        })

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("NNUE status", proc.stdout)
        self.assertNotIn("nnue_sent", log)
        self.assertIn("ai_stdout_sent", log)


if __name__ == "__main__":
    unittest.main()
