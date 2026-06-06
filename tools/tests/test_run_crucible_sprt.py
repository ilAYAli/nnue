#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "validate" / "run_crucible_sprt.py"
SPEC = importlib.util.spec_from_file_location("run_crucible_sprt", SCRIPT)
assert SPEC is not None
run_crucible_sprt = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["run_crucible_sprt"] = run_crucible_sprt
SPEC.loader.exec_module(run_crucible_sprt)


def write_file(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class RunCrucibleSprtTests(unittest.TestCase):
    def args(self, root: Path) -> Namespace:
        return Namespace(
            net=str(write_file(root / "candidate.nn")),
            reference_net=str(write_file(root / "reference.nn")),
            tag="unit-sprt",
            description="",
            workers=str(write_file(root / "workers.json", '{"workers": []}\n')),
            crucible="crucible",
            crucible_runs_dir=str(root / "crucible-runs"),
            crucible_notify_command=str(root / "notify.sh"),
            work_dir=str(root),
            output_dir=str(root / "out"),
            nnue_run="",
            event_command=None,
            cache_dir=str(root / "cache"),
            book=str(write_file(root / "book.epd")),
            games=200,
            chunk_games=100,
            chunk_concurrency=4,
            threads=1,
            hash=512,
            tc="2+0.02",
            elo0=0.0,
            elo1=8.0,
            seed=1234,
            jobs=4,
            task_timeout_seconds=1800,
            remote_timeout_seconds=1800,
            timeout_command="timeout",
            path_map=[],
            coordinator_host=None,
            resume=False,
            replace=False,
            verbose=False,
            crucible_notify=False,
            retry_startup_failures=True,
            notify=False,
            sprt_ntfy_url="",
            task_count=2,
        )

    def test_manifest_rejects_odd_game_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            args = self.args(root)
            args.chunk_games = 25
            with self.assertRaises(SystemExit) as caught:
                run_crucible_sprt.build_manifest(args, root / "manifest-out")
            self.assertIn("--chunk-games must be even", str(caught.exception))

            args = self.args(root)
            args.games = 201
            with self.assertRaises(SystemExit) as caught:
                run_crucible_sprt.build_manifest(args, root / "manifest-out")
            self.assertIn("--games must be even", str(caught.exception))

    def test_manifest_uses_restart_off_and_progress_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            args = self.args(root)
            manifest = run_crucible_sprt.build_manifest(args, root / "manifest-out")

            self.assertEqual("sprt", manifest["kind"])
            self.assertEqual(2, len(manifest["tasks"]))
            command = manifest["tasks"][0]["command"]
            self.assertIn("--restart off", command)
            self.assertIn("--no-ntfy", command)
            self.assertIn("--no-eta", command)
            self.assertIn("--srand 1234", command)
            self.assertIn("$HOME/.local/bin/sprt", command)
            self.assertEqual("sprt:100", manifest["tasks"][0]["label"])
            self.assertEqual("games", manifest["tasks"][0]["progress"]["unit"])

    def test_manifest_maps_worker_run_and_cache_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            workers = {
                "workers": [
                    {
                        "host": "pwa-rce",
                        "profile": "pwa-rce",
                        "cache_dir": "/home/petter/.cache/crucible",
                    },
                    {
                        "host": "pwa-mbp",
                        "profile": "pwa-mbp",
                        "cache_dir": "/Users/pwahlman/.cache/crucible",
                    },
                    {
                        "host": "pwa-mbp0",
                        "profile": "pwa-mbp0",
                        "cache_dir": "/Users/petter/Library/Caches/crucible",
                    },
                ],
            }
            args = self.args(root)
            args.cache_dir = "~/.cache/crucible"
            write_file(Path(args.workers), json.dumps(workers))
            manifest = run_crucible_sprt.build_manifest(args, root / "manifest-out")

            runs = str((root / "crucible-runs").resolve())
            cache = str(Path("~/.cache/crucible").expanduser().resolve())
            self.assertEqual(str((root / "crucible-runs" / "unit-sprt").resolve()), manifest["work_dir"])
            self.assertEqual(
                "/home/petter/.cache/crucible/runs",
                manifest["path_maps"]["pwa-rce"][runs],
            )
            self.assertEqual(
                "/Users/pwahlman/.cache/crucible/runs",
                manifest["path_maps"]["pwa-mbp"][runs],
            )
            self.assertEqual(
                "/Users/petter/Library/Caches/crucible",
                manifest["path_maps"]["pwa-mbp0"][cache],
            )
            self.assertEqual(
                "/Users/petter/Library/Caches/crucible",
                manifest["path_maps"]["pwa-mbp0"]["$HOME/.cache/crucible"],
            )

    def test_home_paths_expand_in_worker_commands(self) -> None:
        self.assertEqual(
            '"$HOME/.cache/crucible/book.epd"',
            run_crucible_sprt.shell_quote_expand_home("$HOME/.cache/crucible/book.epd"),
        )
        self.assertEqual(
            '"nnue_file=$HOME/.cache/crucible/model.nn"',
            run_crucible_sprt.shell_quote_expand_home("nnue_file=$HOME/.cache/crucible/model.nn"),
        )

    def test_aggregate_sprt_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            write_file(
                root / "outputs" / "sprt_0000" / "sprt.log",
                "Score of candidate vs reference: 10 - 5 - 85  [0.525] 100\n",
            )
            write_file(
                root / "outputs" / "sprt_0001" / "sprt.log",
                "Score of candidate vs reference: 7 - 9 - 84  [0.510] 100\n",
            )

            aggregate = run_crucible_sprt.aggregate_sprt_logs(root)

            self.assertEqual(17, aggregate.wins)
            self.assertEqual(14, aggregate.losses)
            self.assertEqual(169, aggregate.draws)
            self.assertEqual(200, aggregate.games)

    def test_transient_startup_classifier(self) -> None:
        self.assertTrue(
            run_crucible_sprt.is_transient_startup_failure(
                'Fatal; reference engine startup failure: "Engine didn\'t respond to uciok after startup"'
            )
        )
        self.assertFalse(run_crucible_sprt.is_transient_startup_failure("missing book"))

    def test_status_json_accepts_failed_run_json(self) -> None:
        old_run_capture = run_crucible_sprt.run_capture

        class Result:
            returncode = 1
            stdout = '{"counts": {"done": 0, "fail": 1}}\n'

        try:
            run_crucible_sprt.run_capture = lambda command: Result()
            status = run_crucible_sprt.status_json("crucible", "failed-run")
        finally:
            run_crucible_sprt.run_capture = old_run_capture

        self.assertEqual(1, status["counts"]["fail"])

    def test_completion_event_accepts_default_hook(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            args = self.args(root)
            args.notify = True
            args.event_command = None
            calls: list[dict] = []
            old_emit_event = run_crucible_sprt.emit_event

            def fake_emit_event(*call_args, **kwargs) -> None:
                calls.append({"args": call_args, "kwargs": kwargs})

            try:
                run_crucible_sprt.emit_event = fake_emit_event
                run_crucible_sprt.emit_completion_event(
                    args,
                    0,
                    root / "run",
                    "ok",
                    log_path=root / "out" / "deploy.log",
                )
            finally:
                run_crucible_sprt.emit_event = old_emit_event

            self.assertEqual(1, len(calls))
            hook = calls[0]["kwargs"]["hook_command"]
            self.assertIn("nnue_event_ntfy.sh", hook)
            self.assertIn("NNUE_NTFY_EVENTS=", hook)
            self.assertIn("NNUE_AI_STDIN_EVENTS=done,fail", hook)
            self.assertNotIn("NNUE_AI_STDOUT_ENABLE=0", hook)
            self.assertEqual(root / "out" / "deploy.log", calls[0]["kwargs"]["log"])

    def test_empty_sprt_ntfy_url_is_noop(self) -> None:
        old_urlopen = run_crucible_sprt.urllib.request.urlopen
        calls: list[object] = []

        def fake_urlopen(*args, **kwargs):
            calls.append(args)
            raise AssertionError("urlopen should not be called")

        try:
            run_crucible_sprt.urllib.request.urlopen = fake_urlopen
            ok = run_crucible_sprt.post_sprt_ntfy("", "message", title="SPRT finished")
        finally:
            run_crucible_sprt.urllib.request.urlopen = old_urlopen

        self.assertTrue(ok)
        self.assertEqual([], calls)

    def test_sprt_ntfy_posts_aggregate_message(self) -> None:
        old_urlopen = run_crucible_sprt.urllib.request.urlopen
        old_auth = run_crucible_sprt.load_ntfy_auth
        requests = []

        class Response:
            def close(self) -> None:
                pass

        def fake_urlopen(request, timeout):
            requests.append((request, timeout))
            return Response()

        try:
            run_crucible_sprt.load_ntfy_auth = lambda: "user:pass"
            run_crucible_sprt.urllib.request.urlopen = fake_urlopen
            ok = run_crucible_sprt.post_sprt_ntfy(
                "https://ntfy.example/sprt",
                "Distributed SPRT result",
                title="SPRT finished",
            )
        finally:
            run_crucible_sprt.urllib.request.urlopen = old_urlopen
            run_crucible_sprt.load_ntfy_auth = old_auth

        self.assertTrue(ok)
        self.assertEqual(1, len(requests))
        request, timeout = requests[0]
        self.assertEqual(10, timeout)
        self.assertEqual("https://ntfy.example/sprt", request.full_url)
        self.assertEqual("SPRT finished", request.headers["Title"])
        self.assertIn("Authorization", request.headers)

    def test_nonzero_deploy_is_ok_when_status_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            args = self.args(root)
            args.sprt_ntfy_url = "https://ntfy.example/sprt"
            calls: list[list[str]] = []
            messages: list[str] = []
            sprt_messages: list[tuple[str, str]] = []
            old_run_streamed = run_crucible_sprt.run_streamed
            old_status_json = run_crucible_sprt.status_json
            old_aggregate = run_crucible_sprt.aggregate_sprt_logs
            old_notify = run_crucible_sprt.notify_stdout
            old_sprt_notify = run_crucible_sprt.notify_sprt

            def fake_run_streamed(command: list[str], log_path: Path) -> int:
                calls.append(command)
                return 1

            def fake_status_json(crucible: str, run_name: str) -> dict:
                return {"counts": {"done": 2, "fail": 0}}

            def fake_aggregate(run_dir: Path):
                return run_crucible_sprt.SprtAggregate(wins=20, losses=10, draws=170, games=200)

            def fake_notify(message: str, *, enabled: bool) -> None:
                messages.append(message)

            def fake_sprt_notify(args: Namespace, message: str, *, title: str) -> None:
                sprt_messages.append((title, message))

            try:
                run_crucible_sprt.run_streamed = fake_run_streamed
                run_crucible_sprt.status_json = fake_status_json
                run_crucible_sprt.aggregate_sprt_logs = fake_aggregate
                run_crucible_sprt.notify_stdout = fake_notify
                run_crucible_sprt.notify_sprt = fake_sprt_notify
                rc = run_crucible_sprt.cmd_run(args)
            finally:
                run_crucible_sprt.run_streamed = old_run_streamed
                run_crucible_sprt.status_json = old_status_json
                run_crucible_sprt.aggregate_sprt_logs = old_aggregate
                run_crucible_sprt.notify_stdout = old_notify
                run_crucible_sprt.notify_sprt = old_sprt_notify

            self.assertEqual(0, rc)
            self.assertTrue(calls)
            self.assertNotIn("--notify-command", calls[0])
            self.assertIn("Distributed SPRT", messages[-1])
            self.assertEqual(1, len(sprt_messages))
            self.assertEqual("SPRT finished", sprt_messages[0][0])
            self.assertIn("Distributed SPRT", sprt_messages[0][1])

    def test_failed_deploy_posts_sprt_failure_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            args = self.args(root)
            args.sprt_ntfy_url = "https://ntfy.example/sprt"
            sprt_messages: list[tuple[str, str]] = []
            old_run_streamed = run_crucible_sprt.run_streamed
            old_status_json = run_crucible_sprt.status_json
            old_notify = run_crucible_sprt.notify_stdout
            old_sprt_notify = run_crucible_sprt.notify_sprt
            old_retry = run_crucible_sprt.retry_transient_failures

            def fake_run_streamed(command: list[str], log_path: Path) -> int:
                return 1

            def fake_status_json(crucible: str, run_name: str) -> dict:
                return {"counts": {"done": 0, "fail": 1}}

            def fake_sprt_notify(args: Namespace, message: str, *, title: str) -> None:
                sprt_messages.append((title, message))

            try:
                run_crucible_sprt.run_streamed = fake_run_streamed
                run_crucible_sprt.status_json = fake_status_json
                run_crucible_sprt.notify_stdout = lambda message, *, enabled: None
                run_crucible_sprt.notify_sprt = fake_sprt_notify
                run_crucible_sprt.retry_transient_failures = lambda args, run_name, run_dir: 0
                rc = run_crucible_sprt.cmd_run(args)
            finally:
                run_crucible_sprt.run_streamed = old_run_streamed
                run_crucible_sprt.status_json = old_status_json
                run_crucible_sprt.notify_stdout = old_notify
                run_crucible_sprt.notify_sprt = old_sprt_notify
                run_crucible_sprt.retry_transient_failures = old_retry

            self.assertEqual(1, rc)
            self.assertEqual(1, len(sprt_messages))
            self.assertEqual("SPRT failed", sprt_messages[0][0])
            self.assertIn("Distributed SPRT failed", sprt_messages[0][1])


if __name__ == "__main__":
    unittest.main()
