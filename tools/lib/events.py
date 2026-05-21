#!/usr/bin/env python3
"""Generic NNUE run event emission and hook execution."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def expand_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def event_path(run_dir: Path) -> Path:
    return run_dir / "events.jsonl"


def hook_from_config(config: dict[str, Any] | None) -> str:
    if not config:
        return ""
    hooks = config.get("hooks", {})
    if not isinstance(hooks, dict):
        return ""
    command = hooks.get("event_command", "")
    return str(command) if command else ""


def emit_event(
    run_dir: str | Path,
    event: str,
    *,
    stage: str = "",
    status: str = "",
    rc: int | None = None,
    message: str = "",
    log: str | Path | None = None,
    command: list[str] | None = None,
    hook_command: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_path = expand_path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "time": timestamp(),
        "event": event,
        "run": str(run_path),
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }
    if stage:
        payload["stage"] = stage
    if status:
        payload["status"] = status
    if rc is not None:
        payload["rc"] = rc
    if message:
        payload["message"] = message
    if log is not None:
        payload["log"] = str(log)
    if command is not None:
        payload["command"] = command
    if extra:
        payload.update(extra)

    line = json.dumps(payload, sort_keys=True)
    with event_path(run_path).open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")

    command_text = hook_command or os.environ.get("NNUE_EVENT_COMMAND", "")
    if command_text:
        run_hook(command_text, payload)

    return payload


def run_hook(command: str, payload: dict[str, Any]) -> None:
    line = json.dumps(payload, sort_keys=True)
    env = os.environ.copy()
    env.update({
        "NNUE_RUN_EVENT_JSON": line,
        "NNUE_RUN_EVENT": str(payload.get("event", "")),
        "NNUE_RUN_STAGE": str(payload.get("stage", "")),
        "NNUE_RUN_DIR": str(payload.get("run", "")),
    })
    try:
        subprocess.run(
            command,
            input=line + "\n",
            text=True,
            shell=True,
            env=env,
            timeout=int(os.environ.get("NNUE_EVENT_HOOK_TIMEOUT", "30")),
            check=False,
        )
    except Exception as exc:  # Hook failures must not kill training.
        print(f"event hook failed: {exc}", file=sys.stderr, flush=True)


def cmd_send(args: argparse.Namespace) -> int:
    extra: dict[str, Any] = {}
    for item in args.extra:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--extra must be key=value, got: {item}")
        extra[key] = value
    emit_event(
        args.run,
        args.event,
        stage=args.stage or "",
        status=args.status or "",
        rc=args.rc,
        message=args.message or "",
        log=args.log,
        hook_command=args.event_command or "",
        extra=extra,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emit a generic NNUE run event.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    send = subparsers.add_parser("send", help="Append an event and run the hook.")
    send.add_argument("--run", required=True)
    send.add_argument("--event", required=True)
    send.add_argument("--stage")
    send.add_argument("--status")
    send.add_argument("--rc", type=int)
    send.add_argument("--message")
    send.add_argument("--log")
    send.add_argument("--event-command")
    send.add_argument("--extra", action="append", default=[])
    send.set_defaults(func=cmd_send)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
