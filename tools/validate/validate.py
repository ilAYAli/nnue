#!/usr/bin/env python3
"""Validate Enyo NNUE candidates."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.events import emit_event
from lib.defaults import DEFAULTS


def expand_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def expand_user(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser()


def tools_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run(command: list[str], *, env: dict[str, str] | None = None) -> int:
    print(" ".join(command), flush=True)
    proc = subprocess.Popen(command, env=env)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        raise


def event_run_dir(*paths: str | Path | None) -> Path:
    for path in paths:
        if path:
            return expand_path(path)
    return tools_root().parent / "runs" / "validation"


def cmd_static(args: argparse.Namespace) -> int:
    script = tools_root() / "validate" / "eval_dataset.py"
    run_dir = event_run_dir(args.run, expand_path(args.data).parents[1])
    command = [
        str(expand_user(args.python)),
        str(script),
        "--net", str(expand_path(args.net)),
        "--data", str(expand_path(args.data)),
        "--rows", str(args.rows),
        "--skip", str(args.skip),
        "--batch-size", str(args.batch_size),
        "--device", args.device,
        "--target-clamp", str(args.target_clamp),
    ]
    if args.buckets:
        command.append("--buckets")
    if args.sources:
        command.append("--sources")
    emit_event(
        run_dir, "phase_start", stage="validate_static", status="running",
        command=command, hook_command=args.event_command or "",
    )
    rc = run(command)
    emit_event(
        run_dir, "phase_done" if rc == 0 else "fail",
        stage="validate_static", status="ok" if rc == 0 else "failed",
        rc=rc, command=command, hook_command=args.event_command or "",
    )
    return rc


def cmd_failure_suite(args: argparse.Namespace) -> int:
    script = tools_root() / "validate" / "replay_failure_suite.py"
    run_dir = event_run_dir(args.run, args.output_dir)
    command = [
        sys.executable,
        str(script),
        "--candidate", str(expand_path(args.candidate)),
        "--reference", str(expand_path(args.reference)),
        "--oracle", args.oracle,
        "--replay", args.replay,
        "--threads", str(args.threads),
        "--jobs", str(args.jobs),
        "--fixed-nodes", str(args.fixed_nodes),
        "--max-replay-nodes", str(args.max_replay_nodes),
        "--oracle-nodes", str(args.oracle_nodes),
        "--count", str(args.count),
        "--output-dir", str(expand_path(args.output_dir)),
    ]
    if args.stderr:
        command += ["--stderr", str(expand_path(args.stderr))]
    command.extend(str(expand_path(log)) for log in args.logs)
    emit_event(
        run_dir, "phase_start", stage="validate_failure_suite", status="running",
        command=command, hook_command=args.event_command or "",
    )
    rc = run(command)
    emit_event(
        run_dir, "phase_done" if rc == 0 else "fail",
        stage="validate_failure_suite", status="ok" if rc == 0 else "failed",
        rc=rc, command=command, hook_command=args.event_command or "",
    )
    return rc


def cmd_net_diff(args: argparse.Namespace) -> int:
    script = tools_root() / "validate" / "net_diff.py"
    run_dir = event_run_dir(args.run, expand_path(args.candidate).parent)
    command = [
        sys.executable,
        str(script),
        "--candidate", str(expand_path(args.candidate)),
        "--reference", str(expand_path(args.reference)),
    ]
    if args.fail_if_identical:
        command.append("--fail-if-identical")
    emit_event(
        run_dir, "phase_start", stage="validate_net_diff", status="running",
        command=command, hook_command=args.event_command or "",
    )
    rc = run(command)
    emit_event(
        run_dir, "phase_done" if rc == 0 else "fail",
        stage="validate_net_diff", status="ok" if rc == 0 else "failed",
        rc=rc, command=command, hook_command=args.event_command or "",
    )
    return rc


def cmd_sprt(args: argparse.Namespace) -> int:
    script = tools_root() / "validate" / "run_net_sprt_pwa.sh"
    run_dir = event_run_dir(args.run, expand_path(args.net).parent)
    env = os.environ.copy()
    env.update({
        "NET": str(expand_path(args.net)),
        "GAMES": str(args.games),
        "CONCURRENCY": str(args.concurrency),
        "THREADS": str(args.threads),
        "HASH": str(args.hash),
        "TC": args.tc,
        "ELO0": str(args.elo0),
        "ELO1": str(args.elo1),
        "RESTART": args.restart,
    })
    if args.tag:
        env["TAG"] = args.tag
    if args.run:
        env["RUN"] = str(expand_path(args.run))
    if args.engine:
        env["ENGINE"] = str(expand_path(args.engine))
    if args.reference_net:
        env["INIT"] = str(expand_path(args.reference_net))
    if args.sprt:
        env["SPRT"] = str(expand_path(args.sprt))
    if args.book:
        env["BOOK"] = str(expand_path(args.book))
    if args.log_dir:
        env["LOG_DIR"] = str(expand_path(args.log_dir))
    if args.ntfy_url:
        env["NTFY_URL"] = args.ntfy_url
    command = [str(script)]
    emit_event(
        run_dir, "phase_start", stage="validate_sprt", status="running",
        command=command, hook_command=args.event_command or "",
        extra={"tag": args.tag or "", "games": args.games, "tc": args.tc},
    )
    rc = run(command, env=env)
    emit_event(
        run_dir, "phase_done" if rc == 0 else "fail",
        stage="validate_sprt", status="ok" if rc == 0 else "failed",
        rc=rc, command=command, hook_command=args.event_command or "",
        extra={"tag": args.tag or "", "games": args.games, "tc": args.tc},
    )
    return rc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Enyo NNUE candidates.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    static = subparsers.add_parser("static", help="Evaluate static dataset metrics.")
    static.add_argument("--net", required=True)
    static.add_argument("--data", required=True)
    static.add_argument("--rows", type=int, default=50000)
    static.add_argument("--skip", type=int, default=0)
    static.add_argument("--batch-size", type=int, default=4096)
    static.add_argument("--device", default="cpu")
    static.add_argument("--target-clamp", type=float, default=0.0)
    static.add_argument("--buckets", action="store_true")
    static.add_argument("--sources", action="store_true")
    static.add_argument("--run")
    static.add_argument("--event-command")
    static.add_argument(
        "--python",
        default=DEFAULTS.python,
        help="Python executable used for eval_dataset.py; defaults to the project venv when present.",
    )
    static.set_defaults(func=cmd_static)

    failure = subparsers.add_parser(
        "failure-suite",
        help="Run replay candidate/reference failure-suite comparison.",
    )
    failure.add_argument("logs", nargs="+")
    failure.add_argument("--candidate", required=True)
    failure.add_argument("--reference", required=True)
    failure.add_argument("--oracle", default="stockfish")
    failure.add_argument("--replay", default="replay")
    failure.add_argument("--threads", type=int, default=1)
    failure.add_argument("--jobs", type=int, default=1)
    failure.add_argument("--fixed-nodes", type=int, default=100000)
    failure.add_argument("--max-replay-nodes", type=int, default=0)
    failure.add_argument("--oracle-nodes", type=int, default=200000)
    failure.add_argument("--count", type=int, default=0)
    failure.add_argument("--output-dir", required=True)
    failure.add_argument("--stderr")
    failure.add_argument("--run")
    failure.add_argument("--event-command")
    failure.set_defaults(func=cmd_failure_suite)

    net_diff = subparsers.add_parser(
        "net-diff",
        help="Compare exported NNUE tensors and catch no-op exports.",
    )
    net_diff.add_argument("--candidate", required=True)
    net_diff.add_argument("--reference", required=True)
    net_diff.add_argument("--run")
    net_diff.add_argument("--event-command")
    net_diff.add_argument("--fail-if-identical", action="store_true")
    net_diff.set_defaults(func=cmd_net_diff)

    sprt = subparsers.add_parser("sprt", help="Run NNUE candidate SPRT.")
    sprt.add_argument("--net", required=True)
    sprt.add_argument("--tag")
    sprt.add_argument("--run")
    sprt.add_argument("--engine")
    sprt.add_argument("--reference-net")
    sprt.add_argument("--sprt")
    sprt.add_argument("--book")
    sprt.add_argument("--games", type=int, default=4000)
    sprt.add_argument("--concurrency", type=int, default=10)
    sprt.add_argument("--threads", type=int, default=2)
    sprt.add_argument("--hash", type=int, default=512)
    sprt.add_argument("--tc", default="2+0.02")
    sprt.add_argument("--elo0", type=float, default=0)
    sprt.add_argument("--elo1", type=float, default=8)
    sprt.add_argument("--restart", choices=("auto", "on", "off"), default="off")
    sprt.add_argument("--log-dir")
    sprt.add_argument("--ntfy-url")
    sprt.add_argument("--event-command")
    sprt.set_defaults(func=cmd_sprt)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
