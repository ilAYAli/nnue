#!/usr/bin/env python3
"""Validate Enyo NNUE candidates."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.events import emit_event


def expand_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


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
        sys.executable,
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


def cmd_engine_static(args: argparse.Namespace) -> int:
    script = tools_root() / "validate" / "eval_jsonl_engine.py"
    run_dir = event_run_dir(args.run, expand_path(args.jsonl).parent)
    command = [
        sys.executable,
        str(script),
        "--engine", str(expand_path(args.engine)),
        "--net", str(expand_path(args.net)),
        "--jsonl", str(expand_path(args.jsonl)),
        "--rows", str(args.rows),
        "--skip", str(args.skip),
        "--score-field", args.score_field,
        "--threads", str(args.threads),
        "--hash", str(args.hash),
        "--timeout", str(args.timeout),
        "--progress", str(args.progress),
    ]
    if args.buckets:
        command.append("--buckets")
    if args.sources:
        command.append("--sources")
    emit_event(
        run_dir, "phase_start", stage="validate_engine_static",
        status="running", command=command,
        hook_command=args.event_command or "",
    )
    rc = run(command)
    emit_event(
        run_dir, "phase_done" if rc == 0 else "fail",
        stage="validate_engine_static", status="ok" if rc == 0 else "failed",
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


def cmd_move_gate(args: argparse.Namespace) -> int:
    script = tools_root() / "validate" / "eval_move_gate.py"
    run_dir = event_run_dir(args.run, expand_path(args.cases).parent)
    command = [
        sys.executable,
        str(script),
        "--cases", str(expand_path(args.cases)),
        "--engine", str(expand_path(args.engine)),
        "--baseline-net", str(expand_path(args.baseline_net)),
        "--candidate-net", str(expand_path(args.candidate_net)),
        "--threads", str(args.threads),
        "--hash", str(args.hash),
        "--timeout", str(args.timeout),
        "--limit", str(args.limit),
    ]
    if args.output:
        command += ["--output", str(expand_path(args.output))]
    if args.summary_json:
        command += ["--summary-json", str(expand_path(args.summary_json))]
    if args.fail_if_candidate_below_baseline:
        command.append("--fail-if-candidate-below-baseline")
    if args.fail_if_regressed_above is not None:
        command += ["--fail-if-regressed-above", str(args.fail_if_regressed_above)]
    if args.fail_if_fixed_below is not None:
        command += ["--fail-if-fixed-below", str(args.fail_if_fixed_below)]
    if args.fail_if_delta_below is not None:
        command += ["--fail-if-delta-below", str(args.fail_if_delta_below)]
    if args.fail_if_loss_weighted_delta_below is not None:
        command += [
            "--fail-if-loss-weighted-delta-below",
            str(args.fail_if_loss_weighted_delta_below),
        ]
    emit_event(
        run_dir, "phase_start", stage="validate_move_gate",
        status="running", command=command,
        hook_command=args.event_command or "",
    )
    rc = run(command)
    emit_event(
        run_dir, "phase_done" if rc == 0 else "fail",
        stage="validate_move_gate", status="ok" if rc == 0 else "failed",
        rc=rc, command=command, hook_command=args.event_command or "",
    )
    return rc


def cmd_sprt(args: argparse.Namespace) -> int:
    script = tools_root() / "validate" / "run_net_sprt_pwa.sh"
    run_dir = event_run_dir(args.run, expand_path(args.net).parent)
    tag = args.tag or ""
    log_dir = expand_path(args.log_dir) if args.log_dir else (
        run_dir / (tag or "sprt") / f"sprt_confirm_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_log = log_dir / "run.log"
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
        "LOG_DIR": str(log_dir),
    })
    if tag:
        env["TAG"] = tag
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
    if args.ntfy_url:
        env["NTFY_URL"] = args.ntfy_url
    command = [str(script)]
    emit_event(
        run_dir, "phase_start", stage="validate_sprt", status="running",
        command=command, hook_command=args.event_command or "",
        log=run_log,
        extra={"tag": tag, "games": args.games, "tc": args.tc},
    )
    rc = run(command, env=env)
    emit_event(
        run_dir, "done" if rc == 0 else "fail",
        stage="validate_sprt", status="ok" if rc == 0 else "failed",
        rc=rc, command=command, hook_command=args.event_command or "",
        log=run_log,
        extra={"tag": tag, "games": args.games, "tc": args.tc},
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
    static.set_defaults(func=cmd_static)

    engine_static = subparsers.add_parser(
        "engine-static",
        help="Evaluate scored JSONL metrics through Enyo evalnet.",
    )
    engine_static.add_argument("--engine", required=True)
    engine_static.add_argument("--net", required=True)
    engine_static.add_argument("--jsonl", required=True)
    engine_static.add_argument("--rows", type=int, default=1000)
    engine_static.add_argument("--skip", type=int, default=0)
    engine_static.add_argument("--score-field", default="score")
    engine_static.add_argument("--threads", type=int, default=1)
    engine_static.add_argument("--hash", type=int, default=128)
    engine_static.add_argument("--timeout", type=float, default=10.0)
    engine_static.add_argument("--progress", type=int, default=1000)
    engine_static.add_argument("--buckets", action="store_true")
    engine_static.add_argument("--sources", action="store_true")
    engine_static.add_argument("--run")
    engine_static.add_argument("--event-command")
    engine_static.set_defaults(func=cmd_engine_static)

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

    move_gate = subparsers.add_parser(
        "move-gate",
        help="Evaluate a fixed parent-position move-choice gate through evalnet.",
    )
    move_gate.add_argument("--cases", required=True)
    move_gate.add_argument("--engine", required=True)
    move_gate.add_argument("--baseline-net", required=True)
    move_gate.add_argument("--candidate-net", required=True)
    move_gate.add_argument("--threads", type=int, default=1)
    move_gate.add_argument("--hash", type=int, default=64)
    move_gate.add_argument("--timeout", type=float, default=10.0)
    move_gate.add_argument("--limit", type=int, default=0)
    move_gate.add_argument("--output")
    move_gate.add_argument("--summary-json")
    move_gate.add_argument("--fail-if-candidate-below-baseline", action="store_true")
    move_gate.add_argument("--fail-if-regressed-above", type=int)
    move_gate.add_argument("--fail-if-fixed-below", type=int)
    move_gate.add_argument("--fail-if-delta-below", type=float)
    move_gate.add_argument("--fail-if-loss-weighted-delta-below", type=float)
    move_gate.add_argument("--run")
    move_gate.add_argument("--event-command")
    move_gate.set_defaults(func=cmd_move_gate)

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
