#!/usr/bin/env python3
"""Launch and inspect Enyo NNUE experiment runs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import signal
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.events import emit_event, hook_from_config


def expand_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_run_dir() -> Path:
    return repo_root() / "runs" / time.strftime("%Y%m%d_%H%M%S")


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "YAML config requires PyYAML. Use JSON or install pyyaml."
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise SystemExit("run config must be an object")
    return data


def format_command(command: list[str], variables: dict[str, str]) -> list[str]:
    return [part.format(**variables) for part in command]


def count_lines(path: Path) -> int:
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def read_events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "events.jsonl"
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def score_progress(run_dir: Path, config: dict[str, Any] | None) -> tuple[int, int, int]:
    source_rows = count_lines(run_dir / "posgen" / "source.jsonl")
    shards = 0
    if config:
        create_args = config.get("create_args", {})
        if isinstance(create_args, dict):
            shards = int(create_args.get("score_shards", 0) or 0)
    if shards <= 0:
        shards = max(
            [
                int(path.name.split(".")[1])
                for path in (run_dir / "score" / "shards").glob("label.*.jsonl")
                if path.name.split(".")[1].isdigit()
            ] or [-1]
        ) + 1

    written = 0
    completed = 0
    for shard in range(shards):
        stats_path = run_dir / "score" / "shards" / f"label.{shard}.jsonl.stats.json"
        if stats_path.exists():
            try:
                stats = json.loads(stats_path.read_text(encoding="utf-8"))
                shard_written = int(stats.get("written", 0))
                written += shard_written
                completed += 1
                continue
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        written += count_lines(run_dir / "score" / "shards" / f"label.{shard}.jsonl")
    return written, source_rows, completed


def print_current_status(run_dir: Path, config: dict[str, Any] | None) -> None:
    events = read_events(run_dir)
    latest = events[-1] if events else {}
    event = str(latest.get("event", "unknown"))
    stage = str(latest.get("stage", "n/a"))
    status = str(latest.get("status", "unknown"))
    print("current:")
    print(f"  event: {event}")
    print(f"  stage: {stage}")
    print(f"  status: {status}")
    if latest.get("time"):
        print(f"  time: {latest['time']}")
    if latest.get("rc") is not None:
        print(f"  rc: {latest['rc']}")
    if latest.get("log"):
        print(f"  log: {latest['log']}")

    if stage.startswith("score_") or (run_dir / "score" / "shards").exists():
        written, source_rows, completed = score_progress(run_dir, config)
        if source_rows:
            percent = written * 100.0 / source_rows
            print(f"  score_progress: {written}/{source_rows} ({percent:.1f}%)")
        else:
            print(f"  score_progress: {written}/unknown")
        print(f"  score_shards_completed: {completed}")


def candidate_nets(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("train/*/model.nn"))


def validation_commands(run_dir: Path, net: Path) -> list[str]:
    validate = repo_root() / "tools" / "validate" / "validate.py"
    tag = f"{run_dir.name}_smoke"
    event_command = '"$HOME/scripts/nnue_event_ntfy.sh"'
    return [
        (
            f"{validate} static --net {net} --data {run_dir / 'pack' / 'train'} "
            f"--rows 100000 --buckets --event-command {event_command}"
        ),
        (
            f"{validate} sprt --net {net} --run {run_dir} --games 1000 "
            f"--tag {tag} --event-command {event_command}"
        ),
    ]


def completion_extra(run_dir: Path) -> dict[str, Any]:
    nets = candidate_nets(run_dir)
    if not nets:
        checkpoint_dirs = sorted(
            path for path in run_dir.glob("train/*/checkpoints") if path.is_dir()
        )
        checkpoint_files = sorted(
            path
            for directory in checkpoint_dirs
            for path in directory.rglob("*")
            if path.is_file()
        )
        if not checkpoint_dirs:
            return {}
        return {
            "bullet_checkpoint_dirs": [str(path) for path in checkpoint_dirs],
            "bullet_checkpoints": [str(path) for path in checkpoint_files],
        }
    net = nets[-1]
    commands = validation_commands(run_dir, net)
    return {
        "candidate_net": str(net),
        "candidate_nets": [str(path) for path in nets],
        "validation_commands": commands,
        "validate_static": commands[0],
        "validate_sprt": commands[1],
    }


def cmd_launch(args: argparse.Namespace) -> int:
    config_path = expand_path(args.config)
    config = load_config(config_path)
    run_dir = expand_path(config["run"]) if "run" in config else default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    for dirname in ("logs", "assets", "posgen", "score", "pack", "train", "validate"):
        (run_dir / dirname).mkdir(exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(
            json.dumps({
                "config": str(config_path),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "run": str(run_dir),
            }, indent=2) + "\n",
            encoding="utf-8",
        )
    steps = config.get("steps", [])
    if not isinstance(steps, list):
        raise SystemExit("config 'steps' must be a list")
    hook_command = args.on_event or hook_from_config(config)

    variables = {
        "run": str(run_dir),
        "repo": str(expand_path(config.get("repo", str(repo_root())))),
        "assets": str(run_dir / "assets"),
        "posgen": str(run_dir / "posgen"),
        "score": str(run_dir / "score"),
        "pack": str(run_dir / "pack"),
        "train": str(run_dir / "train"),
        "validate": str(run_dir / "validate"),
        "logs": str(run_dir / "logs"),
    }
    variables.update({str(k): str(v) for k, v in config.get("vars", {}).items()})

    log_path = run_dir / "runs.log"
    current_proc: subprocess.Popen[str] | None = None

    def handle_signal(signum: int, _frame: object) -> None:
        if current_proc and current_proc.poll() is None:
            current_proc.terminate()
        emit_event(
            run_dir,
            "fail",
            status=f"signal={signum}",
            rc=128 + signum,
            message="run launcher interrupted",
            log=log_path,
            hook_command=hook_command,
        )
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    emit_event(
        run_dir,
        "start",
        status="running",
        message=f"steps={len(steps)}",
        log=log_path,
        hook_command=hook_command,
        extra={"config": str(config_path)},
    )

    with log_path.open("a", encoding="utf-8") as log:
        for step in steps:
            if not isinstance(step, dict):
                raise SystemExit("each step must be an object")
            name = str(step.get("name", "step"))
            command = step.get("command")
            if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
                raise SystemExit(f"step {name}: command must be a string list")
            formatted = format_command(command, variables)
            marker = run_dir / f"{name}.done"
            if marker.exists() and not args.force:
                print(f"skip {name}: {marker} exists", flush=True)
                emit_event(
                    run_dir,
                    "phase_skip",
                    stage=name,
                    status="done marker exists",
                    log=log_path,
                    command=formatted,
                    hook_command=hook_command,
                )
                continue
            print(f"start {name}: {' '.join(formatted)}", flush=True)
            log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} start {name}: {' '.join(formatted)}\n")
            log.flush()
            emit_event(
                run_dir,
                "phase_start",
                stage=name,
                status="running",
                log=log_path,
                command=formatted,
                hook_command=hook_command,
            )
            current_proc = subprocess.Popen(
                formatted,
                cwd=variables["repo"],
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            rc = current_proc.wait()
            current_proc = None
            log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} done {name} rc={rc}\n")
            log.flush()
            if rc != 0:
                print(f"failed {name}: rc={rc}; log={log_path}", flush=True)
                emit_event(
                    run_dir,
                    "fail",
                    stage=name,
                    status="failed",
                    rc=rc,
                    log=log_path,
                    command=formatted,
                    hook_command=hook_command,
                )
                return rc
            marker.write_text(f"rc=0\n", encoding="utf-8")
            print(f"done {name}", flush=True)
            emit_event(
                run_dir,
                "phase_done",
                stage=name,
                status="ok",
                rc=rc,
                log=log_path,
                command=formatted,
                hook_command=hook_command,
            )
    extra = completion_extra(run_dir)
    print(f"run complete: {run_dir}", flush=True)
    if extra.get("candidate_net"):
        print(f"candidate net: {extra['candidate_net']}", flush=True)
    for checkpoint_dir in extra.get("bullet_checkpoint_dirs", []):
        print(f"bullet checkpoints: {checkpoint_dir}", flush=True)
    commands = extra.get("validation_commands", [])
    if commands:
        print("validate:", flush=True)
        for command in commands:
            print(f"  {command}", flush=True)
    emit_event(
        run_dir,
        "done",
        status="ok",
        rc=0,
        message="run complete",
        log=log_path,
        hook_command=hook_command,
        extra=extra,
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    run_dir = expand_path(args.run)
    config = None
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            config = load_config(config_path)
        except SystemExit:
            config = None
    print(f"run={run_dir}")
    print_current_status(run_dir, config)
    for name in (
        "config.yml",
        "config.json",
        "manifest.json",
        "status.json",
        "posgen/selfplay.pgn",
        "posgen/positions.jsonl",
        "posgen/source.jsonl",
        "score/labeled.jsonl",
        "pack/train/meta.json",
        "pack/meta.json",
    ):
        path = run_dir / name
        if path.exists():
            if path.is_file() and path.suffix == ".jsonl":
                try:
                    with path.open(encoding="utf-8", errors="replace") as handle:
                        rows = sum(1 for _ in handle)
                    print(f"{name}: exists rows={rows}")
                except OSError:
                    print(f"{name}: exists")
            else:
                print(f"{name}: exists")
        else:
            print(f"{name}: missing")

    for candidate in sorted(run_dir.glob("train/*/model.nn")):
        print(f"candidate: {candidate}")
    if args.tail > 0:
        for log in (run_dir / "run.log", run_dir / "runs.log", run_dir / "logs" / "run.log"):
            if not log.exists():
                continue
            print(f"{log.name} tail:")
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines[-args.tail:]:
                print(line)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch and inspect NNUE runs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch = subparsers.add_parser("launch", help="Launch a config-defined run.")
    launch.add_argument("config")
    launch.add_argument("--force", action="store_true")
    launch.add_argument(
        "--on-event",
        help="Optional hook command. Event JSON is passed on stdin and in NNUE_RUN_EVENT_JSON.",
    )
    launch.set_defaults(func=cmd_launch)

    status = subparsers.add_parser("status", help="Inspect a run directory.")
    status.add_argument("run")
    status.add_argument("--tail", type=int, default=0)
    status.set_defaults(func=cmd_status)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
