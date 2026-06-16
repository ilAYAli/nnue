#!/usr/bin/env python3
"""Small build.json-driven NNUE run frontend.

The point of this tool is not to know every old NNUE workflow.  It makes the
active hypothesis, changed variables, current stage, and next gate visible.
The actual commands stay explicit in build.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


TOP_LEVEL_KEYS = {
    "run",
    "description",
    "hypothesis",
    "changed_variables",
    "run_dir",
    "hooks",
    "env",
    "stages",
    "gates",
}
REPLACED_HOT_PATH_TOOLS = {
    "tools/posgen/mix_jsonl.py": "nnue-mix-jsonl",
    "tools/bullet/jsonl_to_bullet_text.py": "nnue-jsonl-to-bullet-text",
}
REPLACED_STAGE_COMMANDS = {
    "tools/train/train.py": "./nnue-train supervised or ./nnue-train eval",
    "tools/train/train_pairwise.py": "./nnue-train pairwise",
    "tools/train/train_move_policy.py": "./nnue-train move-policy",
}
HOT_PATH_DIRS = (
    "tools/posgen/",
    "tools/pack/",
    "tools/bullet/",
    "tools/score/",
    "tools/validate/",
)
STAGE_KEYS = {
    "name",
    "why",
    "command",
    "env",
    "timeout_seconds",
    "artifacts",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a JSON object")
    return data


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def expand_path(text: str | Path) -> Path:
    return Path(os.path.expandvars(str(text))).expanduser().resolve()


def repo_relative_path(text: str | Path) -> Path:
    expanded = Path(os.path.expandvars(str(text))).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (repo_root() / expanded).resolve()


def render_template(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        try:
            return value.format(**context)
        except KeyError as exc:
            raise SystemExit(f"unknown template key {{{exc.args[0]}}} in: {value}") from exc
    if isinstance(value, list):
        return [render_template(item, context) for item in value]
    if isinstance(value, dict):
        return {str(key): render_template(item, context) for key, item in value.items()}
    return value


def command_text(command: str | list[Any]) -> str:
    if isinstance(command, str):
        return command
    return shlex.join(str(part) for part in command)


def git_output(args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout


def relative_to_repo(path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def is_hot_path_python(path: str) -> bool:
    return path.endswith(".py") and path.startswith(HOT_PATH_DIRS)


def validate_config(config: dict[str, Any], *, path: Path) -> None:
    unknown = sorted(set(config) - TOP_LEVEL_KEYS)
    if unknown:
        raise SystemExit(f"{path}: unknown top-level field(s): {', '.join(unknown)}")
    for key in ("run", "hypothesis", "changed_variables", "stages"):
        if key not in config:
            raise SystemExit(f"{path}: missing required field: {key}")
    if not isinstance(config["run"], str) or not config["run"].strip():
        raise SystemExit(f"{path}: run must be a non-empty string")
    if not isinstance(config["hypothesis"], str) or not config["hypothesis"].strip():
        raise SystemExit(f"{path}: hypothesis must be a non-empty string")
    if not isinstance(config["changed_variables"], dict) or not config["changed_variables"]:
        raise SystemExit(f"{path}: changed_variables must be a non-empty object")
    stages = config["stages"]
    if not isinstance(stages, list) or not stages:
        raise SystemExit(f"{path}: stages must be a non-empty array")
    names: set[str] = set()
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise SystemExit(f"{path}: stages[{index}] must be an object")
        unknown_stage = sorted(set(stage) - STAGE_KEYS)
        if unknown_stage:
            raise SystemExit(
                f"{path}: stages[{index}] unknown field(s): {', '.join(unknown_stage)}"
            )
        name = str(stage.get("name", "")).strip()
        if not name:
            raise SystemExit(f"{path}: stages[{index}].name is required")
        if name in names:
            raise SystemExit(f"{path}: duplicate stage name: {name}")
        names.add(name)
        if "command" not in stage:
            raise SystemExit(f"{path}: stages[{index}].command is required")
        command = stage["command"]
        if not isinstance(command, (str, list)) or not command:
            raise SystemExit(f"{path}: stages[{index}].command must be a string or array")
    for key in ("hooks", "env", "gates"):
        if key in config and not isinstance(config[key], dict if key != "gates" else list):
            expected = "object" if key != "gates" else "array"
            raise SystemExit(f"{path}: {key} must be an {expected}")


def load_config(path_text: str | Path) -> tuple[Path, dict[str, Any]]:
    path = expand_path(path_text)
    config = read_json(path)
    validate_config(config, path=path)
    return path, config


def compiled_tool_path(name: str) -> Path | None:
    candidates = [
        repo_root() / "build" / "fast" / name,
        repo_root() / "build" / name,
    ]
    for path in candidates:
        if path.exists() and os.access(path, os.X_OK):
            return path
    for item in os.environ.get("PATH", "").split(os.pathsep):
        path = Path(item) / name
        if path.exists() and os.access(path, os.X_OK):
            return path
    return None


def command_uses_replaced_tool(command: str | list[Any]) -> tuple[str, str] | None:
    text = command_text(command)
    for old_tool, new_tool in (REPLACED_HOT_PATH_TOOLS | REPLACED_STAGE_COMMANDS).items():
        if old_tool in text:
            return old_tool, new_tool
    return None


def check_config_policy(config: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for stage in config["stages"]:
        replaced = command_uses_replaced_tool(stage["command"])
        if replaced:
            old_tool, new_tool = replaced
            failures.append(
                f"stage '{stage['name']}' uses slow Python hot-path {old_tool}; use {new_tool}"
            )
    return failures


def check_compiled_tools() -> list[str]:
    failures: list[str] = []
    for new_tool in sorted(set(REPLACED_HOT_PATH_TOOLS.values())):
        if compiled_tool_path(new_tool) is None:
            failures.append(f"compiled hot-path tool not found: {new_tool}; run cmake build first")
    return failures


def check_git_hygiene(config_path: Path) -> list[str]:
    failures: list[str] = []
    allowed_dirty = {
        relative_to_repo(config_path),
        "IMPROVEMENT_PLAN.md",
    }
    for raw in git_output(["status", "--porcelain=v1"]).splitlines():
        if not raw:
            continue
        status = raw[:2]
        path = raw[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path in allowed_dirty:
            continue
        if status == "??":
            failures.append(f"untracked source file before launch: {path}")
        else:
            failures.append(f"dirty source file before launch: {path}")

    for raw in git_output(["diff", "--name-status", "--cached"]).splitlines():
        parts = raw.split("\t")
        if len(parts) >= 2 and parts[0].startswith("A") and is_hot_path_python(parts[-1]):
            failures.append(f"new staged Python hot-path tool is not allowed: {parts[-1]}")

    for raw in git_output(["ls-files", "--others", "--exclude-standard"]).splitlines():
        if is_hot_path_python(raw):
            failures.append(f"new untracked Python hot-path tool is not allowed: {raw}")
    return failures


def run_doctor_checks(
    config_path: Path,
    config: dict[str, Any],
    *,
    skip_git: bool = False,
    require_tools: bool = True,
) -> list[str]:
    failures = check_config_policy(config)
    if require_tools:
        failures.extend(check_compiled_tools())
    if not skip_git:
        failures.extend(check_git_hygiene(config_path))
    return failures


def print_doctor_report(failures: list[str]) -> None:
    if not failures:
        print("doctor: ok")
        return
    print("doctor: failed")
    for failure in failures:
        print(f"  - {failure}")


def run_dir_from_config(config: dict[str, Any]) -> Path:
    context = {
        "run": str(config["run"]),
        "repo": str(repo_root()),
    }
    raw = str(config.get("run_dir") or "runs/{run}")
    return repo_relative_path(str(render_template(raw, context)))


def active_path() -> Path:
    override = os.environ.get("NNUE_RUN_ACTIVE_PATH", "")
    if override:
        return expand_path(override)
    return repo_root() / "runs" / "active.json"


def load_status(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "status.json"
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError, SystemExit):
        return {}


def write_status(run_dir: Path, config: dict[str, Any], **updates: Any) -> dict[str, Any]:
    existing = load_status(run_dir)
    payload = {
        "run": str(config["run"]),
        "description": str(config.get("description", "")),
        "hypothesis": str(config["hypothesis"]),
        "changed_variables": config["changed_variables"],
        "stage_count": len(config["stages"]),
        "updated_at": now(),
    }
    payload.update(existing)
    payload.update(updates)
    payload["updated_at"] = now()
    write_json(run_dir / "status.json", payload)
    write_json(active_path(), {"run": str(config["run"]), "run_dir": str(run_dir)})
    return payload


def event_hook(config: dict[str, Any]) -> str:
    hooks = config.get("hooks", {})
    if isinstance(hooks, dict):
        command = str(hooks.get("event_command") or "")
        if command and "/" in command and not Path(command).is_absolute():
            return str(repo_root() / command)
        return command
    return ""


def emit_event(
    run_dir: Path,
    config: dict[str, Any],
    event: str,
    *,
    stage: str = "",
    status: str = "",
    rc: int | None = None,
    message: str = "",
    log: str | Path | None = None,
    command: str | list[Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "time": now(),
        "event": event,
        "run": str(run_dir),
        "name": str(config["run"]),
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
        payload["command"] = command if isinstance(command, str) else [str(part) for part in command]
    append_jsonl(run_dir / "events.jsonl", payload)

    hook = event_hook(config) or os.environ.get("NNUE_EVENT_COMMAND", "")
    if not hook:
        return
    env = os.environ.copy()
    line = json.dumps(payload, sort_keys=True)
    env.update({
        "NNUE_RUN_EVENT_JSON": line,
        "NNUE_RUN_EVENT": event,
        "NNUE_RUN_STAGE": stage,
        "NNUE_RUN_DIR": str(run_dir),
    })
    subprocess.run(
        hook,
        input=line + "\n",
        text=True,
        shell=True,
        env=env,
        timeout=int(os.environ.get("NNUE_EVENT_HOOK_TIMEOUT", "30")),
        check=False,
    )


def stage_context(config: dict[str, Any], run_dir: Path, stage: dict[str, Any]) -> dict[str, Any]:
    context = {
        "repo": str(repo_root()),
        "run": str(config["run"]),
        "run_dir": str(run_dir),
        "stage": str(stage["name"]),
        "logs": str(run_dir / "logs"),
        "train": str(run_dir / "train"),
        "assets": str(run_dir / "assets"),
    }
    for key, value in config["changed_variables"].items():
        context[f"var_{key}"] = value
    return context


def rendered_stage(config: dict[str, Any], run_dir: Path, stage: dict[str, Any]) -> dict[str, Any]:
    return render_template(stage, stage_context(config, run_dir, stage))


def done_marker(run_dir: Path, stage_name: str) -> Path:
    return run_dir / "stages" / f"{stage_name}.done"


def run_stage(
    run_dir: Path,
    config: dict[str, Any],
    stage: dict[str, Any],
    *,
    index: int,
) -> int:
    rendered = rendered_stage(config, run_dir, stage)
    name = str(rendered["name"])
    command = rendered["command"]
    log = run_dir / "logs" / f"{index + 1:02d}-{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    write_status(
        run_dir,
        config,
        state="running",
        current_stage=name,
        current_stage_index=index + 1,
        current_command=command_text(command),
        current_log=str(log),
    )
    emit_event(run_dir, config, "phase_start", stage=name, status="running", log=log, command=command)

    env = os.environ.copy()
    for source in (config.get("env", {}), rendered.get("env", {})):
        if isinstance(source, dict):
            for key, value in source.items():
                env[str(key)] = str(render_template(value, stage_context(config, run_dir, stage)))

    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {command_text(command)}\n")
        handle.flush()
        if isinstance(command, str):
            proc = subprocess.Popen(
                command,
                cwd=repo_root(),
                shell=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
        else:
            proc = subprocess.Popen(
                [str(part) for part in command],
                cwd=repo_root(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
        try:
            rc = proc.wait(timeout=rendered.get("timeout_seconds"))
        except subprocess.TimeoutExpired:
            proc.terminate()
            rc = 124

    if rc == 0:
        done_marker(run_dir, name).parent.mkdir(parents=True, exist_ok=True)
        done_marker(run_dir, name).write_text(now() + "\n", encoding="utf-8")
        emit_event(run_dir, config, "phase_done", stage=name, status="ok", rc=rc, log=log)
    else:
        write_status(run_dir, config, state="failed", failed_stage=name, failed_rc=rc)
        emit_event(run_dir, config, "fail", stage=name, status="failed", rc=rc, log=log)
    return rc


def print_plan(config: dict[str, Any]) -> None:
    run_dir = run_dir_from_config(config)
    print(f"Run: {config['run']}")
    if config.get("description"):
        print(f"Description: {config['description']}")
    print(f"Run dir: {run_dir}")
    print(f"Hypothesis: {config['hypothesis']}")
    print("Changed variables:")
    for key, value in config["changed_variables"].items():
        print(f"  {key}: {value}")
    print("Stages:")
    for index, stage in enumerate(config["stages"], start=1):
        rendered = rendered_stage(config, run_dir, stage)
        why = str(rendered.get("why", "")).strip()
        suffix = f" - {why}" if why else ""
        print(f"  {index}. {rendered['name']}{suffix}")
        print(f"     {command_text(rendered['command'])}")
    gates = config.get("gates", [])
    if gates:
        print("Gates:")
        for gate in gates:
            print(f"  {gate}")


def latest_run_from_active() -> Path:
    path = active_path()
    if not path.exists():
        raise SystemExit("no active run; pass a run directory or run name")
    data = read_json(path)
    return expand_path(str(data["run_dir"]))


def select_run(run: str | None) -> Path:
    if not run:
        return latest_run_from_active()
    direct = expand_path(run)
    if direct.exists():
        return direct
    candidate = repo_root() / "runs" / run
    if candidate.exists():
        return candidate
    raise SystemExit(f"run not found: {run}")


def print_status(run_dir: Path, *, json_output: bool = False) -> None:
    status = load_status(run_dir)
    if not status:
        raise SystemExit(f"no status found in {run_dir}")
    if json_output:
        print(json.dumps(status, indent=2, sort_keys=True))
        return
    state = str(status.get("state", "unknown"))
    current = str(status.get("current_stage") or status.get("failed_stage") or "n/a")
    idx = status.get("current_stage_index", "?")
    count = status.get("stage_count", "?")
    print(f"Run: {status.get('run', run_dir.name)}")
    print(f"State: {state}")
    print(f"Stage: {idx}/{count} {current}")
    print(f"Hypothesis: {status.get('hypothesis', '')}")
    print("Changed variables:")
    changed = status.get("changed_variables", {})
    if isinstance(changed, dict):
        for key, value in changed.items():
            print(f"  {key}: {value}")
    if status.get("current_log"):
        print(f"Log: {status['current_log']}")
    if status.get("decision"):
        print(f"Decision: {status['decision']}")
    next_stage = status.get("next_stage")
    if next_stage:
        print(f"Next: {next_stage}")


def execute(config_path: Path, config: dict[str, Any], *, force: bool, resume: bool) -> int:
    run_dir = run_dir_from_config(config)
    if run_dir.exists() and not (force or resume):
        raise SystemExit(f"{run_dir} already exists; use resume or --force")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "source_config.json", {"path": str(config_path)})
    (run_dir / "logs").mkdir(exist_ok=True)
    if resume:
        (run_dir / "stop.json").unlink(missing_ok=True)
    write_status(run_dir, config, state="running", started_at=load_status(run_dir).get("started_at") or now())
    emit_event(run_dir, config, "start", status="running")

    for index, stage in enumerate(config["stages"]):
        name = str(stage["name"])
        if done_marker(run_dir, name).exists():
            continue
        if (run_dir / "stop.json").exists():
            write_status(run_dir, config, state="stopped", next_stage=name)
            emit_event(run_dir, config, "stop", stage=name, status="stopped")
            return 130
        rc = run_stage(run_dir, config, stage, index=index)
        if rc != 0:
            return rc

    write_status(run_dir, config, state="done", current_stage="done", decision="complete")
    emit_event(run_dir, config, "done", status="ok")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    _, config = load_config(args.config)
    print_plan(config)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    path, config = load_config(args.config)
    failures = run_doctor_checks(
        path,
        config,
        skip_git=args.skip_git,
        require_tools=not args.skip_tools,
    )
    print_doctor_report(failures)
    return 1 if failures else 0


def cmd_start(args: argparse.Namespace) -> int:
    path, config = load_config(args.config)
    if args.dry_run:
        print_plan(config)
        return 0
    if not args.unsafe_skip_doctor:
        failures = run_doctor_checks(path, config)
        if failures:
            print_doctor_report(failures)
            return 1
    return execute(path, config, force=args.force, resume=False)


def cmd_resume(args: argparse.Namespace) -> int:
    run_dir = select_run(args.run)
    config = read_json(run_dir / "config.json")
    validate_config(config, path=run_dir / "config.json")
    return execute(run_dir / "config.json", config, force=False, resume=True)


def cmd_stop(args: argparse.Namespace) -> int:
    run_dir = select_run(args.run)
    status = load_status(run_dir)
    config = read_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {
        "run": status.get("run", run_dir.name),
        "hypothesis": status.get("hypothesis", ""),
        "changed_variables": status.get("changed_variables", {}),
        "stages": [],
    }
    write_json(run_dir / "stop.json", {"time": now()})
    write_status(run_dir, config, state="stopped")
    emit_event(run_dir, config, "stop", status="stopped")
    print(f"stopped: {run_dir}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    print_status(select_run(args.run), json_output=args.json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple build.json NNUE run frontend.")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Show the run plan and changed variables.")
    plan.add_argument("-c", "--config", default="build.json")
    plan.set_defaults(func=cmd_plan)

    doctor = sub.add_parser("doctor", help="Check whether this repo/config may launch a run.")
    doctor.add_argument("-c", "--config", default="build.json")
    doctor.add_argument("--skip-git", action="store_true", help=argparse.SUPPRESS)
    doctor.add_argument("--skip-tools", action="store_true", help=argparse.SUPPRESS)
    doctor.set_defaults(func=cmd_doctor)

    start = sub.add_parser("start", help="Start a build.json run.")
    start.add_argument("-c", "--config", default="build.json")
    start.add_argument("--dry-run", action="store_true")
    start.add_argument("--force", action="store_true")
    start.add_argument("--unsafe-skip-doctor", action="store_true", help=argparse.SUPPRESS)
    start.set_defaults(func=cmd_start)

    resume = sub.add_parser("resume", help="Resume an incomplete run.")
    resume.add_argument("run", nargs="?")
    resume.set_defaults(func=cmd_resume)

    stop = sub.add_parser("stop", help="Stop after the current stage finishes.")
    stop.add_argument("run", nargs="?")
    stop.set_defaults(func=cmd_stop)

    status = sub.add_parser("status", help="Show current run status.")
    status.add_argument("run", nargs="?")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
