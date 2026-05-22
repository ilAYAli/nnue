#!/usr/bin/env python3
"""Run the Enyo replay failure suite for two NNUE files.

This creates isolated HOME/config directories for candidate and reference
engines, then runs replay_failure_suite.py. It exists to keep tmux launches
short and avoid brittle shell heredocs for wrapper setup.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.defaults import DEFAULTS
from lib.events import emit_event


DEFAULT_LOG_NAMES = [
    "ArasanX_vs_EnyoBot_5Pms0ZXh.log",
    "ArasanX_vs_EnyoBot_7pzXRQFZ.log",
    "ArasanX_vs_EnyoBot_Q0a5I19l.log",
    "Bot1nokk_vs_EnyoBot_mEIqToS1.log",
    "caissa-x_vs_EnyoBot_0waEuGxc.log",
    "CloudNetBot_vs_EnyoBot_KTS7zzI9.log",
    "EnyoBot_vs_Bot1nokk_5j5IYTv6.log",
    "EnyoBot_vs_DarkOnBot_pm9SUqFN.log",
    "EnyoBot_vs_Lynx_BOT_jjThVRPN.log",
    "Hypersion_vs_EnyoBot_npmgxvIO.log",
    "odonata-bot_vs_EnyoBot_u9u9wyRV_oot.log",
    "SF_Bot1nok_vs_EnyoBot_SbzD89kx.log",
    "stage270_vs_EnyoBot_kp3inZBb.log",
    "styx_reckless_vs_EnyoBot_z3PAJH4r.log",
]


def expand(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def write_engine_home(home: Path, *, nnue_file: Path, logfile: str) -> None:
    config_dir = home / ".config" / "enyo"
    config_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "toggles": {"use_lmr": True},
        "constants": {
            "threads": 1,
            "Hash": 256,
            "nnue_file": str(nnue_file),
            "logfile": logfile,
        },
    }
    (config_dir / "settings.json").write_text(
        json.dumps(settings, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def write_wrapper(path: Path, *, home: Path, engine: Path) -> None:
    path.write_text(
        "#!/usr/bin/env bash\n"
        f"export HOME={json.dumps(str(home))}\n"
        f"exec {json.dumps(str(engine))} \"$@\"\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def default_logs(bugs_dir: Path) -> list[Path]:
    return [bugs_dir / name for name in DEFAULT_LOG_NAMES]


def read_summary(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            out[key] = value
    return out


def run_failure_suite(args: argparse.Namespace, out_dir: Path) -> int:
    script = Path(__file__).resolve().with_name("replay_failure_suite.py")
    command = [
        str(expand(args.python)),
        str(script),
        "--candidate", str(out_dir / "candidate.sh"),
        "--reference", str(out_dir / "reference.sh"),
        "--oracle", str(expand(args.oracle)),
        "--replay", str(expand(args.replay)),
        "--threads", str(args.threads),
        "--jobs", str(args.jobs),
        "--fixed-nodes", str(args.fixed_nodes),
        "--max-replay-nodes", str(args.max_replay_nodes),
        "--oracle-nodes", str(args.oracle_nodes),
        "--count", str(args.count),
        "--output-dir", str(out_dir / "out"),
        "--stderr", str(out_dir / "replay.stderr"),
    ]
    logs = [expand(log) for log in args.logs]
    if not logs:
        logs = default_logs(expand(args.bugs_dir))
    command.extend(str(log) for log in logs)

    log_path = out_dir / "failure_suite_stdout.log"
    def emit_failed(rc: int, message: str) -> int:
        emit_event(
            args.run,
            "fail",
            stage="validate_failure_suite",
            status="failed",
            rc=rc,
            log=log_path,
            command=command,
            message=message,
            hook_command=args.event_command or "",
        )
        return rc

    emit_event(
        args.run,
        "phase_start",
        stage="validate_failure_suite",
        status="running",
        log=log_path,
        command=command,
        hook_command=args.event_command or "",
    )
    try:
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
            rc = proc.wait()
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"ERROR: {exc}\n")
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return emit_failed(1, f"exception={exc}")

    summary = read_summary(out_dir / "out" / "summary.txt")
    if summary:
        print((out_dir / "out" / "summary.txt").read_text(encoding="utf-8"),
              end="")
    emit_event(
        args.run,
        "phase_done" if rc == 0 else "fail",
        stage="validate_failure_suite",
        status="ok" if rc == 0 else "failed",
        rc=rc,
        log=log_path,
        command=command,
        message=(
            f"sum_diff_cp={summary.get('sum_diff_cp', '?')} "
            f"worst_regression_cp={summary.get('worst_regression_cp', '?')}"
        ) if summary else "",
        hook_command=args.event_command or "",
    )
    return rc


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-net", required=True)
    ap.add_argument("--reference-net", default=DEFAULTS.reference_net)
    ap.add_argument("--engine", default="~/code/cpp/chess/enyo/build/enyo")
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--output-dir")
    ap.add_argument("--bugs-dir", default="~/code/cpp/chess/enyo/bugs")
    ap.add_argument("--logs", nargs="*", default=[])
    ap.add_argument("--oracle", default=DEFAULTS.score_engine)
    ap.add_argument("--replay", default="~/local/bin/replay")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--fixed-nodes", type=int, default=100000)
    ap.add_argument("--max-replay-nodes", type=int, default=0)
    ap.add_argument("--oracle-nodes", type=int, default=200000)
    ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--python", default=DEFAULTS.python)
    ap.add_argument("--event-command", default="")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    run_dir = expand(args.run)
    tag = args.tag or "failure-suite"
    if args.output_dir:
        out_dir = expand(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = run_dir / "validate" / f"{tag}-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_engine_home(
        out_dir / "candidate_home",
        nnue_file=expand(args.candidate_net),
        logfile=f"/tmp/enyo-{tag}-candidate.log",
    )
    write_engine_home(
        out_dir / "reference_home",
        nnue_file=expand(args.reference_net),
        logfile=f"/tmp/enyo-{tag}-reference.log",
    )
    engine = expand(args.engine)
    write_wrapper(out_dir / "candidate.sh", home=out_dir / "candidate_home",
                  engine=engine)
    write_wrapper(out_dir / "reference.sh", home=out_dir / "reference_home",
                  engine=engine)
    for wrapper in (out_dir / "candidate.sh", out_dir / "reference.sh"):
        if not wrapper.is_file() or not wrapper.stat().st_mode & 0o111:
            raise RuntimeError(f"wrapper not executable: {wrapper}")

    print(f"failure-suite out={out_dir}", flush=True)
    return run_failure_suite(args, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
