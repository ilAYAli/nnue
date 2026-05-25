#!/usr/bin/env python3
"""Export Bullet Enyo checkpoints and run search-target gates."""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bullet.bullet import ENYO_SUPPORTED_INPUT_BUCKETS, expand_enyo_input_buckets
from lib.defaults import DEFAULTS, repo_root
from lib.events import emit_event


def expand_user(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser()


def checkpoint_index(path: Path) -> int:
    match = re.search(r"-(\d+)$", path.parent.name)
    return int(match.group(1)) if match else -1


def selected_indices(value: str) -> set[int] | None:
    if not value:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def checkpoint_files(checkpoint_dir: Path, selection: set[int] | None) -> list[Path]:
    files = sorted(
        checkpoint_dir.glob("*-*/quantised.bin"),
        key=checkpoint_index,
    )
    if selection is not None:
        files = [path for path in files if checkpoint_index(path) in selection]
    if not files:
        raise SystemExit(f"no quantised.bin checkpoints found under {checkpoint_dir}")
    return files


def export_enyo_net(
    checkpoint: Path,
    output: Path,
    input_buckets: int,
    runtime_input_buckets: int,
    hidden: int,
    l2: int,
) -> None:
    raw = expand_enyo_input_buckets(
        checkpoint.read_bytes(),
        input_buckets,
        runtime_input_buckets=runtime_input_buckets,
        hidden=hidden,
        l2=l2,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)


def run(command: list[str], *, log_path: Path) -> int:
    print(" ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            handle.write(line)
        return proc.wait()


def read_summary(path: Path, checkpoint: int) -> list[str]:
    if not path.exists():
        return [f"{checkpoint} missing-summary"]
    lines = path.read_text(encoding="utf-8").splitlines()
    current = ""
    sections: dict[str, dict[str, str]] = {}
    wanted = {
        "targets", "top1", "top3", "candidate_better", "reference_better",
        "capped_sum_diff_cp", "worst_regression_cp",
    }
    for line in lines:
        if line.startswith("[") and line.endswith("]"):
            current = line.strip("[]")
            sections.setdefault(current, {})
            continue
        if current and "=" in line:
            key, value = line.split("=", 1)
            if key in wanted:
                sections.setdefault(current, {})[key] = value
    out: list[str] = []
    for split in ("compare:all", "compare:mate_like", "compare:non_mate"):
        if split not in sections:
            continue
        candidate_split = split.removeprefix("compare:")
        out.append(format_row(
            checkpoint,
            split,
            sections[split],
            sections.get(candidate_split, {}),
        ))
    return out


def format_row(
    checkpoint: int,
    split: str,
    values: dict[str, str],
    candidate_values: dict[str, str],
) -> str:
    return " ".join([
        str(checkpoint),
        split,
        values.get("targets", ""),
        candidate_values.get("top1", ""),
        candidate_values.get("top3", ""),
        values.get("candidate_better", ""),
        values.get("reference_better", ""),
        values.get("capped_sum_diff_cp", ""),
        values.get("worst_regression_cp", ""),
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Bullet Enyo checkpoints and run search-target gates."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--enyo-input-buckets",
        type=int,
        choices=list(ENYO_SUPPORTED_INPUT_BUCKETS),
        default=32,
        help="Input bucket layout used by the Bullet Enyo checkpoints.",
    )
    parser.add_argument(
        "--enyo-runtime-input-buckets",
        type=int,
        choices=list(ENYO_SUPPORTED_INPUT_BUCKETS),
        default=32,
        help="Runtime input bucket layout expected by the candidate engine.",
    )
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--l2", type=int, default=16)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--reference-net")
    parser.add_argument("--reference-engine")
    parser.add_argument(
        "--reference-csv",
        help="Reuse a previously written search_target_gate reference.csv.",
    )
    parser.add_argument(
        "--require-native-net-load",
        action="store_true",
        help="Fail if exported checkpoints are not loaded by the native NNUE loader.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--nodes", type=int, default=300000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cap", type=int, default=200)
    parser.add_argument("--checkpoints", default="")
    parser.add_argument("--python", default=DEFAULTS.python)
    parser.add_argument("--run")
    parser.add_argument("--event-command")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_dir = expand_user(args.checkpoint_dir).resolve()
    out_dir = expand_user(args.out_dir).resolve()
    run_dir = expand_user(args.run).resolve() if args.run else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    nets_dir = out_dir / "nets"
    summary_path = out_dir / "summary.txt"
    run_log = out_dir / "run.log"
    targets = expand_user(args.targets).resolve()
    engine = expand_user(args.engine).resolve()
    reference_engine = expand_user(args.reference_engine).resolve() if args.reference_engine else engine
    reference_net = expand_user(args.reference_net).resolve() if args.reference_net else None
    reference_csv = expand_user(args.reference_csv).resolve() if args.reference_csv else None
    if reference_csv is None and reference_net is None:
        raise SystemExit("--reference-net is required unless --reference-csv is set")
    python = str(expand_user(args.python))
    search_gate = repo_root() / "tools" / "validate" / "search_target_gate.py"

    emit_event(
        run_dir, "phase_start", stage="checkpoint_sweep_start", status="running",
        hook_command=args.event_command or "",
        extra={"checkpoint_dir": str(checkpoint_dir), "out_dir": str(out_dir)},
    )

    selected = selected_indices(args.checkpoints)
    rows = [
        "checkpoint split targets top1 top3 candidate_better reference_better "
        "capped_sum_diff_cp worst_regression_cp"
    ]
    rc = 0
    with run_log.open("w", encoding="utf-8") as handle:
        handle.write(f"checkpoint_dir={checkpoint_dir}\n")
        handle.write(f"targets={targets}\n")
        if reference_csv is not None:
            handle.write(f"reference_csv={reference_csv}\n")
        else:
            handle.write(f"reference_net={reference_net}\n")
        handle.write(f"nodes={args.nodes}\n")

    for checkpoint in checkpoint_files(checkpoint_dir, selected):
        idx = checkpoint_index(checkpoint)
        net = nets_dir / f"{checkpoint.parent.name}.nn"
        stage = f"checkpoint_{idx}"
        try:
            export_enyo_net(
                checkpoint,
                net,
                args.enyo_input_buckets,
                args.enyo_runtime_input_buckets,
                args.hidden,
                args.l2,
            )
        except SystemExit as exc:
            emit_event(
                run_dir, "fail", stage=stage, status="failed",
                hook_command=args.event_command or "",
                extra={"message": str(exc), "checkpoint": str(checkpoint)},
            )
            raise
        gate_dir = out_dir / f"search_{idx}"
        command = [
            python, str(search_gate),
            "--targets", str(targets),
            "--engine", str(engine),
            "--candidate-net", str(net),
            "--out-dir", str(gate_dir),
            "--nodes", str(args.nodes),
            "--threads", str(args.threads),
            "--hash", str(args.hash),
            "--limit", str(args.limit),
            "--cap", str(args.cap),
        ]
        if args.require_native_net_load:
            command.append("--require-native-net-load")
        if reference_csv is not None:
            command.extend(["--reference-csv", str(reference_csv)])
        else:
            assert reference_net is not None
            command.extend([
                "--reference-engine", str(reference_engine),
                "--reference-net", str(reference_net),
            ])
        emit_event(
            run_dir, "phase_start", stage=stage, status="running",
            command=command, hook_command=args.event_command or "",
            extra={"checkpoint": idx, "net": str(net)},
        )
        gate_rc = run(command, log_path=out_dir / f"checkpoint_{idx}.log")
        rc = rc or gate_rc
        rows.extend(read_summary(gate_dir / "summary.txt", idx))
        summary_path.write_text("\n".join(rows).rstrip() + "\n", encoding="utf-8")
        emit_event(
            run_dir, "phase_done" if gate_rc == 0 else "fail",
            stage=stage, status="ok" if gate_rc == 0 else "failed",
            rc=gate_rc, command=command, hook_command=args.event_command or "",
            extra={"checkpoint": idx, "net": str(net), "summary": str(summary_path)},
        )
        if gate_rc != 0:
            break

    emit_event(
        run_dir, "phase_done" if rc == 0 else "fail",
        stage="checkpoint_sweep", status="ok" if rc == 0 else "failed",
        rc=rc, hook_command=args.event_command or "",
        extra={"summary": str(summary_path)},
    )
    print(summary_path.read_text(encoding="utf-8"), end="")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
