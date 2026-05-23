#!/usr/bin/env python3
"""Sweep bucketed-head masks through the broad search-target gate."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.lib.events import emit_event


def run(command: list[str], log_path: Path) -> int:
    print(" ".join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
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


def parse_bucket_sets(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def label_for(bucket_set: str) -> str:
    return bucket_set.replace(",", "_").replace("-", "to")


def read_summary(path: Path, label: str) -> list[str]:
    if not path.exists():
        return [f"{label}\tmissing\t\t\t\t\t\t\t"]

    sections: dict[str, dict[str, str]] = {}
    current = ""
    wanted = {
        "top1", "top3", "candidate_better", "reference_better",
        "capped_sum_diff_cp", "median_nonzero_diff_cp",
        "worst_regression_cp", "best_gain_cp",
    }
    for line in path.read_text(encoding="utf-8").splitlines():
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
        compare = sections.get(split, {})
        candidate = sections.get(split.removeprefix("compare:"), {})
        out.append("\t".join([
            label,
            split,
            candidate.get("top1", ""),
            candidate.get("top3", ""),
            compare.get("candidate_better", ""),
            compare.get("reference_better", ""),
            compare.get("capped_sum_diff_cp", ""),
            compare.get("median_nonzero_diff_cp", ""),
            compare.get("worst_regression_cp", ""),
            compare.get("best_gain_cp", ""),
        ]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--bucket-sets", required=True, type=parse_bucket_sets)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--reference-net", required=True, type=Path)
    parser.add_argument("--reference-engine", type=Path)
    parser.add_argument("--nodes", type=int, default=300000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=128)
    parser.add_argument("--cap", type=int, default=200)
    parser.add_argument("--event-command", default="")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    nets_dir = run_dir / "nets"
    diff_dir = run_dir / "net_diff"
    search_dir = run_dir / "search"
    summary_path = run_dir / "summary.tsv"
    mix_tool = Path(__file__).with_name("mix_material_head.py")
    net_diff = Path(__file__).resolve().parents[1] / "validate" / "net_diff.py"
    search_gate = Path(__file__).resolve().parents[1] / "validate" / "search_target_gate.py"

    emit_event(
        run_dir,
        "phase_start",
        stage="bucket_mix_sweep",
        status="running",
        hook_command=args.event_command,
        extra={"bucket_sets": args.bucket_sets},
    )

    rows = [
        "bucket_set\tsplit\ttop1\ttop3\tcandidate_better\treference_better\t"
        "capped_sum_diff_cp\tmedian_nonzero_diff_cp\tworst_regression_cp\tbest_gain_cp"
    ]
    rc = 0
    for bucket_set in args.bucket_sets:
        label = label_for(bucket_set)
        stage = f"bucket_{label}"
        net = nets_dir / f"bucket_{label}.nn"
        emit_event(
            run_dir,
            "phase_start",
            stage=stage,
            status="running",
            hook_command=args.event_command,
            extra={"bucket_set": bucket_set, "net": str(net)},
        )

        make_rc = run([
            sys.executable, str(mix_tool),
            "--base", str(args.base.expanduser().resolve()),
            "--candidate", str(args.candidate.expanduser().resolve()),
            "--output", str(net),
            "--candidate-buckets", bucket_set,
        ], diff_dir / f"{label}_make.log")
        if make_rc != 0:
            rc = make_rc
            emit_event(
                run_dir, "fail", stage=stage, status="failed", rc=make_rc,
                log=diff_dir / f"{label}_make.log",
                hook_command=args.event_command,
            )
            break

        diff_rc = run([
            sys.executable, str(net_diff),
            "--candidate", str(net),
            "--reference", str(args.base.expanduser().resolve()),
            "--fail-if-identical",
        ], diff_dir / f"{label}.log")
        if diff_rc != 0:
            rc = diff_rc
            emit_event(
                run_dir, "fail", stage=stage, status="failed", rc=diff_rc,
                log=diff_dir / f"{label}.log",
                hook_command=args.event_command,
            )
            break

        gate_out = search_dir / label
        reference_engine = args.reference_engine or args.engine
        gate_rc = run([
            sys.executable, str(search_gate),
            "--targets", str(args.targets.expanduser().resolve()),
            "--engine", str(args.engine.expanduser().resolve()),
            "--candidate-net", str(net),
            "--reference-engine", str(reference_engine.expanduser().resolve()),
            "--reference-net", str(args.reference_net.expanduser().resolve()),
            "--out-dir", str(gate_out),
            "--nodes", str(args.nodes),
            "--threads", str(args.threads),
            "--hash", str(args.hash),
            "--cap", str(args.cap),
        ], gate_out / "run.log")

        rows.extend(read_summary(gate_out / "summary.txt", label))
        summary_path.write_text("\n".join(rows).rstrip() + "\n", encoding="utf-8")
        emit_event(
            run_dir,
            "phase_done" if gate_rc == 0 else "fail",
            stage=stage,
            status="ok" if gate_rc == 0 else "failed",
            rc=gate_rc,
            log=summary_path,
            hook_command=args.event_command,
            extra={"bucket_set": bucket_set, "summary": str(summary_path)},
        )
        if gate_rc != 0:
            rc = gate_rc
            break

    emit_event(
        run_dir,
        "done" if rc == 0 else "fail",
        stage="bucket_mix_sweep",
        status="ok" if rc == 0 else "failed",
        rc=rc,
        log=summary_path,
        hook_command=args.event_command,
    )
    if summary_path.exists():
        print(summary_path.read_text(encoding="utf-8"), end="")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
