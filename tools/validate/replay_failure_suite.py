#!/usr/bin/env python3
"""Run replay candidate/reference CSV and summarize failure-suite deltas."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def parse_cp(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    if value.startswith("+M"):
        return 32000
    if value.startswith("-M"):
        return -32000
    if value.endswith("cp"):
        value = value[:-2]
    try:
        return int(value)
    except ValueError:
        return None


def int_field(row: dict[str, str], key: str) -> int:
    try:
        return int(row.get(key, "0"))
    except ValueError:
        return 0


def replay_command(args: argparse.Namespace, logs: list[str]) -> list[str]:
    cmd = [
        args.replay,
        "--candidate", args.candidate,
        "--reference", args.reference,
        "--oracle", args.oracle,
        "--oracle-nodes", str(args.oracle_nodes),
        "--threads", str(args.threads),
        "--jobs", str(args.jobs),
        "--csv",
    ]
    if args.fixed_nodes > 0:
        cmd.extend(["--fixed-nodes", str(args.fixed_nodes)])
    if args.max_replay_nodes > 0:
        cmd.extend(["--max-replay-nodes", str(args.max_replay_nodes)])
    if args.count > 0:
        cmd.extend(["--count", str(args.count)])
    cmd.extend(logs)
    return cmd


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd, check=False, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    return proc


def csv_without_header(csv_text: str) -> str:
    lines = csv_text.splitlines()
    if not lines:
        return ""
    return "\n".join(lines[1:])


def run_replay(args: argparse.Namespace, logs: list[str]) -> str:
    cmd = replay_command(args, logs)
    proc = run_command(cmd)
    if args.stderr:
        args.stderr.parent.mkdir(parents=True, exist_ok=True)
        args.stderr.write_text(proc.stderr)
    if proc.returncode == 0:
        return proc.stdout
    if len(logs) <= 1:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)

    # A single stale/empty bot log should not invalidate a whole failure-suite
    # gate. Fall back to per-log replay so invalid logs can be skipped while
    # real replay/engine failures still stop the run.
    outputs: list[str] = []
    stderr_parts = [proc.stderr]
    skipped: list[str] = []
    for log in logs:
        one = run_command(replay_command(args, [log]))
        stderr_parts.append(one.stderr)
        if one.returncode == 0:
            if not outputs:
                outputs.append(one.stdout.rstrip("\n"))
            else:
                body = csv_without_header(one.stdout).strip()
                if body:
                    outputs.append(body)
            continue
        if "No UCI go/bestmove pairs found" in one.stderr:
            skipped.append(log)
            continue
        print(one.stderr, file=sys.stderr)
        raise SystemExit(one.returncode)

    if args.stderr:
        args.stderr.write_text("\n".join(stderr_parts))
    if skipped:
        print(
            "skipped invalid logs: " + ", ".join(skipped),
            file=sys.stderr,
            flush=True,
        )
    if not outputs:
        print("no replay rows after skipping invalid logs", file=sys.stderr)
        raise SystemExit(1)
    return "\n".join(outputs) + "\n"


def summarize(csv_text: str) -> tuple[list[dict[str, str]], dict[str, object]]:
    rows = list(csv.DictReader(csv_text.splitlines()))
    diffs = [int_field(row, "diff") for row in rows]
    nonzero = [diff for diff in diffs if diff != 0]
    candidate_better = sum(1 for diff in diffs if diff > 0)
    reference_better = sum(1 for diff in diffs if diff < 0)

    by_log: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "positions": 0,
            "candidate_better": 0,
            "reference_better": 0,
            "sum_diff_cp": 0,
            "worst_regression_cp": 0,
            "best_gain_cp": 0,
        })
    for row in rows:
        log = row.get("log", "")
        diff = int_field(row, "diff")
        item = by_log[log]
        item["positions"] = int(item["positions"]) + 1
        item["sum_diff_cp"] = int(item["sum_diff_cp"]) + diff
        item["worst_regression_cp"] = min(int(item["worst_regression_cp"]), diff)
        item["best_gain_cp"] = max(int(item["best_gain_cp"]), diff)
        if diff > 0:
            item["candidate_better"] = int(item["candidate_better"]) + 1
        elif diff < 0:
            item["reference_better"] = int(item["reference_better"]) + 1

    summary: dict[str, object] = {
        "positions": len(rows),
        "candidate_better": candidate_better,
        "reference_better": reference_better,
        "sum_diff_cp": sum(diffs),
        "median_nonzero_diff_cp": (
            statistics.median(nonzero) if nonzero else 0),
        "worst_regression_cp": min(diffs) if diffs else 0,
        "best_gain_cp": max(diffs) if diffs else 0,
        "logs": by_log,
    }
    return rows, summary


def write_outputs(
        rows: list[dict[str, str]], summary: dict[str, object],
        output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "replay_failure_suite.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    with (output_dir / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"positions={summary['positions']}\n")
        handle.write(f"candidate_better={summary['candidate_better']}\n")
        handle.write(f"reference_better={summary['reference_better']}\n")
        handle.write(f"sum_diff_cp={summary['sum_diff_cp']}\n")
        handle.write(
            f"median_nonzero_diff_cp={summary['median_nonzero_diff_cp']}\n")
        handle.write(f"worst_regression_cp={summary['worst_regression_cp']}\n")
        handle.write(f"best_gain_cp={summary['best_gain_cp']}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--oracle", default="stockfish")
    ap.add_argument("--replay", default="replay")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--fixed-nodes", type=int, default=100000)
    ap.add_argument("--max-replay-nodes", type=int, default=0)
    ap.add_argument("--oracle-nodes", type=int, default=200000)
    ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--stderr", type=Path)
    args = ap.parse_args()

    csv_text = run_replay(args, args.logs)
    rows, summary = summarize(csv_text)
    write_outputs(rows, summary, args.output_dir)
    print(f"positions={summary['positions']}")
    print(f"candidate_better={summary['candidate_better']}")
    print(f"reference_better={summary['reference_better']}")
    print(f"sum_diff_cp={summary['sum_diff_cp']}")
    print(f"median_nonzero_diff_cp={summary['median_nonzero_diff_cp']}")
    print(f"worst_regression_cp={summary['worst_regression_cp']}")
    print(f"best_gain_cp={summary['best_gain_cp']}")


if __name__ == "__main__":
    main()
