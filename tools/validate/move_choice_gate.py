#!/usr/bin/env python3
"""Evaluate an engine on scored repeated-tail target positions."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def send(proc: subprocess.Popen[str], command: str) -> None:
    assert proc.stdin is not None
    proc.stdin.write(command + "\n")
    proc.stdin.flush()


def read_until(proc: subprocess.Popen[str], prefix: str) -> str:
    assert proc.stdout is not None
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(f"engine exited before {prefix!r}")
        line = line.strip()
        if line.startswith(prefix):
            return line


def bestmove(
        proc: subprocess.Popen[str], fen: str, nodes: int) -> tuple[str, int]:
    send(proc, f"position fen {fen}")
    send(proc, f"go nodes {nodes}")
    best = ""
    nodes_seen = 0
    assert proc.stdout is not None
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("engine exited before bestmove")
        line = line.strip()
        if line.startswith("info "):
            parts = line.split()
            if "nodes" in parts:
                try:
                    nodes_seen = int(parts[parts.index("nodes") + 1])
                except (ValueError, IndexError):
                    pass
        if line.startswith("bestmove "):
            parts = line.split()
            best = parts[1] if len(parts) > 1 else ""
            return best, nodes_seen


def load_targets(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    by_target: dict[tuple[str, str], dict[str, object]] = {}
    for row in csv.DictReader(path.open(newline="", encoding="utf-8")):
        key = (row["log"], row["ply"])
        item = by_target.setdefault(key, {
            "log": row["log"],
            "ply": row["ply"],
            "fullmove": row["fullmove"],
            "side": row["side"],
            "fen": row["fen"],
            "moves": {},
            "best_move": row["move"],
        })
        item["moves"][row["move"]] = {
            "rank": int(row["rank"]),
            "score_cp": int(row["score_cp"]),
            "gap_cp": int(row["gap_cp"]),
        }
        if int(row["rank"]) == 1:
            item["best_move"] = row["move"]
    return by_target


def configure_engine(
        proc: subprocess.Popen[str], threads: int, hash_mb: int) -> None:
    send(proc, "uci")
    read_until(proc, "uciok")
    send(proc, f"setoption name Threads value {threads}")
    send(proc, f"setoption name Hash value {hash_mb}")
    send(proc, "isready")
    read_until(proc, "readyok")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an engine over scored move-choice targets.")
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--nodes", type=int, default=100000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=128)
    args = parser.parse_args()

    targets = load_targets(args.scores.expanduser())
    proc = subprocess.Popen(
        [args.engine],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    rows: list[dict[str, object]] = []
    try:
        configure_engine(proc, args.threads, args.hash)
        for target in targets.values():
            move, nodes_seen = bestmove(proc, str(target["fen"]), args.nodes)
            move_info = target["moves"].get(move)
            if move_info:
                rank = move_info["rank"]
                gap = move_info["gap_cp"]
                score = move_info["score_cp"]
            else:
                rank = 999
                gap = 32000
                score = 0
            rows.append({
                "log": target["log"],
                "ply": target["ply"],
                "fullmove": target["fullmove"],
                "side": target["side"],
                "fen": target["fen"],
                "best_move": target["best_move"],
                "engine_move": move,
                "rank": rank,
                "gap_cp": gap,
                "score_cp": score,
                "nodes": nodes_seen,
            })
    finally:
        try:
            send(proc, "quit")
        except Exception:
            pass
        proc.wait(timeout=5)

    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "log",
        "ply",
        "fullmove",
        "side",
        "fen",
        "best_move",
        "engine_move",
        "rank",
        "gap_cp",
        "score_cp",
        "nodes",
    ]
    with args.out.expanduser().open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    top1 = sum(1 for row in rows if int(row["rank"]) == 1)
    top3 = sum(1 for row in rows if int(row["rank"]) <= 3)
    gaps = [int(row["gap_cp"]) for row in rows]
    print(f"targets={total}")
    print(f"top1={top1}/{total}")
    print(f"top3={top3}/{total}")
    print(f"sum_gap_cp={sum(gaps)}")
    print(f"worst_gap_cp={max(gaps) if gaps else 0}")
    print(f"wrote {args.out.expanduser()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
