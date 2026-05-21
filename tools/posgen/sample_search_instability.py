#!/usr/bin/env python3
"""Sample positions where a UCI teacher changes opinion between depths."""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path


SCORE_RE = re.compile(r"\bscore\s+(cp|mate)\s+(-?\d+)\b")


class EngineTimeout(RuntimeError):
    pass


class UciEngine:
    def __init__(
            self, path: str, *, threads: int, hash_mb: int,
            position_timeout_s: float) -> None:
        self.position_timeout_s = position_timeout_s
        self.proc = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.send("uci")
        self.wait_for("uciok")
        self.setoption("Threads", str(threads))
        self.setoption("Hash", str(hash_mb))
        self.send("isready")
        self.wait_for("readyok")

    def close(self) -> None:
        try:
            self.send("quit")
        finally:
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

    def send(self, command: str) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("engine stdin closed")
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def readline(self) -> str:
        if self.proc.stdout is None:
            raise RuntimeError("engine stdout closed")
        line = self.proc.stdout.readline()
        if line == "":
            raise RuntimeError("engine exited")
        return line.strip()

    def wait_for(self, token: str) -> None:
        while True:
            if self.readline() == token:
                return

    def setoption(self, name: str, value: str) -> None:
        self.send(f"setoption name {name} value {value}")

    def analyze(self, fen: str, *, depth: int) -> tuple[int | None, str | None, str | None]:
        self.send(f"position fen {fen}")
        self.send(f"go depth {depth}")
        score_cp: int | None = None
        mate: str | None = None
        timed_out = threading.Event()
        done = threading.Event()

        def watchdog() -> None:
            if not done.wait(self.position_timeout_s):
                timed_out.set()
                try:
                    self.send("stop")
                except Exception:
                    pass

        watchdog_thread: threading.Thread | None = None
        if self.position_timeout_s > 0:
            watchdog_thread = threading.Thread(target=watchdog, daemon=True)
            watchdog_thread.start()
        try:
            while True:
                line = self.readline()
                match = SCORE_RE.search(line)
                if match:
                    kind, value = match.groups()
                    if kind == "cp":
                        score_cp = int(value)
                        mate = None
                    else:
                        mate = value
                if line.startswith("bestmove "):
                    if timed_out.is_set():
                        raise EngineTimeout(
                            f"go depth {depth} timed out after "
                            f"{self.position_timeout_s:.1f}s")
                    parts = line.split()
                    bestmove = parts[1] if len(parts) > 1 else None
                    return score_cp, mate, bestmove
        finally:
            done.set()
            if watchdog_thread is not None:
                watchdog_thread.join(timeout=0.1)


def maybe_take(reservoir: list[str], line: str, accepted: int, rows: int,
               rng: random.Random) -> None:
    if rows <= 0:
        reservoir.append(line)
        return
    if len(reservoir) < rows:
        reservoir.append(line)
        return
    idx = rng.randrange(accepted)
    if idx < rows:
        reservoir[idx] = line


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--engine", default="stockfish")
    ap.add_argument("--low-depth", type=int, default=8)
    ap.add_argument("--high-depth", type=int, default=16)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash", type=int, default=128)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--max-abs-cp", type=int, default=1600)
    ap.add_argument("--min-delta-cp", type=int, default=80)
    ap.add_argument("--bestmove-change", action="store_true")
    ap.add_argument("--max-output", type=int, default=0,
                    help="Reservoir-sample at most this many accepted rows.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress", type=int, default=1000)
    ap.add_argument("--position-timeout-s", type=float, default=0,
                    help="Skip and restart the engine if one depth search "
                         "takes longer than this. 0 disables the timeout.")
    args = ap.parse_args()

    if not (0 <= args.shard_index < args.shard_count):
        raise ValueError("invalid shard index/count")
    if args.high_depth <= args.low_depth:
        raise ValueError("--high-depth must be greater than --low-depth")

    rng = random.Random(args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "input": args.input,
        "output": args.output,
        "engine": args.engine,
        "low_depth": args.low_depth,
        "high_depth": args.high_depth,
        "threads": args.threads,
        "hash": args.hash,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "max_abs_cp": args.max_abs_cp,
        "min_delta_cp": args.min_delta_cp,
        "bestmove_change": args.bestmove_change,
        "max_output": args.max_output,
        "read": 0,
        "selected": 0,
        "accepted": 0,
        "written": 0,
        "skipped_mate": 0,
        "skipped_no_score": 0,
        "skipped_cp": 0,
        "skipped_timeout": 0,
    }

    reservoir: list[str] = []
    engine = UciEngine(
        args.engine, threads=args.threads, hash_mb=args.hash,
        position_timeout_s=args.position_timeout_s)
    start = time.monotonic()
    try:
        with Path(args.input).open(encoding="utf-8", errors="replace") as src:
            for idx, line in enumerate(src):
                stats["read"] += 1
                if args.limit > 0 and stats["selected"] >= args.limit:
                    break
                if idx % args.shard_count != args.shard_index:
                    continue
                line = line.strip()
                if not line:
                    continue
                stats["selected"] += 1

                row = json.loads(line)
                fen = row.get("fen")
                if not isinstance(fen, str):
                    continue

                try:
                    low_score, low_mate, low_best = engine.analyze(
                        fen, depth=args.low_depth)
                    high_score, high_mate, high_best = engine.analyze(
                        fen, depth=args.high_depth)
                except EngineTimeout as exc:
                    stats["skipped_timeout"] += 1
                    print(
                        f"shard {args.shard_index}/{args.shard_count}: "
                        f"timeout selected={stats['selected']} "
                        f"skipped_timeout={stats['skipped_timeout']} {exc}",
                        flush=True,
                    )
                    engine.close()
                    engine = UciEngine(
                        args.engine, threads=args.threads, hash_mb=args.hash,
                        position_timeout_s=args.position_timeout_s)
                    continue
                if low_mate is not None or high_mate is not None:
                    stats["skipped_mate"] += 1
                    continue
                if low_score is None or high_score is None:
                    stats["skipped_no_score"] += 1
                    continue
                if args.max_abs_cp > 0 and abs(high_score) > args.max_abs_cp:
                    stats["skipped_cp"] += 1
                    continue

                delta = abs(high_score - low_score)
                best_changed = (
                    low_best is not None and high_best is not None
                    and low_best != high_best)
                if delta < args.min_delta_cp and not (
                        args.bestmove_change and best_changed):
                    continue

                row["source_score"] = row.get("score")
                row["score"] = high_score
                row["teacher"] = Path(args.engine).name
                row["teacher_depth"] = args.high_depth
                row["instability_low_depth"] = args.low_depth
                row["instability_low_score"] = low_score
                row["instability_high_score"] = high_score
                row["instability_delta_cp"] = delta
                row["instability_low_bestmove"] = low_best
                row["instability_high_bestmove"] = high_best
                row["instability_bestmove_changed"] = best_changed

                stats["accepted"] += 1
                encoded = json.dumps(row, separators=(",", ":"))
                maybe_take(
                    reservoir, encoded, stats["accepted"],
                    args.max_output, rng)

                if args.progress > 0 and stats["selected"] % args.progress == 0:
                    elapsed = max(1e-6, time.monotonic() - start)
                    rate = stats["selected"] / elapsed
                    print(
                        f"shard {args.shard_index}/{args.shard_count}: "
                        f"selected={stats['selected']} accepted={stats['accepted']} "
                        f"rate={rate:.1f}/s",
                        flush=True,
                    )
    finally:
        engine.close()

    rng.shuffle(reservoir)
    with out.open("w", encoding="utf-8") as dst:
        for line in reservoir:
            dst.write(line)
            dst.write("\n")
    stats["written"] = len(reservoir)
    stats["elapsed_s"] = round(time.monotonic() - start, 3)
    out.with_suffix(out.suffix + ".stats.json").write_text(
        json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
