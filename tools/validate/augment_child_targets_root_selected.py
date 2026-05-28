#!/usr/bin/env python3
"""Add root-search-selected moves to child-ranking target groups."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import queue
import re
import shlex
import subprocess
import threading
from collections import Counter
from pathlib import Path

import chess


SCORE_RE = re.compile(r"\bscore\s+(cp|mate)\s+(-?\d+)\b")


def mate_to_cp(mate: int) -> int:
    sign = 1 if mate > 0 else -1
    return sign * (32000 - min(abs(mate), 1000))


def expand(value: str) -> str:
    return str(Path(os.path.expandvars(value)).expanduser())


def parse_search_net(raw: str) -> tuple[str, str]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        return label.strip(), expand(path.strip())
    path = expand(raw.strip())
    return Path(path).stem, path


class UciEngine:
    def __init__(
        self,
        command: str,
        *,
        threads: int,
        hash_mb: int,
        net: str = "",
    ) -> None:
        self.proc = subprocess.Popen(
            shlex.split(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("failed to open engine pipes")
        self.stdin = self.proc.stdin
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.reader = threading.Thread(target=self._read_stdout, daemon=True)
        self.reader.start()
        self.send("uci")
        self.wait_for("uciok", 10.0)
        self.setoption("Threads", str(threads))
        self.setoption("Hash", str(hash_mb))
        if net:
            self.setoption("nnue_file", net)
        self.send("isready")
        self.wait_for("readyok", 20.0)

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.send("quit")
            except Exception:
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines.put(line.rstrip("\n"))
        self.lines.put(None)

    def send(self, command: str) -> None:
        self.stdin.write(command + "\n")
        self.stdin.flush()

    def read_line(self, timeout: float) -> str:
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty as exc:
            raise RuntimeError("engine read timed out") from exc
        if line is None:
            raise RuntimeError("engine exited")
        return line

    def wait_for(self, token: str, timeout: float) -> None:
        while True:
            if token in self.read_line(timeout):
                return

    def setoption(self, name: str, value: str) -> None:
        self.send(f"setoption name {name} value {value}")

    def newgame(self) -> None:
        self.send("ucinewgame")
        self.send("isready")
        self.wait_for("readyok", 20.0)

    def bestmove(self, fen: str, *, nodes: int, depth: int) -> str | None:
        self.newgame()
        self.send(f"position fen {fen}")
        if nodes > 0:
            self.send(f"go nodes {nodes}")
        else:
            self.send(f"go depth {depth}")
        while True:
            line = self.read_line(120.0)
            if line.startswith("bestmove "):
                move = line.split()[1]
                return None if move == "(none)" else move

    def score_stm_cp(self, fen: str, *, nodes: int, depth: int) -> int | None:
        self.newgame()
        self.send(f"position fen {fen}")
        if nodes > 0:
            self.send(f"go nodes {nodes}")
        else:
            self.send(f"go depth {depth}")
        score: int | None = None
        while True:
            line = self.read_line(120.0)
            match = SCORE_RE.search(line)
            if match:
                kind, value = match.groups()
                raw = int(value)
                score = raw if kind == "cp" else mate_to_cp(raw)
            if line.startswith("bestmove "):
                return score


def child_fen(board: chess.Board, move_uci: str) -> str:
    child = board.copy(stack=False)
    child.push(chess.Move.from_uci(move_uci))
    return child.fen()


def as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def add_origin(move: dict, origin: str) -> None:
    origins = as_list(move.get("origins", move.get("role", [])))
    if origin not in origins:
        origins.append(origin)
    move["origins"] = origins


def normalize_best(row: dict) -> tuple[bool, bool]:
    raw_best = str(row.get("best_move", ""))
    moves = row["moves"]
    best = max(moves, key=lambda item: float(item["score_cp"]))
    best_move = str(best["move"])
    raw_missing = bool(raw_best and raw_best not in {str(m["move"]) for m in moves})
    overridden = bool(raw_best and raw_best != best_move)
    if raw_best and "replay_best_move" not in row:
        row["replay_best_move"] = raw_best
    row["best_move"] = best_move
    for move in moves:
        if str(move["move"]) == best_move:
            move["role"] = "best"
        elif move.get("role") == "best":
            move["role"] = "neighbor"
    return overridden, raw_missing


def augment_row(
    row: dict,
    search_engines: list[tuple[str, UciEngine]],
    oracle: UciEngine,
    args: argparse.Namespace,
) -> tuple[dict, Counter[str]]:
    counters: Counter[str] = Counter()
    board = chess.Board(str(row["fen"]))
    moves = list(row.get("moves", []))
    row["moves"] = moves
    by_move = {str(move.get("move", "")): move for move in moves}
    missing: dict[str, set[str]] = {}

    for label, engine in search_engines:
        selected = engine.bestmove(row["fen"], nodes=args.search_nodes,
                                   depth=args.search_depth)
        counters[f"selected:{label}"] += 1
        if not selected:
            counters[f"no_bestmove:{label}"] += 1
            continue
        try:
            move = chess.Move.from_uci(selected)
        except ValueError:
            counters[f"illegal_selected:{label}"] += 1
            continue
        if move not in board.legal_moves:
            counters[f"illegal_selected:{label}"] += 1
            continue
        origin = f"root_search:{label}"
        if selected in by_move:
            add_origin(by_move[selected], origin)
        else:
            counters[f"missing_before:{label}"] += 1
            missing.setdefault(selected, set()).add(origin)

    for move_uci, origins in missing.items():
        stm_score = oracle.score_stm_cp(
            child_fen(board, move_uci),
            nodes=args.oracle_nodes,
            depth=args.oracle_depth,
        )
        if stm_score is None:
            counters["score_failed"] += 1
            continue
        item = {
            "move": move_uci,
            "score_cp": int(-stm_score),
            "role": "neighbor",
            "rank": None,
            "origins": sorted([*origins, "augmented"]),
        }
        moves.append(item)
        by_move[move_uci] = item
        counters["added_moves"] += 1

    if missing:
        counters["groups_augmented"] += 1
    overridden, raw_missing = normalize_best(row)
    counters["best_overrides"] += int(overridden)
    counters["raw_best_missing"] += int(raw_missing)
    counters["missing_after"] += counters["score_failed"]
    return row, counters


def chunks(items: list[tuple[int, dict]], jobs: int) -> list[list[tuple[int, dict]]]:
    jobs = max(1, min(jobs, len(items)))
    size = (len(items) + jobs - 1) // jobs
    return [items[i:i + size] for i in range(0, len(items), size)]


def process_chunk(
    chunk: list[tuple[int, dict]],
    search_nets: list[tuple[str, str]],
    args: argparse.Namespace,
) -> tuple[list[tuple[int, dict]], Counter[str]]:
    search_engines = [
        (label, UciEngine(args.engine, threads=args.threads,
                          hash_mb=args.hash, net=net))
        for label, net in search_nets
    ]
    oracle = UciEngine(args.oracle_engine, threads=args.oracle_threads,
                       hash_mb=args.oracle_hash)
    out: list[tuple[int, dict]] = []
    counters: Counter[str] = Counter()
    try:
        for idx, row in chunk:
            new_row, row_counts = augment_row(row, search_engines, oracle, args)
            out.append((idx, new_row))
            counters.update(row_counts)
    finally:
        for _label, engine in search_engines:
            engine.close()
        oracle.close()
    return out, counters


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Augment child targets with root-search-selected moves.")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--search-net", action="append", required=True,
                    help="label=/path/to/net.nn; repeatable")
    ap.add_argument("--search-nodes", type=int, default=20000)
    ap.add_argument("--search-depth", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--oracle-engine", required=True)
    ap.add_argument("--oracle-nodes", type=int, default=200000)
    ap.add_argument("--oracle-depth", type=int, default=0)
    ap.add_argument("--oracle-threads", type=int, default=1)
    ap.add_argument("--oracle-hash", type=int, default=128)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--min-groups", type=int, default=1)
    ap.add_argument("--fail-if-missing-after", type=int, default=0)
    args = ap.parse_args()

    if args.search_nodes <= 0 and args.search_depth <= 0:
        raise SystemExit("either --search-nodes or --search-depth must be positive")
    if args.oracle_nodes <= 0 and args.oracle_depth <= 0:
        raise SystemExit("either --oracle-nodes or --oracle-depth must be positive")

    search_nets = [parse_search_net(raw) for raw in args.search_net]
    rows: list[tuple[int, dict]] = []
    with args.input.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if line.strip():
                rows.append((idx, json.loads(line)))

    if len(rows) < args.min_groups:
        raise SystemExit(f"groups={len(rows)} below min_groups={args.min_groups}")

    counters: Counter[str] = Counter(rows=len(rows))
    results: list[tuple[int, dict]] = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, args.jobs)) as executor:
        futures = [
            executor.submit(process_chunk, chunk, search_nets, args)
            for chunk in chunks(rows, max(1, args.jobs))
        ]
        for future in concurrent.futures.as_completed(futures):
            out, chunk_counts = future.result()
            results.extend(out)
            counters.update(chunk_counts)

    results.sort(key=lambda item: item[0])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for _idx, row in results:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    counters["written_groups"] = len(results)

    lines = [
        f"{key}={counters[key]}"
        for key in sorted(counters)
    ]
    summary = "\n".join(lines) + f"\noutput={args.out}\n"
    print(summary, end="")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(summary, encoding="utf-8")

    if counters["missing_after"] > args.fail_if_missing_after:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
