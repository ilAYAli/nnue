#!/usr/bin/env python3
"""Score LC0 candidate moves with a UCI oracle for child-ranking targets."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import shlex
import subprocess
import time
from collections import Counter
from pathlib import Path

import chess


SCORE_RE = re.compile(r"\bscore\s+(cp|mate)\s+(-?\d+)\b")


def mate_to_cp(mate: int) -> int:
    sign = 1 if mate > 0 else -1
    return sign * (32000 - min(abs(mate), 1000))


class UciEngine:
    def __init__(self, command: str, *, threads: int, hash_mb: int) -> None:
        self.proc = subprocess.Popen(
            shlex.split(command),
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

    def score_stm_cp(self, fen: str, *, nodes: int, depth: int) -> int | None:
        self.send(f"position fen {fen}")
        if nodes > 0:
            self.send(f"go nodes {nodes}")
        else:
            self.send(f"go depth {depth}")
        score: int | None = None
        while True:
            line = self.readline()
            match = SCORE_RE.search(line)
            if match:
                kind, value = match.groups()
                raw = int(value)
                score = raw if kind == "cp" else mate_to_cp(raw)
            if line.startswith("bestmove "):
                return score


def phase_tag(board: chess.Board) -> str:
    nonking = max(0, len(board.piece_map()) - 2)
    if nonking <= 8:
        return "phase:endgame"
    if nonking <= 20:
        return "phase:late"
    if nonking <= 28:
        return "phase:midgame"
    return "phase:opening"


def role_list(raw: dict) -> list[str]:
    roles = raw.get("roles", raw.get("role", []))
    if isinstance(roles, str):
        return [roles]
    return [str(role) for role in roles]


def policy_gap_cp(best_policy: float, other_policy: float, *,
                  floor: float, scale_cp: float, max_gap_cp: float) -> float:
    best = max(best_policy, floor)
    other = max(other_policy, floor)
    return min(max_gap_cp, max(0.0, scale_cp * (math.log(best) - math.log(other))))


def legal_policy_moves(row: dict, board: chess.Board, max_moves: int) -> list[dict]:
    moves: list[dict] = []
    seen: set[str] = set()
    for raw in row.get("moves", []):
        move_uci = str(raw.get("move", ""))
        if not move_uci or move_uci in seen or raw.get("legal") is not True:
            continue
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            continue
        if move not in board.legal_moves:
            continue
        seen.add(move_uci)
        moves.append(raw)
    moves.sort(key=lambda item: (
        int(item.get("rank", 999999)),
        -float(item.get("policy", 0.0)),
        str(item.get("move", "")),
    ))
    return moves[:max_moves]


def passes_policy_filters(row: dict, moves: list[dict],
                          args: argparse.Namespace,
                          counters: Counter[str]) -> bool:
    if len(moves) < 2:
        counters["skipped_few_moves"] += 1
        return False
    policies = sorted((float(move.get("policy", 0.0)) for move in moves), reverse=True)
    best_policy = policies[0]
    if best_policy < args.min_best_policy:
        counters["skipped_low_best_policy"] += 1
        return False
    gap = policy_gap_cp(
        best_policy,
        policies[1],
        floor=args.policy_floor,
        scale_cp=args.policy_score_scale_cp,
        max_gap_cp=args.policy_gap_cap_cp,
    )
    if gap < args.min_policy_gap_cp:
        counters["skipped_low_policy_gap"] += 1
        return False
    return True


def child_fen(board: chess.Board, move_uci: str) -> str:
    child = board.copy(stack=False)
    child.push(chess.Move.from_uci(move_uci))
    return child.fen()


def convert_row(row: dict, engine: UciEngine, args: argparse.Namespace,
                counters: Counter[str]) -> dict | None:
    if row.get("schema") != "enyo.lc0.v1":
        counters["skipped_schema"] += 1
        return None
    board = chess.Board(str(row["fen"]))
    moves = legal_policy_moves(row, board, args.max_moves_per_position)
    if not passes_policy_filters(row, moves, args, counters):
        return None

    scored_moves: list[dict] = []
    for raw in moves:
        move_uci = str(raw["move"])
        stm_score = engine.score_stm_cp(
            child_fen(board, move_uci),
            nodes=args.nodes,
            depth=args.depth,
        )
        if stm_score is None:
            counters["skipped_no_oracle_score"] += 1
            return None
        parent_score = -stm_score
        scored_moves.append({
            "move": move_uci,
            "score_cp": int(parent_score),
            "role": "neighbor",
            "rank": raw.get("rank"),
            "policy": float(raw.get("policy", 0.0)),
            "origins": role_list(raw),
        })

    best = max(scored_moves, key=lambda item: (item["score_cp"], item["policy"]))
    second = max(
        (move["score_cp"] for move in scored_moves if move["move"] != best["move"]),
        default=best["score_cp"],
    )
    oracle_gap = float(best["score_cp"] - second)
    if oracle_gap < args.min_oracle_gap_cp:
        counters["skipped_low_oracle_gap"] += 1
        return None
    for move in scored_moves:
        if move["move"] == best["move"]:
            move["role"] = "best"
            break

    tags = list(dict.fromkeys([
        *[str(tag) for tag in row.get("tags", [])],
        "source:lc0_oracle",
        "label:oracle_child_cp",
        phase_tag(board),
    ]))
    for tag in tags:
        counters[f"tag:{tag}"] += 1

    return {
        "id": str(row.get("id") or f"lc0-oracle:{counters['rows']}"),
        "fen": board.fen(),
        "best_move": str(best["move"]),
        "max_gap_cp": float(args.max_gap_cp),
        "moves": scored_moves,
        "source": "lc0_oracle",
        "source_file": row.get("source_file"),
        "record_index": row.get("record_index"),
        "license": row.get("license"),
        "score_pov": "parent",
        "oracle": {
            "engine": args.engine,
            "nodes": args.nodes,
            "depth": args.depth,
            "threads": args.threads,
            "hash": args.hash,
        },
        "lc0": row.get("lc0", {}),
        "oracle_top2_gap_cp": oracle_gap,
        "tags": tags,
    }


def convert_chunk(indexed_rows: list[tuple[int, dict]],
                  args: argparse.Namespace) -> list[tuple[int, dict | None, Counter[str]]]:
    engine = UciEngine(args.engine, threads=args.threads, hash_mb=args.hash)
    out: list[tuple[int, dict | None, Counter[str]]] = []
    try:
        for index, row in indexed_rows:
            counters: Counter[str] = Counter()
            counters["rows"] += 1
            out.append((index, convert_row(row, engine, args, counters), counters))
    finally:
        engine.close()
    return out


def chunks(items: list[tuple[int, dict]], jobs: int) -> list[list[tuple[int, dict]]]:
    if jobs <= 1:
        return [items]
    jobs = min(jobs, len(items))
    size = (len(items) + jobs - 1) // jobs
    return [items[i:i + size] for i in range(0, len(items), size)]


def preselect_row(row: dict, args: argparse.Namespace) -> bool:
    if row.get("schema") != "enyo.lc0.v1":
        return False
    board = chess.Board(str(row["fen"]))
    moves = legal_policy_moves(row, board, args.max_moves_per_position)
    counters: Counter[str] = Counter()
    return passes_policy_filters(row, moves, args, counters)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Score LC0 policy candidate moves with a UCI oracle.")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--nodes", type=int, default=200000)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash", type=int, default=128)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--min-groups", type=int, default=1)
    ap.add_argument("--max-groups", type=int, default=0)
    ap.add_argument("--max-moves-per-position", type=int, default=8)
    ap.add_argument("--max-gap-cp", type=float, default=800.0)
    ap.add_argument("--min-oracle-gap-cp", type=float, default=0.0)
    ap.add_argument("--min-best-policy", type=float, default=0.0)
    ap.add_argument("--min-policy-gap-cp", type=float, default=0.0)
    ap.add_argument("--policy-score-scale-cp", type=float, default=50.0)
    ap.add_argument("--policy-floor", type=float, default=1e-4)
    ap.add_argument("--policy-gap-cap-cp", type=float, default=300.0)
    ap.add_argument("--preselect-multiplier", type=int, default=3)
    ap.add_argument("--progress", type=int, default=1000)
    args = ap.parse_args()

    if args.nodes <= 0 and args.depth <= 0:
        raise SystemExit("either --nodes or --depth must be positive")
    if args.max_moves_per_position < 2:
        raise SystemExit("--max-moves-per-position must be at least 2")

    start = time.monotonic()
    selected: list[tuple[int, dict]] = []
    input_rows = 0
    preselect_limit = (
        args.max_groups * max(1, args.preselect_multiplier)
        if args.max_groups > 0 else 0
    )
    with args.input.open(encoding="utf-8") as src:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            input_rows += 1
            row = json.loads(line)
            if not preselect_row(row, args):
                continue
            selected.append((line_no, row))
            if preselect_limit > 0 and len(selected) >= preselect_limit:
                break

    results: list[tuple[int, dict | None, Counter[str]]] = []
    work = chunks(selected, max(1, args.jobs))
    if len(work) == 1:
        results = convert_chunk(work[0], args)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(work)) as executor:
            futures = [executor.submit(convert_chunk, chunk, args) for chunk in work]
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())

    counters: Counter[str] = Counter()
    written_rows: list[tuple[int, dict]] = []
    for index, converted, row_counters in results:
        counters.update(row_counters)
        if converted is not None:
            written_rows.append((index, converted))
    written_rows.sort(key=lambda item: item[0])
    converted_groups = len(written_rows)
    if args.max_groups > 0:
        written_rows = written_rows[:args.max_groups]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as dst:
        for _index, row in written_rows:
            dst.write(json.dumps(row, separators=(",", ":")) + "\n")

    written = len(written_rows)
    if written < args.min_groups:
        raise SystemExit(f"written_groups={written} below min_groups={args.min_groups}")

    written_tags: Counter[str] = Counter()
    for _index, row in written_rows:
        for tag in row.get("tags", []):
            written_tags[str(tag)] += 1

    elapsed = time.monotonic() - start
    lines = [
        f"input_rows={input_rows}",
        f"preselected_rows={len(selected)}",
        f"scored_rows={counters['rows']}",
        f"converted_groups={converted_groups}",
        f"written_groups={written}",
        f"skipped_schema={counters['skipped_schema']}",
        f"skipped_few_moves={counters['skipped_few_moves']}",
        f"skipped_low_best_policy={counters['skipped_low_best_policy']}",
        f"skipped_low_policy_gap={counters['skipped_low_policy_gap']}",
        f"skipped_no_oracle_score={counters['skipped_no_oracle_score']}",
        f"skipped_low_oracle_gap={counters['skipped_low_oracle_gap']}",
        f"engine={args.engine}",
        f"nodes={args.nodes}",
        f"depth={args.depth}",
        f"jobs={args.jobs}",
        f"threads={args.threads}",
        f"hash={args.hash}",
        f"max_moves_per_position={args.max_moves_per_position}",
        f"max_gap_cp={args.max_gap_cp}",
        f"min_oracle_gap_cp={args.min_oracle_gap_cp}",
        f"min_best_policy={args.min_best_policy}",
        f"min_policy_gap_cp={args.min_policy_gap_cp}",
        f"elapsed_s={elapsed:.3f}",
        "tags=" + ",".join(
            f"{key}:{value}"
            for key, value in sorted(written_tags.items())
        ),
        f"output={args.out}",
    ]
    summary = "\n".join(lines) + "\n"
    print(summary, end="")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(summary, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
