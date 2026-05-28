#!/usr/bin/env python3
"""Build child-ranking targets from a fastchess smoke PGN."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import shlex
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn


SCORE_RE = re.compile(r"\bscore\s+(cp|mate)\s+(-?\d+)\b")


def mate_to_cp(mate: int) -> int:
    sign = 1 if mate > 0 else -1
    return sign * (32000 - min(abs(mate), 1000))


def phase_tag(board: chess.Board) -> str:
    nonking = max(0, len(board.piece_map()) - 2)
    if nonking <= 8:
        return "phase:endgame"
    if nonking <= 20:
        return "phase:late"
    if nonking <= 28:
        return "phase:midgame"
    return "phase:opening"


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
        self.send("uci")
        self.wait_for("uciok")
        self.setoption("Threads", str(threads))
        self.setoption("Hash", str(hash_mb))
        if net:
            self.setoption("nnue_file", net)
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

    def bestmove(self, fen: str, *, nodes: int, depth: int) -> str | None:
        self.send(f"position fen {fen}")
        if nodes > 0:
            self.send(f"go nodes {nodes}")
        else:
            self.send(f"go depth {depth}")
        while True:
            line = self.readline()
            if line.startswith("bestmove "):
                move = line.split()[1]
                return None if move == "(none)" else move

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


@dataclass(frozen=True)
class PositionCase:
    case_id: str
    fen: str
    logged_move: str
    logged_engine: str
    game_index: int
    ply: int
    result: str
    candidate_outcome: str
    source_pgn: str
    tags: tuple[str, ...]


def candidate_outcome(result: str, candidate_is_white: bool) -> str:
    if result == "1/2-1/2":
        return "draw"
    if result == "1-0":
        return "candidate_win" if candidate_is_white else "candidate_loss"
    if result == "0-1":
        return "candidate_loss" if candidate_is_white else "candidate_win"
    return "unknown"


def collect_positions(args: argparse.Namespace) -> list[PositionCase]:
    cases: list[PositionCase] = []
    seen_fens: set[str] = set()
    with args.pgn.open(encoding="utf-8", errors="replace") as handle:
        game_index = 0
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            game_index += 1
            headers = game.headers
            result = headers.get("Result", "*")
            candidate_white = headers.get("White", "") == args.candidate_name
            outcome = candidate_outcome(result, candidate_white)
            per_game = 0
            board = game.board()
            node = game
            ply = 0
            while node.variations:
                child = node.variation(0)
                move = child.move
                logged_engine = (
                    headers.get("White", "")
                    if board.turn == chess.WHITE
                    else headers.get("Black", "")
                )
                if (
                    ply >= args.min_ply
                    and per_game < args.max_per_game
                    and (not args.only_candidate_losses
                         or outcome == "candidate_loss")
                ):
                    fen = board.fen()
                    if not args.unique_fen or fen not in seen_fens:
                        if args.unique_fen:
                            seen_fens.add(fen)
                        tags = (
                            "source:smoke_pgn",
                            f"outcome:{outcome}",
                            f"logged:{logged_engine}",
                            phase_tag(board),
                        )
                        cases.append(PositionCase(
                            case_id=f"smoke-{game_index}-{ply}",
                            fen=fen,
                            logged_move=move.uci(),
                            logged_engine=logged_engine,
                            game_index=game_index,
                            ply=ply,
                            result=result,
                            candidate_outcome=outcome,
                            source_pgn=str(args.pgn),
                            tags=tags,
                        ))
                        per_game += 1
                        if args.max_positions > 0 and len(cases) >= args.max_positions:
                            return cases
                board.push(move)
                node = child
                ply += 1
    return cases


def child_fen(board: chess.Board, move_uci: str) -> str:
    child = board.copy(stack=False)
    child.push(chess.Move.from_uci(move_uci))
    return child.fen()


def score_parent_cp(
    oracle: UciEngine,
    board: chess.Board,
    move_uci: str,
    args: argparse.Namespace,
) -> int | None:
    stm_score = oracle.score_stm_cp(
        child_fen(board, move_uci),
        nodes=args.oracle_nodes,
        depth=args.oracle_depth,
    )
    return None if stm_score is None else -stm_score


def convert_case(
    case: PositionCase,
    candidate: UciEngine,
    reference: UciEngine,
    oracle: UciEngine,
    args: argparse.Namespace,
) -> tuple[dict | None, Counter[str]]:
    counters: Counter[str] = Counter()
    board = chess.Board(case.fen)
    legal = {move.uci() for move in board.legal_moves}
    candidate_move = candidate.bestmove(
        case.fen, nodes=args.search_nodes, depth=args.search_depth)
    reference_move = reference.bestmove(
        case.fen, nodes=args.search_nodes, depth=args.search_depth)
    if not candidate_move or not reference_move:
        counters["skipped_no_bestmove"] += 1
        return None, counters

    raw_moves = [
        ("candidate", candidate_move),
        ("reference", reference_move),
        ("logged", case.logged_move),
    ]
    role_map: dict[str, list[str]] = {}
    for role, move in raw_moves:
        if move in legal:
            role_map.setdefault(move, []).append(role)
        else:
            counters[f"skipped_illegal_{role}"] += 1

    if len(role_map) < 2:
        counters["skipped_same_moves"] += 1
        return None, counters

    scored = []
    for move_uci, roles in role_map.items():
        score = score_parent_cp(oracle, board, move_uci, args)
        if score is None:
            counters["skipped_no_oracle_score"] += 1
            return None, counters
        scored.append({
            "move": move_uci,
            "score_cp": int(score),
            "role": "neighbor",
            "origins": roles,
        })
    best = max(scored, key=lambda item: item["score_cp"])
    second = max(
        (move["score_cp"] for move in scored if move["move"] != best["move"]),
        default=best["score_cp"],
    )
    gap = float(best["score_cp"] - second)
    if gap < args.min_oracle_gap_cp:
        counters["skipped_low_oracle_gap"] += 1
        return None, counters
    for move in scored:
        if move["move"] == best["move"]:
            move["role"] = "best"
            break

    if candidate_move != reference_move:
        counters["candidate_reference_differ"] += 1
    if candidate_move == best["move"]:
        counters["candidate_best"] += 1
    if reference_move == best["move"]:
        counters["reference_best"] += 1

    tags = list(dict.fromkeys([
        *case.tags,
        "label:oracle_child_cp",
    ]))
    for tag in tags:
        counters[f"tag:{tag}"] += 1

    return {
        "id": case.case_id,
        "fen": case.fen,
        "best_move": str(best["move"]),
        "max_gap_cp": float(args.max_gap_cp),
        "moves": scored,
        "source": "smoke_pgn",
        "source_pgn": case.source_pgn,
        "game_index": case.game_index,
        "ply": case.ply,
        "result": case.result,
        "candidate_outcome": case.candidate_outcome,
        "candidate_move": candidate_move,
        "reference_move": reference_move,
        "logged_move": case.logged_move,
        "logged_engine": case.logged_engine,
        "score_pov": "parent",
        "oracle_top2_gap_cp": gap,
        "oracle": {
            "engine": args.oracle_engine,
            "nodes": args.oracle_nodes,
            "depth": args.oracle_depth,
            "threads": args.oracle_threads,
            "hash": args.oracle_hash,
        },
        "search": {
            "engine": args.engine,
            "nodes": args.search_nodes,
            "depth": args.search_depth,
            "threads": args.threads,
            "hash": args.hash,
        },
        "tags": tags,
    }, counters


def chunks(items: list[PositionCase], jobs: int) -> list[list[PositionCase]]:
    if jobs <= 1:
        return [items]
    jobs = min(jobs, len(items))
    size = (len(items) + jobs - 1) // jobs
    return [items[i:i + size] for i in range(0, len(items), size)]


def convert_chunk(
    cases: list[PositionCase],
    args: argparse.Namespace,
) -> list[tuple[str, dict | None, Counter[str]]]:
    candidate = UciEngine(
        args.engine, threads=args.threads, hash_mb=args.hash,
        net=args.candidate_net)
    reference = UciEngine(
        args.engine, threads=args.threads, hash_mb=args.hash,
        net=args.reference_net)
    oracle = UciEngine(
        args.oracle_engine, threads=args.oracle_threads,
        hash_mb=args.oracle_hash)
    out = []
    try:
        for case in cases:
            row, counters = convert_case(case, candidate, reference, oracle, args)
            counters["positions"] += 1
            out.append((case.case_id, row, counters))
    finally:
        candidate.close()
        reference.close()
        oracle.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build oracle-scored child-ranking targets from smoke PGN positions.")
    ap.add_argument("--pgn", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--candidate-net", required=True)
    ap.add_argument("--reference-net", required=True)
    ap.add_argument("--oracle-engine", required=True)
    ap.add_argument("--candidate-name", default="candidate")
    ap.add_argument("--reference-name", default="reference")
    ap.add_argument("--search-nodes", type=int, default=20000)
    ap.add_argument("--search-depth", type=int, default=0)
    ap.add_argument("--oracle-nodes", type=int, default=100000)
    ap.add_argument("--oracle-depth", type=int, default=12)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--oracle-threads", type=int, default=1)
    ap.add_argument("--oracle-hash", type=int, default=128)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--min-ply", type=int, default=8)
    ap.add_argument("--max-per-game", type=int, default=8)
    ap.add_argument("--max-positions", type=int, default=1000)
    ap.add_argument("--max-gap-cp", type=float, default=800.0)
    ap.add_argument("--min-oracle-gap-cp", type=float, default=20.0)
    ap.add_argument("--min-groups", type=int, default=1)
    ap.add_argument("--unique-fen", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--only-candidate-losses", action=argparse.BooleanOptionalAction,
                    default=False)
    args = ap.parse_args()

    if args.search_nodes <= 0 and args.search_depth <= 0:
        raise SystemExit("either --search-nodes or --search-depth must be positive")
    if args.oracle_nodes <= 0 and args.oracle_depth <= 0:
        raise SystemExit("either --oracle-nodes or --oracle-depth must be positive")

    start = time.monotonic()
    cases = collect_positions(args)
    if not cases:
        raise SystemExit("no sampled positions")

    results: list[tuple[str, dict | None, Counter[str]]] = []
    work = chunks(cases, max(1, args.jobs))
    if len(work) == 1:
        results = convert_chunk(work[0], args)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(work)) as executor:
            futures = [executor.submit(convert_chunk, chunk, args) for chunk in work]
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())

    order = {case.case_id: index for index, case in enumerate(cases)}
    results.sort(key=lambda item: order[item[0]])
    counters: Counter[str] = Counter()
    rows = []
    for _case_id, row, row_counters in results:
        counters.update(row_counters)
        if row is not None:
            rows.append(row)

    if len(rows) < args.min_groups:
        raise SystemExit(f"written_groups={len(rows)} below min_groups={args.min_groups}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    written_tags: Counter[str] = Counter()
    for row in rows:
        written_tags.update(str(tag) for tag in row.get("tags", []))

    elapsed = time.monotonic() - start
    lines = [
        f"sampled_positions={len(cases)}",
        f"written_groups={len(rows)}",
        f"candidate_reference_differ={counters['candidate_reference_differ']}",
        f"candidate_best={counters['candidate_best']}",
        f"reference_best={counters['reference_best']}",
        f"skipped_same_moves={counters['skipped_same_moves']}",
        f"skipped_low_oracle_gap={counters['skipped_low_oracle_gap']}",
        f"skipped_no_bestmove={counters['skipped_no_bestmove']}",
        f"skipped_no_oracle_score={counters['skipped_no_oracle_score']}",
        f"search_nodes={args.search_nodes}",
        f"search_depth={args.search_depth}",
        f"oracle_nodes={args.oracle_nodes}",
        f"oracle_depth={args.oracle_depth}",
        f"jobs={args.jobs}",
        f"min_ply={args.min_ply}",
        f"max_per_game={args.max_per_game}",
        f"max_positions={args.max_positions}",
        f"min_oracle_gap_cp={args.min_oracle_gap_cp}",
        f"elapsed_s={elapsed:.3f}",
        "tags=" + ",".join(
            f"{tag}:{count}" for tag, count in sorted(written_tags.items())
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
