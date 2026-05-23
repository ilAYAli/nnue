#!/usr/bin/env python3
"""Score all legal moves for repeated tail-regression targets."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import chess
import chess.engine


def score_cp(score: chess.engine.PovScore, turn: chess.Color) -> int:
    return score.pov(turn).score(mate_score=32000) or 0


def analyze_move(
        engine: chess.engine.SimpleEngine, board: chess.Board,
        move: chess.Move, nodes: int) -> int:
    child = board.copy(stack=False)
    root_turn = board.turn
    child.push(move)
    info = engine.analyse(child, chess.engine.Limit(nodes=nodes))
    return score_cp(info["score"], root_turn)


def int_value(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def side_name(fen: str) -> str:
    return "White" if chess.Board(fen).turn == chess.WHITE else "Black"


def target_from_jsonl_row(row: dict[str, object], index: int) -> dict[str, str]:
    fen = str(row["fen"])
    board = chess.Board(fen)
    return {
        "log": str(row.get("source_file") or row.get("source") or "positions_jsonl"),
        "fullmove": str(board.fullmove_number),
        "ply": str(row.get("ply", index)),
        "side": side_name(fen),
        "fen": fen,
        "oracle_moves": "",
        "reference_moves": "",
        "candidate_moves": "",
        "worst_diff": "0",
        "hits": "0",
    }


def reservoir_sample_positions(args: argparse.Namespace) -> list[dict[str, str]]:
    rng = random.Random(args.seed)
    targets: list[dict[str, str]] = []
    accepted = 0
    seen: set[str] = set()
    with args.positions_jsonl.expanduser().open(encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            fen = row.get("fen")
            if not isinstance(fen, str) or fen in seen:
                continue
            try:
                board = chess.Board(fen)
            except ValueError:
                continue
            if args.min_ply > 0:
                ply = 2 * (board.fullmove_number - 1) + (0 if board.turn == chess.WHITE else 1)
                if ply < args.min_ply:
                    continue
            score = int_value(row.get("score", row.get("score_cp")), 0)
            if args.max_abs_cp > 0 and abs(score) > args.max_abs_cp:
                continue
            if board.is_game_over():
                continue
            seen.add(fen)
            accepted += 1
            target = target_from_jsonl_row(row, index)
            if args.max_targets <= 0:
                targets.append(target)
                continue
            if len(targets) < args.max_targets:
                targets.append(target)
                continue
            slot = rng.randrange(accepted)
            if slot < args.max_targets:
                targets[slot] = target
    return targets


def load_targets(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.targets:
        with args.targets.expanduser().open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    return reservoir_sample_positions(args)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score every legal move in repeated tail target FENs.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--targets", type=Path)
    source.add_argument("--positions-jsonl", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--engine", default="stockfish")
    parser.add_argument("--nodes", type=int, default=200000)
    parser.add_argument("--hash", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--syzygy-path", default="")
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--max-abs-cp", type=int, default=0)
    parser.add_argument("--min-ply", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    targets = load_targets(args)

    rows: list[dict[str, object]] = []
    with chess.engine.SimpleEngine.popen_uci(args.engine) as engine:
        options: dict[str, object] = {"Hash": args.hash, "Threads": args.threads}
        if args.syzygy_path:
            options["SyzygyPath"] = str(Path(args.syzygy_path).expanduser())
        engine.configure(options)
        for target in targets:
            board = chess.Board(target["fen"])
            scored = []
            for move in board.legal_moves:
                cp = analyze_move(engine, board, move, args.nodes)
                scored.append((move.uci(), cp))
            scored.sort(key=lambda item: item[1], reverse=True)
            best_score = scored[0][1] if scored else 0
            for rank, (move, cp) in enumerate(scored, start=1):
                rows.append({
                    "log": target["log"],
                    "fullmove": target["fullmove"],
                    "ply": target["ply"],
                    "side": target["side"],
                    "fen": target["fen"],
                    "move": move,
                    "rank": rank,
                    "score_cp": cp,
                    "gap_cp": best_score - cp,
                    "target_oracle_moves": target["oracle_moves"],
                    "target_reference_moves": target["reference_moves"],
                    "target_candidate_moves": target["candidate_moves"],
                    "target_worst_diff": target["worst_diff"],
                    "target_hits": target["hits"],
                })

    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "log",
        "fullmove",
        "ply",
        "side",
        "fen",
        "move",
        "rank",
        "score_cp",
        "gap_cp",
        "target_oracle_moves",
        "target_reference_moves",
        "target_candidate_moves",
        "target_worst_diff",
        "target_hits",
    ]
    with args.out.expanduser().open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"scored {len(targets)} targets, {len(rows)} legal moves")
    print(f"wrote {args.out.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
