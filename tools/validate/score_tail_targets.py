#!/usr/bin/env python3
"""Score all legal moves for repeated tail-regression targets."""

from __future__ import annotations

import argparse
import csv
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score every legal move in repeated tail target FENs.")
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--engine", default="stockfish")
    parser.add_argument("--nodes", type=int, default=200000)
    parser.add_argument("--hash", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--syzygy-path", default="")
    args = parser.parse_args()

    with args.targets.expanduser().open(newline="", encoding="utf-8") as handle:
        targets = list(csv.DictReader(handle))

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
