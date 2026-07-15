#!/usr/bin/env python3
"""Build a deterministic phase-balanced FEN corpus from PGN."""
from __future__ import annotations
import argparse
import random
from pathlib import Path
import chess
import chess.pgn

def phase(board: chess.Board) -> str:
    count = len(board.piece_map())
    if count >= 28:
        return "opening"
    if count >= 18:
        return "middlegame"
    if count >= 10:
        return "late"
    return "endgame"

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--rows", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=5090)
    args = ap.parse_args()
    phases = ("opening", "middlegame", "late", "endgame")
    quota = args.rows // len(phases)
    limits = {name: quota for name in phases}
    limits["endgame"] += args.rows - quota * len(phases)
    rows: dict[str, list[str]] = {name: [] for name in phases}
    rng = random.Random(args.seed)
    games = 0
    with args.pgn.open(encoding="utf-8", errors="replace") as handle:
        while sum(len(items) for items in rows.values()) < args.rows:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            games += 1
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                name = phase(board)
                if len(rows[name]) < limits[name]:
                    rows[name].append(board.fen())
                elif len(rows[name]) < limits[name] * 2:
                    rows[name][rng.randrange(limits[name])] = board.fen()
            if games % 1000 == 0:
                print(f"games={games} rows={sum(map(len, rows.values()))}", flush=True)
    missing = {name: limits[name] - len(rows[name]) for name in phases if len(rows[name]) < limits[name]}
    if missing:
        raise SystemExit(f"could not fill phase quotas: {missing}")
    output = [fen for name in phases for fen in rows[name][:limits[name]]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output) + "\n")
    print(f"games={games} rows={len(output)} quotas={limits}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
