#!/usr/bin/env python3
"""Convert Enyo replay JSONL rows into child-ranking target groups."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess


def move_role(row: dict, move: dict) -> str:
    move_uci = str(move["move"])
    if move_uci == row.get("best_move"):
        return "best"
    roles = move.get("role", [])
    if isinstance(roles, str):
        roles = [roles]
    if "candidate" in roles:
        return "candidate"
    if "logged" in roles:
        return "logged"
    if "reference" in roles:
        return "reference"
    return "neighbor"


def convert_row(row: dict) -> dict | None:
    fen = str(row["fen"])
    board = chess.Board(fen)
    best_move = str(row["best_move"])

    moves = []
    seen: set[str] = set()
    for raw in row.get("moves", []):
        move_uci = str(raw.get("move", ""))
        if not move_uci or move_uci in seen or "score_cp" not in raw:
            continue
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            continue
        seen.add(move_uci)
        moves.append({
            "move": move_uci,
            "score_cp": float(raw["score_cp"]),
            "role": move_role(row, raw),
            "rank": raw.get("rank"),
            "origins": raw.get("origins", raw.get("role", [])),
        })

    if best_move not in seen or len(moves) < 2:
        return None

    tags = list(row.get("tags", []))
    if "source:replay_loss" not in tags:
        tags.append("source:replay_loss")

    return {
        "id": str(row.get("id") or row.get("group_id")),
        "fen": fen,
        "best_move": best_move,
        "max_gap_cp": float(row.get("max_gap_cp", 800)),
        "moves": moves,
        "source": "replay_loss",
        "source_log": row.get("source_log", row.get("log_path")),
        "source_file": row.get("source_file"),
        "game_id": row.get("game_id"),
        "ply": row.get("ply"),
        "fullmove": row.get("fullmove"),
        "side_to_move": row.get("side_to_move"),
        "candidate_move": row.get("candidate_move"),
        "reference_move": row.get("reference_move"),
        "logged_move": row.get("logged_move"),
        "oracle_move": row.get("oracle_move"),
        "history_sensitive": bool(row.get("history_sensitive", False)),
        "tags": tags,
        "score_pov": row.get("score_pov", "parent"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert replay JSONL to child-ranking target JSONL.")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--min-groups", type=int, default=1)
    args = ap.parse_args()

    rows = 0
    written = 0
    skipped = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as src, args.out.open(
            "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            rows += 1
            converted = convert_row(json.loads(line))
            if converted is None:
                skipped += 1
                continue
            dst.write(json.dumps(converted, separators=(",", ":")) + "\n")
            written += 1

    if written < args.min_groups:
        raise SystemExit(
            f"written_groups={written} below min_groups={args.min_groups}")

    summary = (
        f"rows={rows}\n"
        f"written_groups={written}\n"
        f"skipped_groups={skipped}\n"
        f"output={args.out}\n"
    )
    print(summary, end="")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(summary, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
