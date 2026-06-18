#!/usr/bin/env python3
"""Convert child-ranking target groups to train_pairwise hard-pair rows."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import chess


def child_fen(parent_fen: str, move_uci: str) -> str:
    board = chess.Board(parent_fen)
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        raise ValueError(f"illegal move {move_uci} in {parent_fen}")
    board.push(move)
    return board.fen()


def roles(raw: dict) -> list[str]:
    value = raw.get("roles", raw.get("role", []))
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def best_move(row: dict, moves: list[dict]) -> dict | None:
    explicit = str(row.get("best_move", ""))
    if explicit:
        for move in moves:
            if str(move.get("move", "")) == explicit:
                return move
    for move in moves:
        if "best" in roles(move):
            return move
    return max(moves, key=lambda move: float(move.get("score_cp", 0.0)),
               default=None)


def convert_row(row: dict, args: argparse.Namespace,
                counters: Counter[str]) -> list[dict]:
    parent_fen = str(row["fen"])
    source = str(row.get("source", "child_targets"))
    moves = [
        move for move in row.get("moves", [])
        if str(move.get("move", "")) and "score_cp" in move
    ]
    best = best_move(row, moves)
    if best is None:
        counters["skipped_missing_best"] += 1
        return []

    best_uci = str(best["move"])
    best_parent_score = float(best["score_cp"])
    best_child_fen = child_fen(parent_fen, best_uci)
    out: list[dict] = []
    for move in moves:
        move_uci = str(move["move"])
        if move_uci == best_uci:
            continue
        parent_score = float(move["score_cp"])
        gap = best_parent_score - parent_score
        if gap < args.min_gap_cp:
            counters["skipped_low_gap"] += 1
            continue
        if args.max_gap_cp > 0:
            gap = min(gap, args.max_gap_cp)

        pair_id = f"{row.get('id', counters['rows'])}:{best_uci}:{move_uci}"
        common = {
            "wdl": 0.5,
            "source_type": "child_target_pair",
            "hard_parent_fen": parent_fen,
            "hard_parent_source": source,
            "hard_parent_line": int(row.get("record_index", counters["rows"])),
            "hard_parent_ply": 0,
            "hard_parent_severity": "child_target",
            "hard_parent_loss_cp": gap,
            "hard_parent_played": move_uci,
            "hard_parent_best": best_uci,
            "child_target_id": str(row.get("id", "")),
            "child_target_pair_id": pair_id,
            "child_target_tags": row.get("tags", []),
            "score_pov": "child_stm",
        }
        # train_pairwise expects child side-to-move scores. Child target groups
        # store parent-POV scores, so negate them after pushing the move.
        out.append({
            **common,
            "fen": child_fen(parent_fen, move_uci),
            "score": -parent_score,
            "hard_child_kind": "played",
            "hard_child_move": move_uci,
        })
        out.append({
            **common,
            "fen": best_child_fen,
            "score": -best_parent_score,
            "hard_child_kind": "best",
            "hard_child_move": best_uci,
        })
        counters["pairs"] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert child-ranking JSONL groups to train_pairwise rows.")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--min-pairs", type=int, default=1)
    ap.add_argument("--min-gap-cp", type=float, default=1.0)
    ap.add_argument("--max-gap-cp", type=float, default=800.0)
    args = ap.parse_args()

    counters: Counter[str] = Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as src, args.out.open(
            "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            counters["rows"] += 1
            row = json.loads(line)
            try:
                converted = convert_row(row, args, counters)
            except ValueError:
                counters["skipped_illegal"] += 1
                continue
            for item in converted:
                dst.write(json.dumps(item, sort_keys=True, separators=(",", ":")))
                dst.write("\n")
                counters["written_rows"] += 1

    if counters["pairs"] < args.min_pairs:
        raise SystemExit(
            f"pairs={counters['pairs']} below min_pairs={args.min_pairs}")

    summary = "\n".join([
        f"rows={counters['rows']}",
        f"pairs={counters['pairs']}",
        f"written_rows={counters['written_rows']}",
        f"skipped_missing_best={counters['skipped_missing_best']}",
        f"skipped_low_gap={counters['skipped_low_gap']}",
        f"skipped_illegal={counters['skipped_illegal']}",
        f"min_gap_cp={args.min_gap_cp}",
        f"max_gap_cp={args.max_gap_cp}",
        f"output={args.out}",
    ]) + "\n"
    print(summary, end="")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(summary, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
