#!/usr/bin/env python3
"""Validate Enyo replay JSONL target extraction output."""
from __future__ import annotations

import argparse
import json
from json import JSONDecodeError
from pathlib import Path

import chess


REQUIRED_TOP_LEVEL = {
    "schema",
    "score_pov",
    "fen",
    "best_move",
    "candidate_move",
    "logged_move",
    "oracle_move",
    "moves",
    "provenance",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_row(row: dict, line_no: int, *, fail_if_dirty_candidate: bool) -> int:
    missing = REQUIRED_TOP_LEVEL.difference(row)
    require(not missing, f"line {line_no}: missing fields: {sorted(missing)}")
    require(row["schema"] == "enyo.replay.v1",
            f"line {line_no}: unexpected schema {row['schema']!r}")
    require(row["score_pov"] == "parent",
            f"line {line_no}: unexpected score_pov {row['score_pov']!r}")

    board = chess.Board(str(row["fen"]))
    moves = row["moves"]
    require(isinstance(moves, list) and moves,
            f"line {line_no}: moves must be a non-empty list")

    scored = 0
    seen: set[str] = set()
    for item in moves:
        require(isinstance(item, dict), f"line {line_no}: move row is not object")
        move_uci = str(item.get("move", ""))
        require(move_uci, f"line {line_no}: move missing move")
        require(move_uci not in seen, f"line {line_no}: duplicate move {move_uci}")
        seen.add(move_uci)
        move = chess.Move.from_uci(move_uci)
        require(move in board.legal_moves,
                f"line {line_no}: illegal move {move_uci}")
        require(item.get("legal") is True,
                f"line {line_no}: move {move_uci} is not marked legal")
        require("score_cp" in item,
                f"line {line_no}: move {move_uci} missing score_cp")
        float(item["score_cp"])
        scored += 1

    for key in ("best_move", "candidate_move", "logged_move", "oracle_move"):
        move_uci = str(row[key])
        require(move_uci in seen, f"line {line_no}: {key}={move_uci} not scored")

    provenance = row["provenance"]
    require(isinstance(provenance, dict),
            f"line {line_no}: provenance must be object")
    candidate = provenance.get("candidate", {})
    require(isinstance(candidate, dict),
            f"line {line_no}: candidate provenance must be object")
    candidate_id = str(candidate.get("id", ""))
    if fail_if_dirty_candidate:
        require("(dirty)" not in candidate_id,
                f"line {line_no}: dirty candidate binary: {candidate_id}")

    return scored


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate Enyo replay JSONL target extraction output.")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--min-rows", type=int, default=1)
    ap.add_argument("--fail-if-dirty-candidate", action="store_true")
    ap.add_argument("--fail-if-history-sensitive", action="store_true")
    args = ap.parse_args()

    rows = 0
    scored_moves = 0
    history_sensitive = 0
    with args.input.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                require(isinstance(row, dict),
                        f"line {line_no}: row is not object")
                scored_moves += validate_row(
                    row, line_no,
                    fail_if_dirty_candidate=args.fail_if_dirty_candidate)
            except (JSONDecodeError, ValueError) as exc:
                raise SystemExit(f"{args.input}: {exc}") from exc
            rows += 1
            if row.get("history_sensitive"):
                history_sensitive += 1

    require(rows >= args.min_rows,
            f"rows={rows} below min_rows={args.min_rows}")
    if args.fail_if_history_sensitive:
        require(history_sensitive == 0,
                f"history_sensitive={history_sensitive} rows are not valid normal training targets")
    print(
        f"rows={rows} scored_moves={scored_moves} "
        f"history_sensitive={history_sensitive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
