#!/usr/bin/env python3
"""Convert LC0 JSONL rows into Enyo child-ranking target groups."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import chess


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


def choose_best_move(row: dict, moves: list[dict], best_source: str) -> str | None:
    legal_moves = [move for move in moves if move.get("legal") is True]
    if not legal_moves:
        return None
    if best_source == "top-policy":
        return max(
            legal_moves,
            key=lambda move: (float(move.get("policy", 0.0)), -int(move.get("rank", 999999))),
        )["move"]
    if best_source == "best":
        for move in legal_moves:
            if "best" in role_list(move):
                return str(move["move"])
        return None
    if best_source == "played":
        for move in legal_moves:
            if "played" in role_list(move):
                return str(move["move"])
        return None
    raise ValueError(f"unknown best_source: {best_source}")


def policy_score(policy: float, best_policy: float, *,
                 floor: float, scale_cp: float, max_gap_cp: float) -> float:
    p = max(float(policy), floor)
    bp = max(float(best_policy), floor)
    raw_gap = max(0.0, scale_cp * (math.log(bp) - math.log(p)))
    return -min(raw_gap, max_gap_cp)


def convert_row(row: dict, args: argparse.Namespace,
                counters: Counter[str]) -> dict | None:
    if row.get("schema") != "enyo.lc0.v1":
        counters["skipped_schema"] += 1
        return None

    fen = str(row["fen"])
    board = chess.Board(fen)
    seen: set[str] = set()
    legal_moves: list[dict] = []
    for raw in row.get("moves", []):
        move_uci = str(raw.get("move", ""))
        if not move_uci or move_uci in seen or raw.get("legal") is not True:
            continue
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            counters["skipped_illegal_move"] += 1
            continue
        seen.add(move_uci)
        legal_moves.append(raw)

    best_move = choose_best_move(row, legal_moves, args.best_source)
    if not best_move:
        counters["skipped_missing_best"] += 1
        return None
    if len(legal_moves) < 2:
        counters["skipped_few_moves"] += 1
        return None

    best_policy = 0.0
    for move in legal_moves:
        if move["move"] == best_move:
            best_policy = float(move.get("policy", 0.0))
            break
    if best_policy <= 0.0:
        counters["skipped_zero_best_policy"] += 1
        return None

    out_moves = []
    for raw in legal_moves:
        move_uci = str(raw["move"])
        roles = role_list(raw)
        role = "best" if move_uci == best_move else "neighbor"
        out_moves.append({
            "move": move_uci,
            "score_cp": policy_score(
                float(raw.get("policy", 0.0)), best_policy,
                floor=args.policy_floor,
                scale_cp=args.policy_score_scale_cp,
                max_gap_cp=args.max_gap_cp,
            ),
            "role": role,
            "rank": raw.get("rank"),
            "policy": float(raw.get("policy", 0.0)),
            "origins": roles,
        })

    tags = list(dict.fromkeys([
        *[str(tag) for tag in row.get("tags", [])],
        "source:lc0_policy",
        "label:lc0_policy_logit",
        f"best_source:{args.best_source}",
        phase_tag(board),
    ]))
    for tag in tags:
        counters[f"tag:{tag}"] += 1

    return {
        "id": str(row.get("id") or f"lc0:{counters['rows']}"),
        "fen": fen,
        "best_move": best_move,
        "max_gap_cp": float(args.max_gap_cp),
        "moves": out_moves,
        "source": "lc0_policy",
        "source_file": row.get("source_file"),
        "record_index": row.get("record_index"),
        "license": row.get("license"),
        "score_pov": "parent_policy",
        "policy_score_scale_cp": float(args.policy_score_scale_cp),
        "policy_floor": float(args.policy_floor),
        "lc0": row.get("lc0", {}),
        "tags": tags,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert LC0 JSONL rows to child-ranking targets.")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--min-groups", type=int, default=1)
    ap.add_argument("--max-groups", type=int, default=0)
    ap.add_argument("--unique-fen", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--best-source", choices=["top-policy", "best", "played"],
                    default="top-policy")
    ap.add_argument("--policy-score-scale-cp", type=float, default=50.0)
    ap.add_argument("--policy-floor", type=float, default=1e-4)
    ap.add_argument("--max-gap-cp", type=float, default=300.0)
    args = ap.parse_args()

    counters: Counter[str] = Counter()
    seen_fens: set[str] = set()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as src, args.out.open(
            "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            counters["rows"] += 1
            row = json.loads(line)
            fen = str(row.get("fen", ""))
            if args.unique_fen and fen in seen_fens:
                counters["skipped_duplicate_fen"] += 1
                continue
            converted = convert_row(row, args, counters)
            if converted is None:
                continue
            if args.unique_fen:
                seen_fens.add(fen)
            dst.write(json.dumps(converted, separators=(",", ":")) + "\n")
            counters["written_groups"] += 1
            if args.max_groups > 0 and counters["written_groups"] >= args.max_groups:
                break

    if counters["written_groups"] < args.min_groups:
        raise SystemExit(
            f"written_groups={counters['written_groups']} below min_groups={args.min_groups}")

    lines = [
        f"rows={counters['rows']}",
        f"written_groups={counters['written_groups']}",
        f"skipped_duplicate_fen={counters['skipped_duplicate_fen']}",
        f"skipped_missing_best={counters['skipped_missing_best']}",
        f"skipped_few_moves={counters['skipped_few_moves']}",
        f"skipped_illegal_move={counters['skipped_illegal_move']}",
        f"best_source={args.best_source}",
        f"policy_score_scale_cp={args.policy_score_scale_cp}",
        f"policy_floor={args.policy_floor}",
        f"max_gap_cp={args.max_gap_cp}",
        "tags=" + ",".join(
            f"{key[4:]}:{value}"
            for key, value in sorted(counters.items())
            if key.startswith("tag:")
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
