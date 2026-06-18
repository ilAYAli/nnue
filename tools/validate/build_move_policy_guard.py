#!/usr/bin/env python3
"""Build no-override guard cases from replay child-target JSONL."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import chess


SCHEMA = "enyo.move_policy_guard.v1"


def roles(raw: dict[str, Any]) -> list[str]:
    value = raw.get("roles", raw.get("role", []))
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def legal_move(fen: str, move_uci: str) -> bool:
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
    except ValueError:
        return False
    return move in board.legal_moves


def best_move(row: dict[str, Any], moves: list[dict[str, Any]]) -> dict[str, Any] | None:
    explicit = str(row.get("best_move", row.get("oracle_move", "")))
    if explicit:
        for move in moves:
            if str(move.get("move", "")) == explicit:
                return move
    for move in moves:
        if "best" in roles(move):
            return move
    return max(moves, key=lambda move: float(move.get("score_cp", 0.0)), default=None)


def current_move(row: dict[str, Any], moves: list[dict[str, Any]]) -> dict[str, Any] | None:
    for key in ("candidate_move", "logged_move", "reference_move"):
        explicit = str(row.get(key, ""))
        if explicit:
            for move in moves:
                if str(move.get("move", "")) == explicit:
                    return move
    for move in moves:
        origins = {str(item) for item in move.get("origins", [])}
        if origins & {"candidate", "logged", "candidate_root"}:
            return move
    return None


def row_tags(row: dict[str, Any], source_label: str) -> list[str]:
    tags = [str(tag) for tag in row.get("tags", [])]
    if source_label and f"guard_source:{source_label}" not in tags:
        tags.append(f"guard_source:{source_label}")
    return tags


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as src:
        for line in src:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_labeled_paths(items: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for item in items:
        if "=" in item:
            label, path = item.split("=", 1)
        else:
            path = item
            label = Path(path).stem
        out.append((label, Path(path)))
    return out


def build_guard_case(
    row: dict[str, Any],
    *,
    source_label: str,
    max_current_loss_cp: float,
    max_alternatives: int,
    counters: Counter[str],
) -> dict[str, Any] | None:
    fen = str(row.get("fen", ""))
    moves = [
        move for move in row.get("moves", [])
        if str(move.get("move", "")) and "score_cp" in move
    ]
    best = best_move(row, moves)
    current = current_move(row, moves)
    if best is None:
        counters["skipped_missing_best"] += 1
        return None
    if current is None:
        counters["skipped_missing_current"] += 1
        return None

    best_uci = str(best["move"])
    current_uci = str(current["move"])
    if not legal_move(fen, best_uci) or not legal_move(fen, current_uci):
        counters["skipped_illegal_best_or_current"] += 1
        return None

    best_score = float(best["score_cp"])
    current_score = float(current["score_cp"])
    current_loss = max(0.0, best_score - current_score)
    if current_loss > max_current_loss_cp:
        counters["skipped_current_too_weak"] += 1
        return None

    alternatives: list[dict[str, Any]] = []
    for move in moves:
        move_uci = str(move["move"])
        if move_uci == current_uci:
            continue
        if not legal_move(fen, move_uci):
            counters["skipped_illegal_alternative"] += 1
            continue
        score = float(move["score_cp"])
        alternatives.append({
            "move": move_uci,
            "score_cp": score,
            "rank": int(move.get("rank", 999999) or 999999),
            "delta_cp": score - current_score,
            "role": move.get("role", ""),
            "origins": move.get("origins", []),
        })

    alternatives.sort(key=lambda item: (
        -float(item["score_cp"]),
        int(item["rank"]),
        str(item["move"]),
    ))
    if max_alternatives > 0:
        alternatives = alternatives[:max_alternatives]
    if not alternatives:
        counters["skipped_no_alternatives"] += 1
        return None

    source_id = str(row.get("id", f"row{counters['rows']}"))
    counters["cases"] += 1
    return {
        "schema": SCHEMA,
        "id": f"{source_label}:{source_id}:{current_uci}",
        "source": str(row.get("source", "child_targets")),
        "source_label": source_label,
        "source_file": str(row.get("source_file", "")),
        "fen": fen,
        "current": current_uci,
        "best": best_uci,
        "current_score_cp": current_score,
        "best_score_cp": best_score,
        "current_loss_cp": int(round(current_loss)),
        "score_pov": str(row.get("score_pov", "parent")),
        "alternatives": alternatives,
        "tags": row_tags(row, source_label),
    }


def dedupe_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row["fen"]), str(row["current"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build no-override guard cases from child-target JSONL.")
    ap.add_argument("--child-targets", action="append", default=[],
                    metavar="LABEL=PATH", help="Child-target JSONL.")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--max-current-loss-cp", type=float, default=10.0)
    ap.add_argument("--max-alternatives", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-cases", type=int, default=1)
    ap.add_argument("--no-dedupe", action="store_true")
    args = ap.parse_args()

    counters: Counter[str] = Counter()
    cases: list[dict[str, Any]] = []
    for label, path in parse_labeled_paths(args.child_targets):
        for row in read_jsonl(path):
            counters["rows"] += 1
            case = build_guard_case(
                row,
                source_label=label,
                max_current_loss_cp=args.max_current_loss_cp,
                max_alternatives=args.max_alternatives,
                counters=counters,
            )
            if case is not None:
                cases.append(case)

    cases.sort(key=lambda row: (
        str(row.get("source_label", "")),
        int(row.get("current_loss_cp", 0)),
        str(row.get("id", "")),
    ))
    if not args.no_dedupe:
        cases = dedupe_cases(cases)
    if args.limit > 0:
        cases = cases[:args.limit]
    if len(cases) < args.min_cases:
        raise SystemExit(f"cases={len(cases)} below min_cases={args.min_cases}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as dst:
        for row in cases:
            dst.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            dst.write("\n")

    by_source = Counter(str(row.get("source_label", "")) for row in cases)
    summary_lines = [
        f"schema={SCHEMA}",
        f"cases={len(cases)}",
        f"sources={dict(sorted(by_source.items()))}",
        f"max_current_loss_cp={args.max_current_loss_cp}",
        f"max_alternatives={args.max_alternatives}",
        f"output={args.output}",
    ]
    for key in sorted(counters):
        summary_lines.append(f"{key}={counters[key]}")
    summary = "\n".join(summary_lines) + "\n"
    print(summary, end="")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(summary, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
