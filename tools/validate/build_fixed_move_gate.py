#!/usr/bin/env python3
"""Build a fixed parent-position move-choice gate from target JSONL files."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import chess


SCHEMA = "enyo.move_choice_gate.v1"


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
    explicit = str(row.get("best_move", ""))
    if explicit:
        for move in moves:
            if str(move.get("move", "")) == explicit:
                return move
    for move in moves:
        if "best" in roles(move):
            return move
    return max(moves, key=lambda move: float(move.get("score_cp", 0.0)), default=None)


def move_rank(move: dict[str, Any], default: int) -> int:
    value = move.get("rank", default)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def row_tags(row: dict[str, Any], source_label: str) -> list[str]:
    tags = [str(tag) for tag in row.get("tags", [])]
    if source_label and f"gate_source:{source_label}" not in tags:
        tags.append(f"gate_source:{source_label}")
    return tags


def child_target_cases(
    row: dict[str, Any],
    *,
    source_label: str,
    min_gap_cp: float,
    max_gap_cp: float,
    max_per_parent: int,
    counters: Counter[str],
) -> list[dict[str, Any]]:
    parent_fen = str(row.get("fen", ""))
    moves = [
        move for move in row.get("moves", [])
        if str(move.get("move", "")) and "score_cp" in move
    ]
    best = best_move(row, moves)
    if best is None:
        counters["skipped_missing_best"] += 1
        return []

    best_uci = str(best["move"])
    if not legal_move(parent_fen, best_uci):
        counters["skipped_illegal_best"] += 1
        return []

    best_score = float(best["score_cp"])
    candidates: list[tuple[float, dict[str, Any]]] = []
    for move in moves:
        move_uci = str(move["move"])
        if move_uci == best_uci:
            continue
        if not legal_move(parent_fen, move_uci):
            counters["skipped_illegal_neighbor"] += 1
            continue
        gap = best_score - float(move["score_cp"])
        if gap < min_gap_cp:
            counters["skipped_low_gap"] += 1
            continue
        if max_gap_cp > 0:
            gap = min(gap, max_gap_cp)
        candidates.append((gap, move))

    candidates.sort(
        key=lambda item: (-item[0], move_rank(item[1], 999999), str(item[1]["move"]))
    )
    if max_per_parent > 0:
        candidates = candidates[:max_per_parent]

    out: list[dict[str, Any]] = []
    source_id = str(row.get("id", f"row{counters['child_rows']}"))
    for gap, move in candidates:
        move_uci = str(move["move"])
        case_id = f"{source_label}:{source_id}:{best_uci}>{move_uci}"
        out.append({
            "schema": SCHEMA,
            "id": case_id,
            "source": str(row.get("source", "child_targets")),
            "source_label": source_label,
            "source_file": str(row.get("source_file", "")),
            "record_index": int(row.get("record_index", counters["child_rows"])),
            "fen": parent_fen,
            "best": best_uci,
            "played": move_uci,
            "loss_cp": int(round(gap)),
            "score_pov": str(row.get("score_pov", "parent")),
            "best_score_cp": best_score,
            "played_score_cp": float(move["score_cp"]),
            "best_rank": move_rank(best, 0),
            "played_rank": move_rank(move, 0),
            "tags": row_tags(row, source_label),
        })
        counters["child_cases"] += 1
    return out


def normalize_case(row: dict[str, Any], source_label: str, counters: Counter[str]) -> dict[str, Any] | None:
    fen = str(row.get("fen", ""))
    best = str(row.get("best", row.get("best_move", "")))
    played = str(row.get("played", row.get("candidate_move", row.get("reference_move", ""))))
    if not fen or not best or not played:
        counters["skipped_malformed_case"] += 1
        return None
    if not legal_move(fen, best) or not legal_move(fen, played):
        counters["skipped_illegal_case"] += 1
        return None
    out = dict(row)
    out.update({
        "schema": SCHEMA,
        "id": str(row.get("id", f"{source_label}:case{counters['case_rows']}")),
        "source_label": source_label,
        "fen": fen,
        "best": best,
        "played": played,
        "loss_cp": int(round(float(row.get("loss_cp", 0)))),
    })
    if "tags" not in out:
        out["tags"] = row_tags(row, source_label)
    counters["case_rows"] += 1
    return out


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


def dedupe_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row["fen"]), str(row["best"]), str(row["played"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def limit_per_source(rows: list[dict[str, Any]], max_per_source: int) -> list[dict[str, Any]]:
    if max_per_source <= 0:
        return rows
    counts: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    for row in rows:
        source = str(row.get("source_label", row.get("source", "")))
        if counts[source] >= max_per_source:
            continue
        counts[source] += 1
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build a deterministic move-choice gate from child targets and case JSONL.")
    ap.add_argument("--child-targets", action="append", default=[],
                    metavar="LABEL=PATH",
                    help="Child-target JSONL. May be repeated.")
    ap.add_argument("--cases", action="append", default=[], metavar="LABEL=PATH",
                    help="Existing move-gate case JSONL. May be repeated.")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--min-gap-cp", type=float, default=30.0)
    ap.add_argument("--max-gap-cp", type=float, default=800.0)
    ap.add_argument("--max-per-parent", type=int, default=1)
    ap.add_argument("--max-per-source", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-cases", type=int, default=1)
    ap.add_argument("--no-dedupe", action="store_true")
    args = ap.parse_args()

    counters: Counter[str] = Counter()
    cases: list[dict[str, Any]] = []

    for label, path in parse_labeled_paths(args.child_targets):
        for row in read_jsonl(path):
            counters["child_rows"] += 1
            cases.extend(child_target_cases(
                row,
                source_label=label,
                min_gap_cp=args.min_gap_cp,
                max_gap_cp=args.max_gap_cp,
                max_per_parent=args.max_per_parent,
                counters=counters,
            ))

    for label, path in parse_labeled_paths(args.cases):
        for row in read_jsonl(path):
            normalized = normalize_case(row, label, counters)
            if normalized is not None:
                cases.append(normalized)

    cases.sort(key=lambda row: (
        str(row.get("source_label", "")),
        -int(row.get("loss_cp", 0)),
        str(row.get("id", "")),
    ))
    if not args.no_dedupe:
        cases = dedupe_cases(cases)
    cases = limit_per_source(cases, args.max_per_source)
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
        f"min_gap_cp={args.min_gap_cp}",
        f"max_gap_cp={args.max_gap_cp}",
        f"max_per_parent={args.max_per_parent}",
        f"max_per_source={args.max_per_source}",
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
