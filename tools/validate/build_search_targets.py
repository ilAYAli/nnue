#!/usr/bin/env python3
"""Build unified search-aware move-choice targets from scored legal-move CSVs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


MATE_SCORE_CP = 30000


@dataclass
class SourceSpec:
    name: str
    path: Path


@dataclass
class MoveScore:
    uci: str
    rank: int
    score_cp: int
    gap_cp: int
    flags: set[str] = field(default_factory=set)


@dataclass
class TargetGroup:
    source: str
    source_file: str
    log: str
    ply: str
    fullmove: str
    side: str
    fen: str
    moves: dict[str, MoveScore] = field(default_factory=dict)
    oracle_moves: set[str] = field(default_factory=set)
    reference_moves: set[str] = field(default_factory=set)
    candidate_moves: set[str] = field(default_factory=set)


def parse_source(value: str) -> SourceSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must use NAME=PATH")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("source name is empty")
    return SourceSpec(name=name, path=Path(path).expanduser())


def int_field(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, "") or default)
    except ValueError:
        return default


def split_moves(value: str | None) -> set[str]:
    if not value:
        return set()
    return {m.strip().lower() for m in value.replace(",", ";").split(";") if m.strip()}


def material_count(fen: str) -> int:
    board = fen.split()[0] if fen else ""
    return sum(1 for ch in board if ch.isalpha())


def phase_tag(pieces: int) -> str:
    if pieces <= 8:
        return "phase:endgame"
    if pieces <= 18:
        return "phase:late"
    if pieces <= 26:
        return "phase:midgame"
    return "phase:opening"


def target_id(source: str, log: str, ply: str, fen: str) -> str:
    raw = "\0".join((source, log, ply, fen))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def soft_policy(gaps: Iterable[int], temperature_cp: float) -> list[float]:
    if temperature_cp <= 0:
        return []
    values = [-min(max(gap, 0), MATE_SCORE_CP) / temperature_cp for gap in gaps]
    if not values:
        return []
    offset = max(values)
    exp_values = [math.exp(value - offset) for value in values]
    total = sum(exp_values)
    return [value / total for value in exp_values] if total > 0 else [0.0 for _ in exp_values]


def load_groups(sources: list[SourceSpec]) -> dict[tuple[str, str, str, str], TargetGroup]:
    groups: dict[tuple[str, str, str, str], TargetGroup] = {}
    for spec in sources:
        if not spec.path.exists():
            raise SystemExit(f"missing score CSV: {spec.path}")
        with spec.path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                fen = row.get("fen", "")
                move = row.get("move", "").lower()
                if not fen or not move:
                    continue
                key = (spec.name, row.get("log", ""), row.get("ply", ""), fen)
                group = groups.setdefault(key, TargetGroup(
                    source=spec.name,
                    source_file=str(spec.path),
                    log=row.get("log", ""),
                    ply=row.get("ply", ""),
                    fullmove=row.get("fullmove", ""),
                    side=row.get("side", ""),
                    fen=fen,
                ))
                rank = int_field(row, "rank", 999)
                score = int_field(row, "score_cp", 0)
                gap = int_field(row, "gap_cp", 0)
                info = group.moves.setdefault(move, MoveScore(move, rank, score, gap))
                if rank < info.rank:
                    info.rank = rank
                    info.score_cp = score
                    info.gap_cp = gap
                if rank == 1:
                    info.flags.add("best")
                    group.oracle_moves.add(move)
                group.oracle_moves.update(split_moves(row.get("target_oracle_moves")))
                group.reference_moves.update(split_moves(row.get("target_reference_moves")))
                group.reference_moves.update(split_moves(row.get("reference_move")))
                group.candidate_moves.update(split_moves(row.get("target_candidate_moves")))
                group.candidate_moves.update(split_moves(row.get("candidate_move")))
    return groups


def dedupe_groups(groups: list[TargetGroup]) -> list[TargetGroup]:
    seen: set[str] = set()
    out: list[TargetGroup] = []
    for group in groups:
        if group.fen in seen:
            continue
        seen.add(group.fen)
        out.append(group)
    return out


def row_for_group(group: TargetGroup, *, max_moves: int, policy_temperature_cp: float,
                  mate_gap_cp: int) -> dict[str, object]:
    all_moves = sorted(group.moves.values(), key=lambda item: (item.rank, -item.score_cp))
    marked = group.oracle_moves | group.reference_moves | group.candidate_moves
    if max_moves > 0:
        selected = {move.uci for move in all_moves[:max_moves]}
        selected.update(move for move in marked if move in group.moves)
        moves = [move for move in all_moves if move.uci in selected]
    else:
        moves = all_moves
    for move in moves:
        if move.uci in group.oracle_moves:
            move.flags.add("oracle")
        if move.uci in group.reference_moves:
            move.flags.add("reference")
        if move.uci in group.candidate_moves:
            move.flags.add("candidate")
    pieces = material_count(group.fen)
    tags = {f"source:{group.source}", phase_tag(pieces)}
    if any(move.gap_cp >= mate_gap_cp or abs(move.score_cp) >= mate_gap_cp for move in moves):
        tags.add("mate_like")
    else:
        tags.add("non_mate")
    policies = soft_policy((move.gap_cp for move in moves), policy_temperature_cp)
    move_rows = []
    for move, policy in zip(moves, policies or [0.0] * len(moves)):
        move_rows.append({
            "uci": move.uci,
            "rank": move.rank,
            "score_cp": move.score_cp,
            "gap_cp": move.gap_cp,
            "policy": round(policy, 8),
            "flags": sorted(move.flags),
        })
    return {
        "id": target_id(group.source, group.log, group.ply, group.fen),
        "source": group.source,
        "source_file": group.source_file,
        "log": group.log,
        "ply": group.ply,
        "fullmove": group.fullmove,
        "side": group.side,
        "fen": group.fen,
        "tags": sorted(tags),
        "material_count": pieces,
        "legal_move_count": len(group.moves),
        "scored_move_count": len(move_rows),
        "best_move": move_rows[0]["uci"] if move_rows else "",
        "oracle_moves": sorted(group.oracle_moves),
        "reference_moves": sorted(group.reference_moves),
        "candidate_moves": sorted(group.candidate_moves),
        "moves": move_rows,
    }


def summarize(rows: list[dict[str, object]]) -> list[str]:
    by_source: dict[str, int] = defaultdict(int)
    by_tag: dict[str, int] = defaultdict(int)
    move_rows = 0
    for row in rows:
        by_source[str(row["source"])] += 1
        move_rows += int(row["scored_move_count"])
        for tag in row.get("tags", []):
            by_tag[str(tag)] += 1
    return [
        f"targets={len(rows)}",
        f"moves={move_rows}",
        "sources=" + ",".join(f"{k}:{v}" for k, v in sorted(by_source.items())),
        "tags=" + ",".join(f"{k}:{v}" for k, v in sorted(by_tag.items())),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build unified search-aware JSONL targets.")
    parser.add_argument("--scores", action="append", required=True, type=parse_source)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--dedupe-fen", action="store_true")
    parser.add_argument("--max-moves", type=int, default=0)
    parser.add_argument("--policy-temperature-cp", type=float, default=200.0)
    parser.add_argument("--mate-gap-cp", type=int, default=MATE_SCORE_CP)
    args = parser.parse_args()

    groups = list(load_groups(args.scores).values())
    groups.sort(key=lambda group: (group.source, group.log, int(group.ply or 0), group.fen))
    if args.dedupe_fen:
        groups = dedupe_groups(groups)
    rows = [row_for_group(group, max_moves=args.max_moves,
                          policy_temperature_cp=args.policy_temperature_cp,
                          mate_gap_cp=args.mate_gap_cp) for group in groups]
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    lines = summarize(rows)
    print("\n".join(lines))
    if args.summary:
        summary = args.summary.expanduser()
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
