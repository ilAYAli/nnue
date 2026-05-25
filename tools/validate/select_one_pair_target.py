#!/usr/bin/env python3
"""Select one high-signal search-aware target for exported overfit diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.expanduser().open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def required_tags(text: str) -> set[str]:
    return {tag.strip() for tag in text.split(",") if tag.strip()}


def move_uci(move: dict[str, Any]) -> str:
    return str(move.get("uci", "")).lower()


def move_gap(move: dict[str, Any]) -> float:
    return float(move.get("gap_cp", 0.0))


def move_rank(move: dict[str, Any]) -> int:
    return int(move.get("rank", 999999))


def row_score(row: dict[str, Any]) -> tuple[float, float, int]:
    moves = [move for move in row.get("moves", []) if move_uci(move)]
    candidate_moves = {str(move).lower() for move in row.get("candidate_moves", [])}
    candidate_gap = max(
        (move_gap(move) for move in moves if move_uci(move) in candidate_moves),
        default=-1.0,
    )
    worst_gap = max((move_gap(move) for move in moves), default=-1.0)
    return candidate_gap, worst_gap, len(moves)


def select_row(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    tags = required_tags(args.required_tags)
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if args.id and str(row.get("id", "")) != args.id:
            continue
        if tags and not tags.issubset({str(tag) for tag in row.get("tags", [])}):
            continue
        moves = [move for move in row.get("moves", []) if move_uci(move)]
        if len(moves) < args.min_moves:
            continue
        if not any(move_rank(move) == 1 for move in moves):
            continue
        if max((move_gap(move) for move in moves), default=0.0) < args.min_gap_cp:
            continue
        candidates.append(row)

    if not candidates:
        raise SystemExit("no target row matched selection criteria")
    if args.id:
        return candidates[0]
    return max(candidates, key=row_score)


def retag_move(move: dict[str, Any], extra_flags: set[str]) -> dict[str, Any]:
    out = dict(move)
    flags = {str(flag) for flag in out.get("flags", [])}
    flags.update(extra_flags)
    out["flags"] = sorted(flags)
    return out


def make_one_pair(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    moves = sorted(
        [move for move in row.get("moves", []) if move_uci(move)],
        key=move_rank,
    )
    best = next((move for move in moves if move_rank(move) == 1), moves[0])
    best_uci = move_uci(best)

    candidate_set = {str(move).lower() for move in row.get("candidate_moves", [])}
    bad = next(
        (move for move in moves if move_uci(move) in candidate_set and move_uci(move) != best_uci),
        None,
    )
    if bad is None:
        bad = max(
            (move for move in moves if move_uci(move) != best_uci),
            key=move_gap,
            default=None,
        )
    if bad is None:
        raise SystemExit("selected row has no non-best move")
    bad_uci = move_uci(bad)

    worst = max(
        (move for move in moves if move_uci(move) != best_uci),
        key=move_gap,
        default=bad,
    )
    worst_uci = move_uci(worst)

    selected: dict[str, dict[str, Any]] = {}
    selected[best_uci] = retag_move(best, {"diagnostic_best"})
    selected[bad_uci] = retag_move(bad, {"diagnostic_bad"})

    for move in moves:
        uci = move_uci(move)
        if uci in selected:
            continue
        selected[uci] = retag_move(move, {"diagnostic_neighbor"})
        if len(selected) >= args.neighbor_moves + 2:
            break

    if worst_uci not in selected:
        selected[worst_uci] = retag_move(worst, {"diagnostic_blunder"})
    else:
        selected[worst_uci] = retag_move(selected[worst_uci], {"diagnostic_blunder"})

    out_moves = sorted(selected.values(), key=move_rank)
    if args.hard_policy:
        for move in out_moves:
            move["policy"] = 1.0 if move_uci(move) == best_uci else 0.0

    out = dict(row)
    out["id"] = f"{row.get('id', 'target')}:onepair"
    out["moves"] = out_moves
    out["best_move"] = best_uci
    out["oracle_moves"] = [best_uci]
    out["candidate_moves"] = [bad_uci]
    out["reference_moves"] = [best_uci] if best_uci in {
        str(move).lower() for move in row.get("reference_moves", [])
    } else []
    out["scored_move_count"] = len(out_moves)
    out["diagnostic"] = {
        "source_id": row.get("id", ""),
        "bad_move": bad_uci,
        "bad_gap_cp": move_gap(bad),
        "worst_move": worst_uci,
        "worst_gap_cp": move_gap(worst),
        "selected_move_count": len(out_moves),
    }
    tags = [str(tag) for tag in out.get("tags", [])]
    if "diagnostic:one_pair" not in tags:
        tags.append("diagnostic:one_pair")
    out["tags"] = tags
    return out


def write_summary(path: Path, row: dict[str, Any]) -> None:
    lines = [
        f"id={row.get('id', '')}",
        f"fen={row.get('fen', '')}",
        f"best={row.get('best_move', '')}",
        f"candidate={','.join(row.get('candidate_moves', []))}",
        f"tags={','.join(row.get('tags', []))}",
        "moves:",
    ]
    for move in row.get("moves", []):
        flags = ",".join(str(flag) for flag in move.get("flags", []))
        lines.append(
            f"  {move_uci(move)} rank={move_rank(move)} "
            f"gap={move_gap(move):.0f} policy={float(move.get('policy', 0.0)):.6f} "
            f"flags={flags}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True, help="Input search-aware JSONL")
    parser.add_argument("--out", required=True, help="Output one-row JSONL")
    parser.add_argument("--summary", default="", help="Optional summary path")
    parser.add_argument("--id", default="", help="Select a specific source target id")
    parser.add_argument("--required-tags", default="mate_like")
    parser.add_argument("--min-gap-cp", type=float, default=300.0)
    parser.add_argument("--min-moves", type=int, default=3)
    parser.add_argument("--neighbor-moves", type=int, default=4)
    parser.add_argument("--hard-policy", action="store_true",
                        help="Set policy to 1.0 for best move and 0.0 otherwise")
    args = parser.parse_args()

    row = make_one_pair(select_row(read_rows(Path(args.targets)), args), args)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    if args.summary:
        write_summary(Path(args.summary).expanduser(), row)
    else:
        write_summary(out.with_suffix(".summary.txt"), row)
    print(f"wrote {out}")
    print(
        f"best={row.get('best_move')} bad={','.join(row.get('candidate_moves', []))} "
        f"moves={len(row.get('moves', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
