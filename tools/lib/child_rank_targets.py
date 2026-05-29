"""Shared parser for child-move ranking target groups."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import chess


@dataclass(frozen=True)
class ChildMoveTarget:
    move: str
    score_cp: float
    role: str


@dataclass(frozen=True)
class ChildRankGroup:
    group_id: str
    fen: str
    max_gap_cp: float
    moves: tuple[ChildMoveTarget, ...]
    best_move: str
    base_move: str = ""
    tags: tuple[str, ...] = ()

    @property
    def best(self) -> ChildMoveTarget:
        for move in self.moves:
            if move.move == self.best_move:
                return move
        raise ValueError(f"{self.group_id}: best move missing: {self.best_move}")


def child_fen(fen: str, move_uci: str) -> str:
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        raise ValueError(f"illegal move {move_uci} in {fen}")
    board.push(move)
    return board.fen()


def _move_score(row: dict, group_id: str) -> float:
    if "score_cp" not in row:
        move = row.get("move") or row.get("uci")
        raise ValueError(f"{group_id}: move {move} missing score_cp")
    return float(row["score_cp"])


def _best_move(raw: dict, moves: list[ChildMoveTarget], group_id: str) -> str:
    if raw.get("best_move"):
        return str(raw["best_move"])

    role_best = [move.move for move in moves if move.role == "best"]
    if len(role_best) == 1:
        return role_best[0]
    if len(role_best) > 1:
        raise ValueError(f"{group_id}: multiple role=best moves")

    return max(moves, key=lambda move: move.score_cp).move


def parse_group(raw: dict, line_no: int) -> ChildRankGroup:
    group_id = str(raw.get("id") or raw.get("group_id") or f"line:{line_no}")
    fen = str(raw["fen"])
    max_gap_cp = float(raw.get("max_gap_cp", 800.0))
    if max_gap_cp <= 0:
        raise ValueError(f"{group_id}: max_gap_cp must be positive")

    board = chess.Board(fen)
    seen: set[str] = set()
    moves: list[ChildMoveTarget] = []
    for move_raw in raw.get("moves", []):
        move_uci = str(move_raw.get("move") or move_raw["uci"])
        if move_uci in seen:
            raise ValueError(f"{group_id}: duplicate move {move_uci}")
        seen.add(move_uci)
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            raise ValueError(f"{group_id}: illegal move {move_uci}")
        flags = set(move_raw.get("flags", []))
        moves.append(ChildMoveTarget(
            move=move_uci,
            score_cp=_move_score(move_raw, group_id),
            role=str(move_raw.get(
                "role", "best" if "best" in flags else "neighbor")),
        ))

    if len(moves) < 2:
        raise ValueError(f"{group_id}: need at least two scored moves")

    best_move = _best_move(raw, moves, group_id)
    if best_move not in seen:
        raise ValueError(f"{group_id}: best move {best_move} is not scored")
    base_move = str(raw.get("base_move", ""))
    if base_move and base_move not in seen:
        raise ValueError(f"{group_id}: base move {base_move} is not scored")

    return ChildRankGroup(
        group_id=group_id,
        fen=fen,
        max_gap_cp=max_gap_cp,
        moves=tuple(moves),
        best_move=best_move,
        base_move=base_move,
        tags=tuple(str(tag) for tag in raw.get("tags", [])),
    )


def load_groups(path: str | Path, *, min_groups: int = 1,
                skip_invalid: bool = False
                ) -> list[ChildRankGroup]:
    groups: list[ChildRankGroup] = []
    skipped = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                groups.append(parse_group(json.loads(line), line_no))
            except ValueError:
                if not skip_invalid:
                    raise
                skipped += 1

    if skipped:
        print(f"{path}: skipped_invalid_groups={skipped}", file=sys.stderr)

    if len(groups) < min_groups:
        raise SystemExit(
            f"valid_groups={len(groups)} below min_groups={min_groups}")
    return groups


def target_gap_cp(group: ChildRankGroup, move: ChildMoveTarget) -> float:
    best = group.best
    raw_gap = max(0.0, best.score_cp - move.score_cp)
    return min(raw_gap, group.max_gap_cp)
