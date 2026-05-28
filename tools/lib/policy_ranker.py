"""Sidecar move-ranking features and model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import chess
import torch
import torch.nn as nn

from . import enyo_nnue as nn2
from .child_rank_targets import (
    ChildRankGroup,
    ChildMoveTarget,
    child_fen,
    load_groups,
    target_gap_cp,
)
from .nnue_model import EnyoNNUE, load_model_from_nn


@dataclass
class PolicyGroup:
    group_id: str
    moves: tuple[str, ...]
    features: torch.Tensor
    oracle_scores: torch.Tensor
    best_idx: int
    base_idx: int
    gaps: torch.Tensor


def _parse_tags(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def load_policy_source_groups(paths: list[str | Path], *,
                              min_groups: int = 1,
                              include_tags: str = "",
                              exclude_tags: str = "") -> list[ChildRankGroup]:
    groups: list[ChildRankGroup] = []
    seen: set[str] = set()
    include = _parse_tags(include_tags)
    exclude = _parse_tags(exclude_tags)
    for path in paths:
        for group in load_groups(path, min_groups=0, skip_invalid=True):
            tags = set(group.tags)
            if include and not include.issubset(tags):
                continue
            if exclude and exclude.intersection(tags):
                continue
            if group.group_id in seen:
                continue
            seen.add(group.group_id)
            groups.append(group)

    if len(groups) < min_groups:
        raise SystemExit(
            f"valid_groups={len(groups)} below min_groups={min_groups}")
    return groups


def split_policy_groups(groups: list[PolicyGroup], seed: int,
                        val_fraction: float
                        ) -> tuple[list[PolicyGroup], list[PolicyGroup]]:
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    val_n = int(round(len(shuffled) * val_fraction))
    if val_fraction > 0.0:
        val_n = max(1, min(len(shuffled) - 1, val_n))
    return shuffled[val_n:], shuffled[:val_n]


class PolicyRanker(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128,
                 dropout: float = 0.0) -> None:
        super().__init__()
        drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            drop,
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            drop,
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def fen_features(fen: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pieces, stm = nn2.parse_fen(fen)
    pieces.sort(key=lambda item: item[2])
    return (
        torch.tensor(nn2.features_from_pieces(pieces, nn2.WHITE), dtype=torch.long),
        torch.tensor(nn2.features_from_pieces(pieces, nn2.BLACK), dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([stm], dtype=torch.long),
        torch.tensor([nn2.phase_scale_from_pieces(pieces)], dtype=torch.float32),
    )


@torch.no_grad()
def eval_fen(model: EnyoNNUE, fen: str, device: str) -> float:
    w, b, w_off, b_off, stm, phase = fen_features(fen)
    return float(model(
        w.to(device), b.to(device), w_off.to(device), b_off.to(device),
        stm.to(device), phase.to(device)).item())


def _one_hot(index: int, size: int) -> list[float]:
    out = [0.0] * size
    if 0 <= index < size:
        out[index] = 1.0
    return out


def _piece_index(piece_type: int | None) -> int:
    return 0 if piece_type is None else piece_type


def _material_features(board: chess.Board) -> list[float]:
    features: list[float] = []
    for color in (chess.WHITE, chess.BLACK):
        for piece_type in (
            chess.PAWN,
            chess.KNIGHT,
            chess.BISHOP,
            chess.ROOK,
            chess.QUEEN,
            chess.KING,
        ):
            features.append(len(board.pieces(piece_type, color)) / 8.0)
    return features


def _oriented_square(square: chess.Square, view: chess.Color,
                     king_square: chess.Square | None) -> chess.Square:
    out = square
    king = king_square
    if view == chess.BLACK:
        out ^= 56
        if king is not None:
            king ^= 56
    if king is not None and (king & 0x4):
        out ^= 7
    return out


def _board_planes(board: chess.Board, view: chess.Color,
                  king_square: chess.Square | None) -> list[float]:
    features = [0.0] * (12 * 64)
    for square, piece in board.piece_map().items():
        rel_color = 0 if piece.color == view else 1
        plane = rel_color * 6 + (piece.piece_type - 1)
        oriented = _oriented_square(square, view, king_square)
        features[plane * 64 + oriented] = 1.0
    return features


def move_features(board: chess.Board, move: chess.Move, base_score: float,
                  base_best: float, base_rank: int, move_count: int,
                  feature_set: str = "compact") -> list[float]:
    moving = board.piece_at(move.from_square)
    captured = board.piece_at(move.to_square)
    if board.is_en_passant(move):
        captured_type = chess.PAWN
    else:
        captured_type = captured.piece_type if captured else None

    try:
        gives_check = board.gives_check(move)
    except AssertionError:
        gives_check = False

    features: list[float] = [
        base_score / 800.0,
        (base_score - base_best) / 800.0,
        base_rank / max(1.0, float(move_count - 1)),
        1.0 if board.turn == chess.WHITE else -1.0,
        float(move_count) / 80.0,
        1.0 if board.is_check() else 0.0,
        1.0 if board.is_capture(move) else 0.0,
        1.0 if gives_check else 0.0,
        1.0 if move.promotion else 0.0,
        1.0 if board.is_castling(move) else 0.0,
        1.0 if board.is_en_passant(move) else 0.0,
    ]
    features.extend(_one_hot(move.from_square, 64))
    features.extend(_one_hot(move.to_square, 64))
    features.extend(_one_hot((moving.piece_type - 1) if moving else -1, 6))
    features.extend(_one_hot(_piece_index(captured_type), 7))
    promo_index = 0 if move.promotion is None else move.promotion
    features.extend(_one_hot(promo_index, 7))
    features.extend(_material_features(board))
    if feature_set == "board":
        view = board.turn
        king_square = board.king(view)
        child = board.copy(stack=False)
        child.push(move)
        features.extend(_board_planes(board, view, king_square))
        features.extend(_board_planes(child, view, king_square))
    elif feature_set != "compact":
        raise ValueError(f"unknown policy feature set: {feature_set}")
    return features


class PolicyFeatureBuilder:
    def __init__(self, base_net: str | Path, device: str = "cpu",
                 feature_set: str = "compact") -> None:
        self.device = device
        self.feature_set = feature_set
        self.base_model = load_model_from_nn(base_net, device=device)
        self.base_model.eval()

    def build_group(self, group: ChildRankGroup) -> PolicyGroup:
        board = chess.Board(group.fen)
        base_scores: dict[str, float] = {}
        for move in group.moves:
            # Base score is parent POV child eval from the frozen reference net.
            base_scores[move.move] = -eval_fen(
                self.base_model, child_fen(group.fen, move.move), self.device)

        ranked = sorted(base_scores, key=base_scores.get, reverse=True)
        base_rank = {move: idx for idx, move in enumerate(ranked)}
        base_best = base_scores[ranked[0]]

        rows: list[list[float]] = []
        oracle_scores: list[float] = []
        gaps: list[float] = []
        moves: list[str] = []
        best_idx = -1
        base_idx = -1
        for idx, target in enumerate(group.moves):
            move = chess.Move.from_uci(target.move)
            rows.append(move_features(
                board, move, base_scores[target.move], base_best,
                base_rank[target.move], len(group.moves), self.feature_set))
            oracle_scores.append(target.score_cp)
            gaps.append(target_gap_cp(group, target))
            moves.append(target.move)
            if target.move == group.best_move:
                best_idx = idx
            if target.move == ranked[0]:
                base_idx = idx

        if best_idx < 0 or base_idx < 0:
            raise ValueError(f"{group.group_id}: missing best/base move")

        return PolicyGroup(
            group_id=group.group_id,
            moves=tuple(moves),
            features=torch.tensor(rows, dtype=torch.float32),
            oracle_scores=torch.tensor(oracle_scores, dtype=torch.float32),
            best_idx=best_idx,
            base_idx=base_idx,
            gaps=torch.tensor(gaps, dtype=torch.float32),
        )


def normalize_features(groups: list[PolicyGroup]) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.cat([group.features for group in groups], dim=0)
    mean = rows.mean(dim=0)
    std = rows.std(dim=0)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


def score_group(model: PolicyRanker, group: PolicyGroup,
                mean: torch.Tensor, std: torch.Tensor,
                device: str) -> torch.Tensor:
    x = ((group.features.to(device) - mean.to(device)) / std.to(device))
    return model(x).detach().cpu()
