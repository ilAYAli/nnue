"""Tiny sidecar move-policy model for fixed move-choice gates."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import chess
import torch
import torch.nn as nn


FEATURE_VERSION = "move_policy_v1"


class MovePolicy(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _one_hot(index: int, size: int) -> list[float]:
    out = [0.0] * size
    if 0 <= index < size:
        out[index] = 1.0
    return out


def _material(board: chess.Board) -> list[float]:
    out: list[float] = []
    for color in (chess.WHITE, chess.BLACK):
        for piece_type in (
            chess.PAWN,
            chess.KNIGHT,
            chess.BISHOP,
            chess.ROOK,
            chess.QUEEN,
            chess.KING,
        ):
            out.append(len(board.pieces(piece_type, color)) / 8.0)
    return out


def _piece_index(piece: chess.Piece | None) -> int:
    return -1 if piece is None else piece.piece_type - 1


def _capture_index(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        return chess.PAWN - 1
    captured = board.piece_at(move.to_square)
    return _piece_index(captured)


def _oriented_square(square: chess.Square, view: chess.Color) -> chess.Square:
    if view == chess.BLACK:
        return square ^ 56
    return square


def _board_planes(board: chess.Board, view: chess.Color) -> list[float]:
    features = [0.0] * (12 * 64)
    for square, piece in board.piece_map().items():
        rel_color = 0 if piece.color == view else 1
        plane = rel_color * 6 + (piece.piece_type - 1)
        oriented = _oriented_square(square, view)
        features[plane * 64 + oriented] = 1.0
    return features


def move_features(fen: str, move_uci: str, feature_set: str = "compact") -> list[float]:
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        raise ValueError(f"illegal move {move_uci} in {fen}")
    moving = board.piece_at(move.from_square)
    try:
        gives_check = board.gives_check(move)
    except AssertionError:
        gives_check = False
    legal_count = board.legal_moves.count()
    child = board.copy(stack=False)
    child.push(move)

    features: list[float] = [
        1.0 if board.turn == chess.WHITE else -1.0,
        float(legal_count) / 80.0,
        min(float(board.fullmove_number), 120.0) / 120.0,
        float(board.halfmove_clock) / 100.0,
        1.0 if board.is_check() else 0.0,
        1.0 if board.is_capture(move) else 0.0,
        1.0 if gives_check else 0.0,
        1.0 if board.is_castling(move) else 0.0,
        1.0 if board.is_en_passant(move) else 0.0,
        1.0 if move.promotion else 0.0,
        float(chess.square_file(move.to_square) - chess.square_file(move.from_square)) / 7.0,
        float(chess.square_rank(move.to_square) - chess.square_rank(move.from_square)) / 7.0,
        float(child.legal_moves.count()) / 80.0,
        1.0 if child.is_check() else 0.0,
    ]
    features.extend(_one_hot(move.from_square, 64))
    features.extend(_one_hot(move.to_square, 64))
    features.extend(_one_hot(_piece_index(moving), 6))
    features.extend(_one_hot(_capture_index(board, move), 6))
    promo_index = -1 if move.promotion is None else move.promotion - 1
    features.extend(_one_hot(promo_index, 6))
    features.extend(_material(board))
    if feature_set == "board":
        view = board.turn
        features.extend(_board_planes(board, view))
        features.extend(_board_planes(child, view))
    elif feature_set != "compact":
        raise ValueError(f"unknown move-policy feature set: {feature_set}")
    return features


def load_gate_cases(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if "fen" in row and "best" in row and "played" in row:
                    rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no move-gate cases")
    return rows


def pair_tensors(
    rows: list[dict[str, Any]],
    feature_set: str = "compact",
) -> tuple[torch.Tensor, torch.Tensor]:
    best_rows: list[list[float]] = []
    played_rows: list[list[float]] = []
    for row in rows:
        fen = str(row["fen"])
        best_rows.append(move_features(fen, str(row["best"]), feature_set))
        played_rows.append(move_features(fen, str(row["played"]), feature_set))
    return (
        torch.tensor(best_rows, dtype=torch.float32),
        torch.tensor(played_rows, dtype=torch.float32),
    )


def normalize(best: torch.Tensor, played: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    all_rows = torch.cat([best, played], dim=0)
    mean = all_rows.mean(dim=0)
    std = all_rows.std(dim=0).clamp_min(1e-6)
    return mean, std


def score_margins(model: MovePolicy, best: torch.Tensor, played: torch.Tensor,
                  mean: torch.Tensor, std: torch.Tensor,
                  device: str) -> torch.Tensor:
    x_best = ((best.to(device) - mean.to(device)) / std.to(device))
    x_played = ((played.to(device) - mean.to(device)) / std.to(device))
    return model(x_best) - model(x_played)


def source_breakdown(rows: list[dict[str, Any]], correct: torch.Tensor) -> dict[str, str]:
    counts: dict[str, Counter[str]] = {}
    for row, ok in zip(rows, correct.tolist()):
        source = str(row.get("source_label", row.get("source", "unknown")))
        counts.setdefault(source, Counter())
        counts[source]["rows"] += 1
        counts[source]["correct"] += int(bool(ok))
    return {
        source: f"{counter['correct']}/{counter['rows']}"
        for source, counter in sorted(counts.items())
    }
