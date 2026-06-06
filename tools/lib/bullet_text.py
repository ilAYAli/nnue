"""Helpers for Bullet text import rows."""
from __future__ import annotations


def phase_scale_from_fen(fen: str) -> float:
    board = fen.split()[0]
    minors = sum(1 for ch in board if ch in "NnBb")
    rooks = sum(1 for ch in board if ch in "Rr")
    queens = sum(1 for ch in board if ch in "Qq")
    phase = 3 * minors + 5 * rooks + 10 * queens
    return (128.0 + float(phase)) / 128.0


def categorical_result(value: float) -> float:
    if value > 0.5:
        return 1.0
    if value < 0.5:
        return 0.0
    return 0.5


def white_result_from_row(row: dict) -> float:
    result = row.get("result")
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return 0.0
    if result == "1/2-1/2":
        return 0.5

    wdl = float(row.get("wdl", 0.5))
    stm = str(row["fen"]).split()[1]
    white_wdl = wdl if stm == "w" else 1.0 - wdl
    return categorical_result(white_wdl)


def white_score_from_row(row: dict, *, enyo_runtime_target: bool) -> int:
    fen = str(row["fen"])
    stm = fen.split()[1]
    score = int(round(float(row["score"])))
    if enyo_runtime_target:
        score = int(round(score / phase_scale_from_fen(fen)))
    return score if stm == "w" else -score


def row_to_text(row: dict, *, enyo_runtime_target: bool) -> str:
    fen = str(row["fen"])
    score = white_score_from_row(row, enyo_runtime_target=enyo_runtime_target)
    result = white_result_from_row(row)
    return f"{fen} | {score} | {result:.1f}"
