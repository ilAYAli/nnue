"""Tiny sidecar move-policy model for fixed move-choice gates."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from lib.move_policy_features import move_features


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
