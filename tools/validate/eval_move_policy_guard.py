#!/usr/bin/env python3
"""Evaluate sidecar move-policy false positives on no-override guards."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.move_policy import MovePolicy, move_features  # noqa: E402


def load_model(path: str, device: str) -> tuple[MovePolicy, torch.Tensor, torch.Tensor, dict]:
    checkpoint = torch.load(path, map_location=device)
    model = MovePolicy(
        input_dim=int(checkpoint["input_dim"]),
        hidden=int(checkpoint["hidden"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return (
        model,
        checkpoint["mean"].to(device),
        checkpoint["std"].to(device),
        checkpoint,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as src:
        for line in src:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{path}: no guard cases")
    return rows


def score_features(
    model: MovePolicy,
    features: list[list[float]],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: str,
) -> torch.Tensor:
    x = torch.tensor(features, dtype=torch.float32, device=device)
    x = (x - mean.to(device)) / std.to(device)
    with torch.no_grad():
        return model(x).cpu()


def evaluate(
    rows: list[dict[str, Any]],
    model: MovePolicy,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: str,
    feature_set: str,
    thresholds: list[float],
    bad_tolerance_cp: float,
    good_tolerance_cp: float,
) -> dict[str, Any]:
    features: list[list[float]] = []
    groups: list[tuple[int, int]] = []
    for row in rows:
        start = len(features)
        fen = str(row["fen"])
        features.append(move_features(fen, str(row["current"]), feature_set))
        for alt in row.get("alternatives", []):
            features.append(move_features(fen, str(alt["move"]), feature_set))
        groups.append((start, len(features)))

    scores = score_features(model, features, mean, std, device)
    threshold_summaries: list[dict[str, Any]] = []
    by_source: dict[float, Counter[str]] = {threshold: Counter() for threshold in thresholds}
    for threshold in thresholds:
        overrides = 0
        harmful = 0
        helpful = 0
        neutral = 0
        max_margins: list[float] = []
        for row, (start, end) in zip(rows, groups):
            if end <= start + 1:
                continue
            current_score = scores[start]
            alt_scores = scores[start + 1:end]
            best_alt_index = int(torch.argmax(alt_scores).item())
            margin = float((alt_scores[best_alt_index] - current_score).item())
            max_margins.append(margin)
            if margin < threshold:
                continue
            overrides += 1
            source = str(row.get("source_label", row.get("source", "unknown")))
            by_source[threshold][source] += 1
            alt = row["alternatives"][best_alt_index]
            delta_cp = float(alt.get("delta_cp", 0.0))
            if delta_cp < -bad_tolerance_cp:
                harmful += 1
            elif delta_cp > good_tolerance_cp:
                helpful += 1
            else:
                neutral += 1
        threshold_summaries.append({
            "threshold": threshold,
            "cases": len(rows),
            "overrides": overrides,
            "harmful": harmful,
            "helpful": helpful,
            "neutral": neutral,
            "avg_max_margin": sum(max_margins) / max(1, len(max_margins)),
            "by_source": dict(sorted(by_source[threshold].items())),
        })
    return {
        "cases": len(rows),
        "thresholds": threshold_summaries,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"guard cases={summary['cases']}", flush=True)
    for item in summary["thresholds"]:
        print(
            f"threshold={item['threshold']:g} "
            f"overrides={item['overrides']}/{item['cases']} "
            f"helpful={item['helpful']} neutral={item['neutral']} "
            f"harmful={item['harmful']} "
            f"avg_max_margin={item['avg_max_margin']:.3f} "
            f"by_source={item['by_source']}",
            flush=True,
        )


def parse_thresholds(raw: str) -> list[float]:
    return [float(item) for item in raw.split(",") if item.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guards", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--feature-set", choices=["auto", "compact", "board"],
                    default="auto")
    ap.add_argument("--thresholds", default="0,1,2,4,8,16")
    ap.add_argument("--bad-tolerance-cp", type=float, default=10.0)
    ap.add_argument("--good-tolerance-cp", type=float, default=10.0)
    ap.add_argument("--summary-json")
    ap.add_argument("--fail-if-harmful-above", type=int, default=-1)
    args = ap.parse_args()

    rows = read_jsonl(Path(args.guards))
    model, mean, std, checkpoint = load_model(args.model, args.device)
    feature_set = args.feature_set
    if feature_set == "auto":
        feature_set = str(checkpoint.get("feature_set", "compact"))
    summary = evaluate(
        rows,
        model,
        mean,
        std,
        args.device,
        feature_set,
        parse_thresholds(args.thresholds),
        args.bad_tolerance_cp,
        args.good_tolerance_cp,
    )
    summary["model_config"] = checkpoint.get("config", {})
    print_summary(summary)
    if args.summary_json:
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    if args.fail_if_harmful_above >= 0:
        worst = max(item["harmful"] for item in summary["thresholds"])
        if worst > args.fail_if_harmful_above:
            raise SystemExit(
                f"harmful {worst} > {args.fail_if_harmful_above}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
