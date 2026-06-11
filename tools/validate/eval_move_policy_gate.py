#!/usr/bin/env python3
"""Evaluate a sidecar move policy on fixed move-gate cases."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.move_policy import (  # noqa: E402
    MovePolicy,
    load_gate_cases,
    pair_tensors,
    score_margins,
    source_breakdown,
)


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


def threshold_rows(rows: list[dict], margins: torch.Tensor,
                   threshold: float) -> dict[str, float]:
    selected = margins >= threshold
    correct = margins > 0
    selected_correct = selected & correct
    selected_incorrect = selected & ~correct
    missed_correct = correct & ~selected
    correct_count = int(correct.sum().item())
    selected_count = int(selected.sum().item())
    selected_correct_count = int(selected_correct.sum().item())
    selected_incorrect_count = int(selected_incorrect.sum().item())
    missed_correct_count = int(missed_correct.sum().item())
    return {
        "threshold": threshold,
        "correct": float(correct_count),
        "selected": float(selected_count),
        "selected_correct": float(selected_correct_count),
        "selected_incorrect": float(selected_incorrect_count),
        "missed_correct": float(missed_correct_count),
        "cases": float(len(rows)),
        "avg_margin": float(margins.mean().item()) if rows else 0.0,
    }


def print_summary(label: str, rows: list[dict], margins: torch.Tensor,
                  thresholds: list[float]) -> list[dict[str, float]]:
    correct = margins > 0
    count = int(correct.sum().item())
    print(
        f"{label} correct={count}/{len(rows)} "
        f"({100.0 * count / max(1, len(rows)):.1f}%) "
        f"avg_margin={float(margins.mean().item()):.3f} "
        f"by_source={source_breakdown(rows, correct)}",
        flush=True,
    )
    summaries: list[dict[str, float]] = []
    for threshold in thresholds:
        item = threshold_rows(rows, margins, threshold)
        summaries.append(item)
        print(
            f"{label} threshold={threshold:g} "
            f"selected={int(item['selected'])}/{len(rows)} "
            f"selected_correct={int(item['selected_correct'])} "
            f"selected_incorrect={int(item['selected_incorrect'])} "
            f"missed_correct={int(item['missed_correct'])} "
            f"total_correct={int(item['correct'])}/{len(rows)}",
            flush=True,
        )
    return summaries


def parse_thresholds(raw: str) -> list[float]:
    return [float(item) for item in raw.split(",") if item.strip()]


def split_rows(rows: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    indexed = list(enumerate(rows))
    random.Random(seed).shuffle(indexed)
    val_n = int(round(len(rows) * val_fraction))
    if val_fraction > 0.0:
        val_n = max(1, min(len(rows) - 1, val_n))
    val = [row for _idx, row in indexed[:val_n]]
    train = [row for _idx, row in indexed[val_n:]]
    return train, val


def evaluate_rows(label: str, rows: list[dict], model: MovePolicy,
                  mean: torch.Tensor, std: torch.Tensor, device: str,
                  thresholds: list[float], feature_set: str) -> dict:
    best, played = pair_tensors(rows, feature_set)
    margins = score_margins(model, best, played, mean, std, device).cpu()
    summaries = print_summary(label, rows, margins, thresholds)
    return {
        "cases": len(rows),
        "correct": int((margins > 0).sum().item()),
        "avg_margin": float(margins.mean().item()) if rows else 0.0,
        "thresholds": summaries,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--feature-set", choices=["auto", "compact", "board"],
                    default="auto")
    ap.add_argument("--thresholds", default="0,1,2,4")
    ap.add_argument("--split-seed", type=int, default=1)
    ap.add_argument("--split-val-fraction", type=float, default=0.0)
    ap.add_argument("--summary-json")
    ap.add_argument("--fail-if-correct-below", type=int, default=-1)
    ap.add_argument("--fail-if-val-correct-below", type=int, default=-1)
    args = ap.parse_args()

    rows = load_gate_cases(args.cases)
    model, mean, std, checkpoint = load_model(args.model, args.device)
    feature_set = args.feature_set
    if feature_set == "auto":
        feature_set = str(checkpoint.get("feature_set", "compact"))
    thresholds = parse_thresholds(args.thresholds)
    all_summary = evaluate_rows(
        "all", rows, model, mean, std, args.device, thresholds, feature_set)
    split_summary = {}
    if args.split_val_fraction > 0.0:
        train_rows, val_rows = split_rows(
            rows, args.split_val_fraction, args.split_seed)
        split_summary["train"] = evaluate_rows(
            "train", train_rows, model, mean, std, args.device, thresholds,
            feature_set)
        split_summary["val"] = evaluate_rows(
            "val", val_rows, model, mean, std, args.device, thresholds,
            feature_set)

    payload = {
        "cases": len(rows),
        "correct": all_summary["correct"],
        "avg_margin": all_summary["avg_margin"],
        "thresholds": all_summary["thresholds"],
        "split": split_summary,
        "model_config": checkpoint.get("config", {}),
    }
    if args.summary_json:
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    if args.fail_if_correct_below >= 0 and payload["correct"] < args.fail_if_correct_below:
        raise SystemExit(
            f"correct {payload['correct']} < {args.fail_if_correct_below}")
    val_summary = split_summary.get("val")
    if (
        args.fail_if_val_correct_below >= 0
        and val_summary is not None
        and val_summary["correct"] < args.fail_if_val_correct_below
    ):
        raise SystemExit(
            f"val_correct {val_summary['correct']} < "
            f"{args.fail_if_val_correct_below}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
