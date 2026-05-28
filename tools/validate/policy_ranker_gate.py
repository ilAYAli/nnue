#!/usr/bin/env python3
"""Offline gate for a sidecar child-move ranker."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.policy_ranker import (
    PolicyFeatureBuilder,
    PolicyRanker,
    load_policy_source_groups,
    score_group,
)


def load_ranker(path: str, device: str):
    checkpoint = torch.load(path, map_location=device)
    model = PolicyRanker(
        input_dim=int(checkpoint["input_dim"]),
        hidden=int(checkpoint["hidden"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint["mean"].to(device), checkpoint["std"].to(device)


def parse_thresholds(raw: str) -> list[float]:
    return [float(item) for item in raw.split(",") if item.strip()]


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> list[dict[str, float]]:
    raw_groups = load_policy_source_groups(args.targets, min_groups=args.min_groups)
    builder = PolicyFeatureBuilder(args.base_net, device=args.device)
    groups = [builder.build_group(group) for group in raw_groups]
    model, mean, std = load_ranker(args.model, args.device)

    rows = []
    for group in groups:
        policy_scores = score_group(model, group, mean, std, args.device)
        policy_idx = int(torch.argmax(policy_scores).item())
        base_idx = group.base_idx
        confidence = float(policy_scores[policy_idx] - policy_scores[base_idx])
        rows.append({
            "group": group,
            "policy_idx": policy_idx,
            "base_idx": base_idx,
            "confidence": confidence,
        })

    out: list[dict[str, float]] = []
    base_top1 = sum(1 for row in rows if row["base_idx"] == row["group"].best_idx)
    policy_top1 = sum(1 for row in rows if row["policy_idx"] == row["group"].best_idx)
    base_sum_gap = sum(float(row["group"].gaps[row["base_idx"]]) for row in rows)
    policy_sum_gap = sum(float(row["group"].gaps[row["policy_idx"]]) for row in rows)
    print(
        f"base top1={base_top1}/{len(rows)} sum_gap={base_sum_gap:.0f}",
        flush=True,
    )
    print(
        f"policy top1={policy_top1}/{len(rows)} sum_gap={policy_sum_gap:.0f}",
        flush=True,
    )

    for threshold in parse_thresholds(args.thresholds):
        selected_top1 = 0
        selected_sum_gap = 0.0
        overrides = 0
        good = 0
        bad = 0
        same = 0
        worst_harm = 0.0
        best_gain = 0.0
        for row in rows:
            group = row["group"]
            base_idx = int(row["base_idx"])
            policy_idx = int(row["policy_idx"])
            selected_idx = base_idx
            if policy_idx != base_idx and row["confidence"] >= threshold:
                selected_idx = policy_idx
                overrides += 1

            base_gap = float(group.gaps[base_idx])
            selected_gap = float(group.gaps[selected_idx])
            diff = base_gap - selected_gap
            if diff > 0:
                good += 1
            elif diff < 0:
                bad += 1
            else:
                same += 1
            worst_harm = min(worst_harm, diff)
            best_gain = max(best_gain, diff)
            selected_top1 += int(selected_idx == group.best_idx)
            selected_sum_gap += selected_gap

        summary = {
            "threshold": threshold,
            "top1": selected_top1,
            "groups": len(rows),
            "sum_gap": selected_sum_gap,
            "overrides": overrides,
            "good": good,
            "bad": bad,
            "same": same,
            "worst_harm": worst_harm,
            "best_gain": best_gain,
        }
        out.append(summary)
        print(
            f"threshold={threshold:g} selected_top1={selected_top1}/{len(rows)} "
            f"sum_gap={selected_sum_gap:.0f} overrides={overrides} "
            f"good={good} bad={bad} same={same} "
            f"worst_harm={worst_harm:.0f} best_gain={best_gain:.0f}",
            flush=True,
        )

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-net", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-groups", type=int, default=1)
    ap.add_argument("--thresholds", default="0,1,2,4,8")
    ap.add_argument("--fail-if-top1-below", type=int, default=-1)
    ap.add_argument("--fail-if-bad-above", type=int, default=-1)
    args = ap.parse_args()

    summaries = evaluate(args)
    best = max(
        summaries,
        key=lambda item: (
            -item["bad"],
            item["top1"],
            -item["sum_gap"],
            item["overrides"],
        ),
    )
    failed = False
    if args.fail_if_top1_below >= 0 and best["top1"] < args.fail_if_top1_below:
        failed = True
    if args.fail_if_bad_above >= 0 and best["bad"] > args.fail_if_bad_above:
        failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
