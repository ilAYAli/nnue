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
    split_policy_groups,
)


def load_ranker(path: str, device: str):
    checkpoint = torch.load(path, map_location=device)
    model = PolicyRanker(
        input_dim=int(checkpoint["input_dim"]),
        hidden=int(checkpoint["hidden"]),
        dropout=float(checkpoint.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def parse_thresholds(raw: str) -> list[float]:
    return [float(item) for item in raw.split(",") if item.strip()]


def parse_tags(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


@torch.no_grad()
def build_rows(model: PolicyRanker, groups, mean: torch.Tensor,
               std: torch.Tensor, device: str):
    rows = []
    for group in groups:
        policy_scores = score_group(model, group, mean, std, device)
        policy_idx = int(torch.argmax(policy_scores).item())
        base_idx = group.base_idx
        confidence = float(policy_scores[policy_idx] - policy_scores[base_idx])
        rows.append({
            "group": group,
            "policy_idx": policy_idx,
            "base_idx": base_idx,
            "confidence": confidence,
        })
    return rows


def summarize_rows(label: str, rows, thresholds: list[float],
                   bad_tolerance_cp: float
                   ) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    base_top1 = sum(1 for row in rows if row["base_idx"] == row["group"].best_idx)
    policy_top1 = sum(1 for row in rows if row["policy_idx"] == row["group"].best_idx)
    base_sum_gap = sum(float(row["group"].gaps[row["base_idx"]]) for row in rows)
    policy_sum_gap = sum(float(row["group"].gaps[row["policy_idx"]]) for row in rows)
    print(
        f"{label} base top1={base_top1}/{len(rows)} sum_gap={base_sum_gap:.0f}",
        flush=True,
    )
    print(
        f"{label} policy top1={policy_top1}/{len(rows)} sum_gap={policy_sum_gap:.0f}",
        flush=True,
    )

    for threshold in thresholds:
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
            if diff > bad_tolerance_cp:
                good += 1
            elif diff < -bad_tolerance_cp:
                bad += 1
            else:
                same += 1
            worst_harm = min(worst_harm, diff)
            best_gain = max(best_gain, diff)
            selected_top1 += int(selected_idx == group.best_idx)
            selected_sum_gap += selected_gap

        summary = {
            "label": label,
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
            f"{label} threshold={threshold:g} selected_top1={selected_top1}/{len(rows)} "
            f"sum_gap={selected_sum_gap:.0f} overrides={overrides} "
            f"good={good} bad={bad} same={same} "
            f"worst_harm={worst_harm:.0f} best_gain={best_gain:.0f}",
            flush=True,
        )

    return out


def summarize_breakdowns(label: str, rows, thresholds: list[float],
                         tags: list[str], bad_tolerance_cp: float) -> None:
    for tag in tags:
        tagged_rows = [
            row for row in rows
            if tag in set(row["group"].tags)
        ]
        if not tagged_rows:
            continue
        summarize_rows(
            f"{label}:{tag}", tagged_rows, thresholds, bad_tolerance_cp)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, list[dict[str, float]]]:
    model, checkpoint = load_ranker(args.model, args.device)
    feature_set = args.feature_set
    if feature_set == "auto":
        feature_set = str(checkpoint.get("feature_set", "compact"))
    raw_groups = load_policy_source_groups(
        args.targets,
        min_groups=args.min_groups,
        include_tags=args.include_tags,
        exclude_tags=args.exclude_tags)
    builder = PolicyFeatureBuilder(
        args.base_net, device=args.device, feature_set=feature_set)
    groups = [builder.build_group(group) for group in raw_groups]
    mean = checkpoint["mean"].to(args.device)
    std = checkpoint["std"].to(args.device)
    thresholds = parse_thresholds(args.thresholds)
    breakdown_tags = parse_tags(args.breakdown_tags)

    all_rows = build_rows(model, groups, mean, std, args.device)
    out = {
        "all": summarize_rows(
            "all", all_rows, thresholds, args.bad_tolerance_cp),
    }
    summarize_breakdowns(
        "all", all_rows, thresholds, breakdown_tags, args.bad_tolerance_cp)
    if args.split_val_fraction > 0.0:
        train_groups, val_groups = split_policy_groups(
            groups, args.split_seed, args.split_val_fraction)
        train_rows = build_rows(model, train_groups, mean, std, args.device)
        val_rows = build_rows(model, val_groups, mean, std, args.device)
        out["train"] = summarize_rows(
            "train", train_rows, thresholds, args.bad_tolerance_cp)
        summarize_breakdowns(
            "train", train_rows, thresholds, breakdown_tags,
            args.bad_tolerance_cp)
        out["val"] = summarize_rows(
            "val", val_rows, thresholds, args.bad_tolerance_cp)
        summarize_breakdowns(
            "val", val_rows, thresholds, breakdown_tags,
            args.bad_tolerance_cp)
    return out


def best_summary(summaries: list[dict[str, float]]) -> dict[str, float]:
    return max(
        summaries,
        key=lambda item: (
            -item["bad"],
            item["top1"],
            -item["sum_gap"],
            item["overrides"],
        ),
    )


def summary_passes(summary: dict[str, float], *,
                   min_top1: int = -1, max_bad: int = -1,
                   min_good: int = -1, min_overrides: int = -1,
                   max_overrides: int = -1) -> bool:
    if min_top1 >= 0 and summary["top1"] < min_top1:
        return False
    if max_bad >= 0 and summary["bad"] > max_bad:
        return False
    if min_good >= 0 and summary["good"] < min_good:
        return False
    if min_overrides >= 0 and summary["overrides"] < min_overrides:
        return False
    if max_overrides >= 0 and summary["overrides"] > max_overrides:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-net", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--feature-set", choices=["auto", "compact", "board"],
                    default="auto")
    ap.add_argument("--min-groups", type=int, default=1)
    ap.add_argument("--thresholds", default="0,1,2,4,8")
    ap.add_argument("--include-tags", default="")
    ap.add_argument("--exclude-tags", default="")
    ap.add_argument("--breakdown-tags", default="")
    ap.add_argument("--bad-tolerance-cp", type=float, default=0.0)
    ap.add_argument("--split-seed", type=int, default=1)
    ap.add_argument("--split-val-fraction", type=float, default=0.0)
    ap.add_argument("--fail-if-top1-below", type=int, default=-1)
    ap.add_argument("--fail-if-bad-above", type=int, default=-1)
    ap.add_argument("--fail-if-overrides-above", type=int, default=-1)
    ap.add_argument("--fail-if-val-top1-below", type=int, default=-1)
    ap.add_argument("--fail-if-val-bad-above", type=int, default=-1)
    ap.add_argument("--fail-if-val-good-below", type=int, default=-1)
    ap.add_argument("--fail-if-val-overrides-below", type=int, default=-1)
    args = ap.parse_args()

    summaries = evaluate(args)
    val_constraints = (
        args.fail_if_val_top1_below >= 0
        or args.fail_if_val_bad_above >= 0
        or args.fail_if_val_good_below >= 0
        or args.fail_if_val_overrides_below >= 0
    )
    if val_constraints and "val" in summaries:
        val_by_threshold = {
            item["threshold"]: item for item in summaries["val"]
        }
        failed = True
        for all_summary in summaries["all"]:
            val_summary = val_by_threshold[all_summary["threshold"]]
            if not summary_passes(
                    all_summary,
                    min_top1=args.fail_if_top1_below,
                    max_bad=args.fail_if_bad_above,
                    max_overrides=args.fail_if_overrides_above):
                continue
            if summary_passes(
                    val_summary,
                    min_top1=args.fail_if_val_top1_below,
                    max_bad=args.fail_if_val_bad_above,
                    min_good=args.fail_if_val_good_below,
                    min_overrides=args.fail_if_val_overrides_below):
                failed = False
                break
    else:
        best = best_summary(summaries["all"])
        failed = not summary_passes(
            best,
            min_top1=args.fail_if_top1_below,
            max_bad=args.fail_if_bad_above,
            max_overrides=args.fail_if_overrides_above)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
