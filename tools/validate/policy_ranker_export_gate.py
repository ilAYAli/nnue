#!/usr/bin/env python3
"""Validate exported policy ranker parity against its PyTorch checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.policy_ranker import (
    PolicyFeatureBuilder,
    PolicyRanker,
    load_policy_ranker_export,
    load_policy_source_groups,
    score_group,
    score_group_export,
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


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--base-net", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--feature-set", choices=["auto", "compact", "board"],
                    default="auto")
    ap.add_argument("--include-tags", default="")
    ap.add_argument("--exclude-tags", default="")
    ap.add_argument("--min-groups", type=int, default=1)
    ap.add_argument("--max-abs-diff", type=float, default=1e-4)
    args = ap.parse_args()

    model, checkpoint = load_ranker(args.model, args.device)
    payload = load_policy_ranker_export(args.export)
    feature_set = args.feature_set
    if feature_set == "auto":
        feature_set = str(checkpoint.get("feature_set", payload["feature_set"]))
    if feature_set != payload["feature_set"]:
        raise SystemExit(
            f"feature_set mismatch: gate={feature_set} export={payload['feature_set']}")

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

    max_abs_diff = 0.0
    argmax_mismatches = 0
    score_count = 0
    for group in groups:
        pt_scores = score_group(model, group, mean, std, args.device)
        exported_scores = score_group_export(payload, group)
        diff = torch.max(torch.abs(pt_scores - exported_scores)).item()
        max_abs_diff = max(max_abs_diff, float(diff))
        score_count += int(pt_scores.numel())
        if int(torch.argmax(pt_scores).item()) != int(torch.argmax(exported_scores).item()):
            argmax_mismatches += 1

    print(
        f"groups={len(groups)} scores={score_count} "
        f"max_abs_diff={max_abs_diff:.8f} "
        f"argmax_mismatches={argmax_mismatches}",
        flush=True,
    )
    if max_abs_diff > args.max_abs_diff:
        raise SystemExit(
            f"max_abs_diff {max_abs_diff:.8f} > {args.max_abs_diff}")
    if argmax_mismatches:
        raise SystemExit(f"argmax_mismatches={argmax_mismatches}")


if __name__ == "__main__":
    main()
