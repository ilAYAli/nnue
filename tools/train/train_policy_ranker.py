#!/usr/bin/env python3
"""Train a sidecar child-move ranker without changing scalar NNUE eval."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.policy_ranker import (
    PolicyFeatureBuilder,
    PolicyGroup,
    PolicyRanker,
    load_policy_source_groups,
    normalize_features,
    score_group,
    split_policy_groups,
)


def group_loss(model: PolicyRanker, group: PolicyGroup,
               mean: torch.Tensor, std: torch.Tensor,
               args: argparse.Namespace) -> torch.Tensor:
    x = ((group.features.to(args.device) - mean.to(args.device)) / std.to(args.device))
    logits = model(x)
    target = torch.softmax(
        (group.oracle_scores.to(args.device) - group.oracle_scores.max().to(args.device))
        / args.target_temperature_cp,
        dim=0,
    )
    return F.kl_div(
        F.log_softmax(logits / args.rank_temperature_cp, dim=0),
        target,
        reduction="batchmean",
    )


def preserve_loss(model: PolicyRanker, group: PolicyGroup,
                  mean: torch.Tensor, std: torch.Tensor,
                  args: argparse.Namespace) -> torch.Tensor:
    x = ((group.features.to(args.device) - mean.to(args.device)) / std.to(args.device))
    logits = model(x)
    base = logits[group.base_idx]
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[group.base_idx] = False
    if not torch.any(mask):
        return logits.sum() * 0.0
    return torch.relu(logits[mask] - base + args.preserve_margin).mean()


@torch.no_grad()
def summarize(label: str, model: PolicyRanker, groups: list[PolicyGroup],
              mean: torch.Tensor, std: torch.Tensor,
              device: str) -> tuple[int, float]:
    if not groups:
        print(f"{label} top1=0/0 sum_gap=0")
        return 0, 0.0
    correct = 0
    sum_gap = 0.0
    for group in groups:
        scores = score_group(model, group, mean, std, device)
        selected = int(torch.argmax(scores).item())
        correct += int(selected == group.best_idx)
        sum_gap += float(group.gaps[selected])
    print(f"{label} top1={correct}/{len(groups)} sum_gap={sum_gap:.0f}", flush=True)
    return correct, sum_gap


@torch.no_grad()
def summarize_preserve(label: str, model: PolicyRanker,
                       groups: list[PolicyGroup], mean: torch.Tensor,
                       std: torch.Tensor, args: argparse.Namespace) -> float:
    if not groups:
        print(f"{label} preserve_loss=0.000000", flush=True)
        return 0.0
    loss_sum = 0.0
    for group in groups:
        loss_sum += float(preserve_loss(model, group, mean, std, args).detach().cpu())
    value = loss_sum / len(groups)
    print(f"{label} preserve_loss={value:.6f}", flush=True)
    return value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--base-net", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--feature-set", choices=["compact", "board"], default="compact")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--rank-temperature-cp", type=float, default=1.0)
    ap.add_argument("--target-temperature-cp", type=float, default=80.0)
    ap.add_argument("--include-tags", default="")
    ap.add_argument("--exclude-tags", default="")
    ap.add_argument("--preserve-include-tags", default="")
    ap.add_argument("--preserve-exclude-tags", default="")
    ap.add_argument("--preserve-weight", type=float, default=0.0)
    ap.add_argument("--preserve-margin", type=float, default=4.0)
    ap.add_argument("--preserve-max-groups", type=int, default=0)
    ap.add_argument("--preserve-val-fraction", type=float, default=-1.0)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--print-rate", type=int, default=100)
    args = ap.parse_args()

    raw_groups = load_policy_source_groups(
        args.targets,
        include_tags=args.include_tags,
        exclude_tags=args.exclude_tags)
    raw_preserve_groups = []
    if args.preserve_weight > 0.0 and (
            args.preserve_include_tags or args.preserve_exclude_tags):
        raw_preserve_groups = load_policy_source_groups(
            args.targets,
            min_groups=0,
            include_tags=args.preserve_include_tags,
            exclude_tags=args.preserve_exclude_tags)
        if args.preserve_max_groups > 0 and len(raw_preserve_groups) > args.preserve_max_groups:
            random.Random(args.seed).shuffle(raw_preserve_groups)
            raw_preserve_groups = raw_preserve_groups[:args.preserve_max_groups]
    builder = PolicyFeatureBuilder(
        args.base_net, device=args.device, feature_set=args.feature_set)
    groups = [builder.build_group(group) for group in raw_groups]
    preserve_groups = [builder.build_group(group) for group in raw_preserve_groups]
    train_groups, val_groups = split_policy_groups(
        groups, args.seed, args.val_fraction)
    preserve_val_fraction = (
        args.val_fraction
        if args.preserve_val_fraction < 0.0
        else args.preserve_val_fraction)
    train_preserve_groups, val_preserve_groups = split_policy_groups(
        preserve_groups, args.seed, preserve_val_fraction) if preserve_groups else ([], [])
    if not train_groups:
        raise SystemExit("no train groups")

    mean, std = normalize_features(train_groups + train_preserve_groups)
    input_dim = int(mean.numel())
    print(
        f"policy groups: train={len(train_groups)} val={len(val_groups)} "
        f"preserve_train={len(train_preserve_groups)} "
        f"preserve_val={len(val_preserve_groups)} "
        f"input_dim={input_dim} hidden={args.hidden} "
        f"feature_set={args.feature_set} dropout={args.dropout}",
        flush=True,
    )

    model = PolicyRanker(
        input_dim=input_dim, hidden=args.hidden, dropout=args.dropout).to(args.device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_key = (-1, float("-inf"))
    for epoch in range(args.epochs):
        model.train()
        random.Random(args.seed + epoch).shuffle(train_groups)
        random.Random(args.seed + epoch + 1000003).shuffle(train_preserve_groups)
        loss_sum = 0.0
        loss_count = 0
        for group in train_groups:
            opt.zero_grad(set_to_none=True)
            loss = group_loss(model, group, mean, std, args)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach().cpu())
            loss_count += 1
        for group in train_preserve_groups:
            opt.zero_grad(set_to_none=True)
            loss = args.preserve_weight * preserve_loss(model, group, mean, std, args)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach().cpu())
            loss_count += 1

        if epoch % args.print_rate == 0 or epoch == args.epochs - 1:
            model.eval()
            train_top1, train_gap = summarize(
                f"epoch {epoch:4d} train", model, train_groups,
                mean, std, args.device)
            val_top1, val_gap = summarize(
                f"epoch {epoch:4d} val", model, val_groups,
                mean, std, args.device)
            val_preserve = summarize_preserve(
                f"epoch {epoch:4d} preserve_val", model, val_preserve_groups,
                mean, std, args)
            key = (
                val_top1 if val_groups else train_top1,
                -(val_gap if val_groups else train_gap),
                -val_preserve,
            )
            print(
                f"epoch {epoch:4d} loss={loss_sum / max(1, loss_count):.6f}",
                flush=True,
            )
            if key > best_key:
                best_key = key
                best_state = {
                    "model": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "input_dim": input_dim,
                    "hidden": args.hidden,
                    "dropout": args.dropout,
                    "feature_set": args.feature_set,
                    "seed": args.seed,
                    "val_fraction": args.val_fraction,
                    "train_group_ids": [group.group_id for group in train_groups],
                    "val_group_ids": [group.group_id for group in val_groups],
                    "train_preserve_group_ids": [
                        group.group_id for group in train_preserve_groups],
                    "val_preserve_group_ids": [
                        group.group_id for group in val_preserve_groups],
                    "targets": list(args.targets),
                    "base_net": args.base_net,
                    "include_tags": args.include_tags,
                    "exclude_tags": args.exclude_tags,
                    "preserve_include_tags": args.preserve_include_tags,
                    "preserve_exclude_tags": args.preserve_exclude_tags,
                    "preserve_weight": args.preserve_weight,
                    "preserve_margin": args.preserve_margin,
                    "preserve_val_fraction": preserve_val_fraction,
                }

    if best_state is None:
        raise SystemExit("no checkpoint selected")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
