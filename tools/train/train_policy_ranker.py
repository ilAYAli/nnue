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
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--print-rate", type=int, default=100)
    args = ap.parse_args()

    raw_groups = load_policy_source_groups(args.targets)
    builder = PolicyFeatureBuilder(
        args.base_net, device=args.device, feature_set=args.feature_set)
    groups = [builder.build_group(group) for group in raw_groups]
    train_groups, val_groups = split_policy_groups(
        groups, args.seed, args.val_fraction)
    if not train_groups:
        raise SystemExit("no train groups")

    mean, std = normalize_features(train_groups)
    input_dim = int(mean.numel())
    print(
        f"policy groups: train={len(train_groups)} val={len(val_groups)} "
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
        loss_sum = 0.0
        for group in train_groups:
            opt.zero_grad(set_to_none=True)
            loss = group_loss(model, group, mean, std, args)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach().cpu())

        if epoch % args.print_rate == 0 or epoch == args.epochs - 1:
            model.eval()
            train_top1, train_gap = summarize(
                f"epoch {epoch:4d} train", model, train_groups,
                mean, std, args.device)
            val_top1, val_gap = summarize(
                f"epoch {epoch:4d} val", model, val_groups,
                mean, std, args.device)
            key = (
                val_top1 if val_groups else train_top1,
                -(val_gap if val_groups else train_gap),
            )
            print(
                f"epoch {epoch:4d} loss={loss_sum / len(train_groups):.6f}",
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
                    "targets": list(args.targets),
                    "base_net": args.base_net,
                }

    if best_state is None:
        raise SystemExit("no checkpoint selected")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
