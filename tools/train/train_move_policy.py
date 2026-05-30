#!/usr/bin/env python3
"""Train a sidecar best-vs-bad move policy on a fixed move gate."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.move_policy import (  # noqa: E402
    FEATURE_VERSION,
    MovePolicy,
    load_gate_cases,
    normalize,
    pair_tensors,
    score_margins,
    source_breakdown,
)


def split_indices(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    val_n = int(round(n * val_fraction))
    if val_fraction > 0.0:
        val_n = max(1, min(n - 1, val_n))
    return indices[val_n:], indices[:val_n]


def take(tensor: torch.Tensor, indices: list[int]) -> torch.Tensor:
    return tensor[torch.tensor(indices, dtype=torch.long)]


@torch.no_grad()
def evaluate(label: str, model: MovePolicy, rows: list[dict],
             best: torch.Tensor, played: torch.Tensor, mean: torch.Tensor,
             std: torch.Tensor, device: str) -> tuple[int, float]:
    margins = score_margins(model, best, played, mean, std, device).cpu()
    correct = margins > 0
    n = len(rows)
    count = int(correct.sum().item())
    avg_margin = float(margins.mean().item()) if n else 0.0
    print(
        f"{label} correct={count}/{n} ({100.0 * count / max(1, n):.1f}%) "
        f"avg_margin={avg_margin:.3f} by_source={source_breakdown(rows, correct)}",
        flush=True,
    )
    return count, avg_margin


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--feature-set", choices=["compact", "board"],
                    default="compact")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-fraction", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--print-rate", type=int, default=50)
    args = ap.parse_args()

    rows = load_gate_cases(args.cases)
    best, played = pair_tensors(rows, args.feature_set)
    train_idx, val_idx = split_indices(len(rows), args.val_fraction, args.seed)
    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    train_best = take(best, train_idx)
    train_played = take(played, train_idx)
    val_best = take(best, val_idx) if val_idx else best[:0]
    val_played = take(played, val_idx) if val_idx else played[:0]
    mean, std = normalize(train_best, train_played)

    model = MovePolicy(input_dim=int(mean.numel()), hidden=args.hidden).to(args.device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"policy cases: train={len(train_idx)} val={len(val_idx)} "
        f"input_dim={mean.numel()} hidden={args.hidden} "
        f"feature_set={args.feature_set}",
        flush=True,
    )
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        margins = score_margins(
            model, train_best, train_played, mean, std, args.device)
        loss = F.softplus(-margins).mean()
        loss.backward()
        opt.step()
        if epoch % args.print_rate == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d} loss={float(loss.item()):.6f}", flush=True)
            model.eval()
            evaluate(
                "train", model, train_rows, train_best, train_played,
                mean, std, args.device)
            if val_idx:
                evaluate(
                    "val", model, val_rows, val_best, val_played,
                    mean, std, args.device)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "feature_version": FEATURE_VERSION,
        "input_dim": int(mean.numel()),
        "hidden": int(args.hidden),
        "feature_set": args.feature_set,
        "model": model.cpu().state_dict(),
        "mean": mean.cpu(),
        "std": std.cpu(),
        "config": vars(args),
    }, out)
    manifest = out.with_suffix(".json")
    manifest.write_text(json.dumps({
        "feature_version": FEATURE_VERSION,
        "input_dim": int(mean.numel()),
        "hidden": int(args.hidden),
        "feature_set": args.feature_set,
        "cases": len(rows),
        "train": len(train_idx),
        "val": len(val_idx),
        "model": str(out),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
