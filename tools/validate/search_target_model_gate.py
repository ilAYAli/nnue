#!/usr/bin/env python3
"""Evaluate saved PyTorch/exported NNUE models on search-aware targets."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.nnue_model import EnyoNNUE
from train.train_search_aware import (
    SearchAwareTargetDataset,
    collate_targets,
    load_model_from_nn,
    parse_required_tags,
    parse_tag_weights,
    predict,
    search_metrics,
    to_device,
)


def print_metrics(label: str, metrics: dict[str, float]) -> None:
    targets = int(metrics["targets"])
    top1 = int(metrics["top1"])
    top3 = int(metrics["top3"])
    print(f"[{label}]")
    print(f"targets={targets}")
    print(f"top1={top1}/{targets}")
    print(f"top3={top3}/{targets}")
    print(f"sum_gap_cp={metrics['sum_gap']:.0f}")
    print(f"median_gap_cp={metrics['median_gap']:.1f}")
    print(f"worst_gap_cp={metrics['worst_gap']:.0f}")


@torch.no_grad()
def write_predictions_csv(path: Path, model: EnyoNNUE, loader: DataLoader,
                          args: argparse.Namespace) -> None:
    model.eval()
    rows: list[dict[str, object]] = []
    for batch in loader:
        tensors = to_device(batch[:8], args.device)
        w, b, w_off, b_off, stm, phase_scale, group_offsets, best_indices = tensors
        gaps = batch[8].to(args.device)
        policies = batch[9].to(args.device)
        ids = batch[11]
        tag_sets = batch[12]
        moves = batch[13]
        pred = predict(model, args, w, b, w_off, b_off, stm, phase_scale)
        for i, target_id in enumerate(ids):
            start = int(group_offsets[i])
            end = int(group_offsets[i + 1])
            group_pred = pred[start:end]
            best = int(best_indices[i]) - start
            if args.search_score_mode == "root-high":
                order = torch.argsort(group_pred, descending=True)
            else:
                order = torch.argsort(group_pred)
            selected = int(order[0])
            top3 = {int(v) for v in order[:min(3, len(order))]}
            rows.append({
                "id": target_id,
                "tags": tag_sets[i],
                "best_move": moves[start + best],
                "model_move": moves[start + selected],
                "model_top1": int(selected == best),
                "model_top3": int(best in top3),
                "model_gap_cp": f"{float(gaps[start + selected].detach().cpu()):.0f}",
                "best_pred_cp": f"{float(group_pred[best].detach().cpu()):.3f}",
                "model_pred_cp": f"{float(group_pred[selected].detach().cpu()):.3f}",
                "model_policy": f"{float(policies[start + selected].detach().cpu()):.8f}",
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    model.train()


def load_pt(path: Path, device: str) -> EnyoNNUE:
    model = EnyoNNUE().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--net", required=True, help="Exported .nn model")
    parser.add_argument("--pt", default="", help="Optional PyTorch state dict")
    parser.add_argument("--forward", default="quantized", choices=["float", "quantized"])
    parser.add_argument("--search-score-mode", default="child-low",
                        choices=["child-low", "root-high"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--target-limit", type=int, default=0)
    parser.add_argument("--required-tags", default="")
    parser.add_argument("--tag-weights", default="")
    parser.add_argument("--max-moves", type=int, default=0)
    parser.add_argument("--max-gap-cp", type=float, default=800.0)
    parser.add_argument("--fail-if-net-top1-below", type=int, default=0)
    parser.add_argument("--fail-if-pt-top1-below", type=int, default=0)
    parser.add_argument(
        "--out-csv",
        default="",
        help="Optional CSV path for exported .nn per-target selected moves.",
    )
    args = parser.parse_args()

    target_set = SearchAwareTargetDataset.from_jsonl(
        args.targets,
        limit=args.target_limit,
        max_moves=args.max_moves,
        max_gap_cp=args.max_gap_cp,
        tag_weights=parse_tag_weights(args.tag_weights),
        required_tags=parse_required_tags(args.required_tags),
    )
    if len(target_set) == 0:
        raise SystemExit("no targets after filtering")

    loader = DataLoader(
        target_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_targets,
    )
    metric_args = argparse.Namespace(
        device=args.device,
        forward=args.forward,
        search_margin_beta=100.0,
        search_policy_temperature_cp=200.0,
        search_score_mode=args.search_score_mode,
    )

    rc = 0
    if args.pt:
        pt_model = load_pt(Path(args.pt), args.device)
        pt_metrics = search_metrics(pt_model, loader, metric_args)
        print_metrics("pt", pt_metrics)
        if int(pt_metrics["top1"]) < args.fail_if_pt_top1_below:
            rc = 1

    net_model = load_model_from_nn(args.net, device=args.device)
    net_metrics = search_metrics(net_model, loader, metric_args)
    print_metrics("nn", net_metrics)
    if args.out_csv:
        write_predictions_csv(Path(args.out_csv), net_model, loader, metric_args)
    if int(net_metrics["top1"]) < args.fail_if_net_top1_below:
        rc = 1

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
