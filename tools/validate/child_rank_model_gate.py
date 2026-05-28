#!/usr/bin/env python3
"""Gate child-move ranking targets with PyTorch and exported NNUE models."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import enyo_nnue as nn2
from lib.child_rank_targets import ChildRankGroup, child_fen, load_groups, target_gap_cp
from lib.nnue_model import EnyoNNUE, load_model_from_nn


def fen_features(fen: str):
    pieces, stm = nn2.parse_fen(fen)
    pieces.sort(key=lambda item: item[2])
    return (
        torch.tensor(nn2.features_from_pieces(pieces, nn2.WHITE), dtype=torch.long),
        torch.tensor(nn2.features_from_pieces(pieces, nn2.BLACK), dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([stm], dtype=torch.long),
        torch.tensor([nn2.phase_scale_from_pieces(pieces)], dtype=torch.float32),
    )


@torch.no_grad()
def eval_fen(model: EnyoNNUE, fen: str, device: str) -> float:
    w, b, w_off, b_off, stm, phase = fen_features(fen)
    return float(model(
        w.to(device), b.to(device), w_off.to(device), b_off.to(device),
        stm.to(device), phase.to(device)).item())


@torch.no_grad()
def group_result(model: EnyoNNUE, group: ChildRankGroup, device: str) -> dict:
    evals = {}
    for move in group.moves:
        evals[move.move] = -eval_fen(model, child_fen(group.fen, move.move), device)
    selected = max(evals, key=evals.get)
    best_eval = evals[group.best_move]
    selected_gap = target_gap_cp(
        group,
        next(move for move in group.moves if move.move == selected),
    )
    worst_margin = min(
        best_eval - value
        for move, value in evals.items()
        if move != group.best_move
    )
    return {
        "group_id": group.group_id,
        "best": group.best_move,
        "selected": selected,
        "correct": selected == group.best_move,
        "selected_gap": selected_gap,
        "worst_margin": worst_margin,
    }


def summarize(model: EnyoNNUE, groups: list[ChildRankGroup], device: str,
              label: str) -> dict[str, float]:
    results = [group_result(model, group, device) for group in groups]
    correct = sum(1 for result in results if result["correct"])
    sum_gap = sum(float(result["selected_gap"]) for result in results)
    worst_margin = min((float(result["worst_margin"]) for result in results), default=0.0)
    print(
        f"{label} top1={correct}/{len(groups)} "
        f"sum_gap={sum_gap:.0f} worst_margin={worst_margin:.1f}",
        flush=True)
    for result in results:
        if result["correct"]:
            continue
        print(
            f"{label} miss {result['group_id']} "
            f"best={result['best']} selected={result['selected']} "
            f"gap={result['selected_gap']:.0f} "
            f"worst_margin={result['worst_margin']:.1f}",
            flush=True)
    return {
        "top1": correct,
        "groups": len(groups),
        "sum_gap": sum_gap,
        "worst_margin": worst_margin,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--pt")
    ap.add_argument("--pt-export-quantize-forward", default=False,
                    action=argparse.BooleanOptionalAction)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-groups", type=int, default=1)
    ap.add_argument("--fail-if-net-top1-below", type=int, default=-1)
    ap.add_argument("--fail-if-pt-top1-below", type=int, default=-1)
    args = ap.parse_args()

    groups = load_groups(args.targets, min_groups=args.min_groups)
    failed = False

    if args.pt:
        pt_model = EnyoNNUE().to(args.device)
        pt_model.load_state_dict(torch.load(args.pt, map_location=args.device))
        pt_model.export_quantize_forward = args.pt_export_quantize_forward
        pt_model.eval()
        pt_summary = summarize(pt_model, groups, args.device, "pt")
        failed |= (
            args.fail_if_pt_top1_below >= 0
            and pt_summary["top1"] < args.fail_if_pt_top1_below)

    net_model = load_model_from_nn(args.net, device=args.device)
    net_model.eval()
    net_summary = summarize(net_model, groups, args.device, "nn")
    failed |= (
        args.fail_if_net_top1_below >= 0
        and net_summary["top1"] < args.fail_if_net_top1_below)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
