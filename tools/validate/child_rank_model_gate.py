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


FeatureRow = tuple[list[int], list[int], int, float]


def fen_features(fen: str) -> FeatureRow:
    pieces, stm = nn2.parse_fen(fen)
    pieces.sort(key=lambda item: item[2])
    return (
        nn2.features_from_pieces(pieces, nn2.WHITE),
        nn2.features_from_pieces(pieces, nn2.BLACK),
        stm,
        nn2.phase_scale_from_pieces(pieces),
    )


def pack_feature_rows(rows: list[FeatureRow], device: str):
    w_all: list[int] = []
    b_all: list[int] = []
    w_offsets: list[int] = []
    b_offsets: list[int] = []
    stms: list[int] = []
    phase_scales: list[float] = []
    w_pos = 0
    b_pos = 0
    for w, b, stm, phase_scale in rows:
        w_offsets.append(w_pos)
        b_offsets.append(b_pos)
        w_all.extend(w)
        b_all.extend(b)
        w_pos += len(w)
        b_pos += len(b)
        stms.append(stm)
        phase_scales.append(phase_scale)
    return (
        torch.tensor(w_all, dtype=torch.long, device=device),
        torch.tensor(b_all, dtype=torch.long, device=device),
        torch.tensor(w_offsets, dtype=torch.long, device=device),
        torch.tensor(b_offsets, dtype=torch.long, device=device),
        torch.tensor(stms, dtype=torch.long, device=device),
        torch.tensor(phase_scales, dtype=torch.float32, device=device),
    )


@torch.no_grad()
def eval_rows(model: EnyoNNUE, rows: list[FeatureRow], *,
              device: str, batch_size: int) -> list[float]:
    values: list[float] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        values.extend(
            model(*pack_feature_rows(batch, device)).detach().cpu().tolist())
    return [float(value) for value in values]


def prepare_rows(groups: list[ChildRankGroup]) -> tuple[list[FeatureRow], list[int]]:
    rows: list[FeatureRow] = []
    group_sizes: list[int] = []
    for group in groups:
        for move in group.moves:
            rows.append(fen_features(child_fen(group.fen, move.move)))
        group_sizes.append(len(group.moves))
    return rows, group_sizes


def group_result(group: ChildRankGroup, parent_scores: list[float]) -> dict:
    evals = {
        move.move: score
        for move, score in zip(group.moves, parent_scores)
    }
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


def summarize(model: EnyoNNUE, groups: list[ChildRankGroup], *,
              rows: list[FeatureRow], group_sizes: list[int],
              device: str, batch_size: int, label: str) -> dict[str, float]:
    child_scores = [-value for value in eval_rows(
        model, rows, device=device, batch_size=batch_size)]
    results = []
    offset = 0
    for group, size in zip(groups, group_sizes):
        results.append(group_result(group, child_scores[offset:offset + size]))
        offset += size
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
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--min-groups", type=int, default=1)
    ap.add_argument("--fail-if-net-top1-below", type=int, default=-1)
    ap.add_argument("--fail-if-pt-top1-below", type=int, default=-1)
    args = ap.parse_args()

    groups = load_groups(args.targets, min_groups=args.min_groups)
    rows, group_sizes = prepare_rows(groups)
    failed = False

    if args.pt:
        pt_model = EnyoNNUE().to(args.device)
        pt_model.load_state_dict(torch.load(args.pt, map_location=args.device))
        pt_model.export_quantize_forward = args.pt_export_quantize_forward
        pt_model.eval()
        pt_summary = summarize(
            pt_model, groups, rows=rows, group_sizes=group_sizes,
            device=args.device, batch_size=args.batch_size, label="pt")
        failed |= (
            args.fail_if_pt_top1_below >= 0
            and pt_summary["top1"] < args.fail_if_pt_top1_below)

    net_model = load_model_from_nn(args.net, device=args.device)
    net_model.eval()
    net_summary = summarize(
        net_model, groups, rows=rows, group_sizes=group_sizes,
        device=args.device, batch_size=args.batch_size, label="nn")
    failed |= (
        args.fail_if_net_top1_below >= 0
        and net_summary["top1"] < args.fail_if_net_top1_below)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
