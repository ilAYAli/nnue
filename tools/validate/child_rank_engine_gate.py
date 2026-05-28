#!/usr/bin/env python3
"""Gate child-move ranking targets with Enyo's engine eval2/evalnet path."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.child_rank_targets import ChildRankGroup, child_fen, load_groups, target_gap_cp
from eval_move_gate import EnyoEval2, original_pov_child_eval


def group_result(engine: EnyoEval2, group: ChildRankGroup) -> dict:
    evals = {}
    for move in group.moves:
        evals[move.move] = original_pov_child_eval(
            engine, child_fen(group.fen, move.move))
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


def summarize(results: list[dict]) -> dict[str, float]:
    correct = sum(1 for result in results if result["correct"])
    sum_gap = sum(float(result["selected_gap"]) for result in results)
    worst_margin = min((float(result["worst_margin"]) for result in results), default=0.0)
    print(
        f"engine top1={correct}/{len(results)} "
        f"sum_gap={sum_gap:.0f} worst_margin={worst_margin:.1f}",
        flush=True)
    for result in results:
        if result["correct"]:
            continue
        print(
            f"engine miss {result['group_id']} "
            f"best={result['best']} selected={result['selected']} "
            f"gap={result['selected_gap']:.0f} "
            f"worst_margin={result['worst_margin']:.1f}",
            flush=True)
    return {
        "top1": correct,
        "groups": len(results),
        "sum_gap": sum_gap,
        "worst_margin": worst_margin,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, type=Path)
    ap.add_argument("--engine", required=True, type=Path)
    ap.add_argument("--net", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--min-groups", type=int, default=1)
    ap.add_argument("--fail-if-top1-below", type=int, default=-1)
    args = ap.parse_args()

    groups = load_groups(args.targets, min_groups=args.min_groups)
    engine = EnyoEval2(args.engine, args.net, args.threads, args.hash)
    try:
        results = [group_result(engine, group) for group in groups]
        summary = summarize(results)
    finally:
        engine.close()

    if (
        args.fail_if_top1_below >= 0
        and summary["top1"] < args.fail_if_top1_below
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
