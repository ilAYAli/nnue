#!/usr/bin/env python3
"""Gate an exported one-pair search-aware overfit diagnostic."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.enyo_nnue import load_net
from lib.nnue_model import EnyoNNUE
from train.train_search_aware import (
    SearchAwareTargetDataset,
    collate_targets,
    load_model_from_nn,
    predict,
    to_device,
)


@dataclass
class MoveReport:
    uci: str
    rank: int
    gap_cp: float
    policy: float
    pred_cp: float
    is_best: bool
    is_bad: bool


@dataclass
class TargetReport:
    target_id: str
    best_move: str
    bad_move: str
    selected_move: str
    selected_gap_cp: float
    bad_margin_cp: float
    worst_margin_cp: float
    moves: list[MoveReport]


def read_rows(path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.expanduser().open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def raw_row_map(path: Path, *, limit: int = 0) -> dict[str, dict[str, Any]]:
    return {str(row.get("id", "")): row for row in read_rows(path, limit=limit)}


def row_candidate_move(row: dict[str, Any], moves: list[str], gaps: list[float],
                       best_move: str) -> str:
    candidates = [str(move).lower() for move in row.get("candidate_moves", [])]
    for candidate in candidates:
        if candidate in moves and candidate != best_move:
            return candidate
    worst_index = max(
        (idx for idx, move in enumerate(moves) if move != best_move),
        key=lambda idx: gaps[idx],
        default=-1,
    )
    if worst_index < 0:
        return best_move
    return moves[worst_index]


def choose_order(pred: torch.Tensor, score_mode: str) -> torch.Tensor:
    if score_mode == "root-high":
        return torch.argsort(pred, descending=True)
    return torch.argsort(pred)


def margin_vs_best(best_pred: float, move_pred: float, score_mode: str) -> float:
    if score_mode == "root-high":
        return best_pred - move_pred
    return move_pred - best_pred


@torch.no_grad()
def score_model(
    model: EnyoNNUE,
    target_path: Path,
    *,
    raw_rows: dict[str, dict[str, Any]],
    device: str,
    forward: str,
    score_mode: str,
    batch_size: int,
    target_limit: int,
    max_moves: int,
    max_gap_cp: float,
) -> list[TargetReport]:
    target_set = SearchAwareTargetDataset.from_jsonl(
        target_path,
        limit=target_limit,
        max_moves=max_moves,
        max_gap_cp=max_gap_cp,
        tag_weights={},
        required_tags=[],
    )
    if len(target_set) == 0:
        raise SystemExit(f"{target_path}: no targets after filtering")
    loader = DataLoader(
        target_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_targets,
    )
    args = argparse.Namespace(
        device=device,
        forward=forward,
        search_score_mode=score_mode,
    )
    model.eval()
    reports: list[TargetReport] = []
    for batch in loader:
        tensors = to_device(batch[:8], device)
        w, b, w_off, b_off, stm, phase_scale, group_offsets, best_indices = tensors
        gaps_t = batch[8].to(device)
        policies_t = batch[9].to(device)
        ids = batch[11]
        moves = batch[13]
        pred = predict(model, args, w, b, w_off, b_off, stm, phase_scale)
        for i, target_id in enumerate(ids):
            start = int(group_offsets[i])
            end = int(group_offsets[i + 1])
            group_pred = pred[start:end]
            group_gaps = gaps_t[start:end]
            group_policies = policies_t[start:end]
            group_moves = [str(move).lower() for move in moves[start:end]]
            best_local = int(best_indices[i]) - start
            best_move = group_moves[best_local]
            best_pred = float(group_pred[best_local].detach().cpu())
            order = choose_order(group_pred, score_mode)
            selected = int(order[0])
            selected_move = group_moves[selected]
            selected_gap = float(group_gaps[selected].detach().cpu())

            raw = raw_rows.get(str(target_id), {})
            gaps = [float(value.detach().cpu()) for value in group_gaps]
            bad_move = row_candidate_move(raw, group_moves, gaps, best_move)
            bad_local = group_moves.index(bad_move)
            bad_margin = margin_vs_best(
                best_pred,
                float(group_pred[bad_local].detach().cpu()),
                score_mode,
            )
            all_margins = [
                margin_vs_best(best_pred, float(group_pred[j].detach().cpu()), score_mode)
                for j, move in enumerate(group_moves)
                if move != best_move
            ]
            worst_margin = min(all_margins) if all_margins else 0.0
            reports.append(TargetReport(
                target_id=str(target_id),
                best_move=best_move,
                bad_move=bad_move,
                selected_move=selected_move,
                selected_gap_cp=selected_gap,
                bad_margin_cp=bad_margin,
                worst_margin_cp=worst_margin,
                moves=[
                    MoveReport(
                        uci=move,
                        rank=idx + 1,
                        gap_cp=float(group_gaps[idx].detach().cpu()),
                        policy=float(group_policies[idx].detach().cpu()),
                        pred_cp=float(group_pred[idx].detach().cpu()),
                        is_best=(move == best_move),
                        is_bad=(move == bad_move),
                    )
                    for idx, move in enumerate(group_moves)
                ],
            ))
    model.train()
    return reports


def load_pt_model(path: Path, device: str) -> EnyoNNUE:
    model = EnyoNNUE().to(device)
    state = torch.load(path.expanduser(), map_location=device)
    model.load_state_dict(state)
    return model


def print_report(label: str, report: TargetReport) -> None:
    print(f"[{label}]")
    print(f"id={report.target_id}")
    print(f"best={report.best_move} bad={report.bad_move} selected={report.selected_move}")
    print(f"selected_gap_cp={report.selected_gap_cp:.0f}")
    print(f"bad_margin_cp={report.bad_margin_cp:.1f}")
    print(f"worst_margin_cp={report.worst_margin_cp:.1f}")
    for move in sorted(report.moves, key=lambda item: item.rank):
        markers = []
        if move.is_best:
            markers.append("best")
        if move.is_bad:
            markers.append("bad")
        print(
            f"  {move.uci:5s} gap={move.gap_cp:7.0f} "
            f"pred={move.pred_cp:9.2f} policy={move.policy:.6f} "
            f"{','.join(markers)}"
        )


def write_reports_csv(path: Path, label: str, reports: list[TargetReport]) -> None:
    rows = []
    for report in reports:
        for move in report.moves:
            rows.append({
                "label": label,
                "id": report.target_id,
                "move": move.uci,
                "gap_cp": f"{move.gap_cp:.0f}",
                "pred_cp": f"{move.pred_cp:.3f}",
                "policy": f"{move.policy:.8f}",
                "is_best": int(move.is_best),
                "is_bad": int(move.is_bad),
                "selected_move": report.selected_move,
                "bad_margin_cp": f"{report.bad_margin_cp:.3f}",
                "worst_margin_cp": f"{report.worst_margin_cp:.3f}",
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)


def count_net_diffs(candidate_path: Path, reference_path: Path) -> int:
    candidate = load_net(candidate_path.expanduser())
    reference = load_net(reference_path.expanduser())
    pairs = [
        ("input_weights", candidate.input_weights, reference.input_weights),
        ("input_biases", candidate.input_biases, reference.input_biases),
        ("l1_weights", candidate.l1_weights, reference.l1_weights),
        ("l1_biases", candidate.l1_biases, reference.l1_biases),
        ("l2_weights", candidate.l2_weights, reference.l2_weights),
        ("l2_biases", candidate.l2_biases, reference.l2_biases),
        ("output_weights", candidate.output_weights, reference.output_weights),
        ("output_bias", np.asarray(candidate.output_bias), np.asarray(reference.output_bias)),
    ]
    total_changed = 0
    print("[net-diff]")
    for name, cand, ref in pairs:
        delta = np.asarray(cand) - np.asarray(ref)
        changed = int(np.count_nonzero(delta))
        total_changed += changed
        max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
        print(f"{name:14s} changed={changed}/{delta.size} max_abs={max_abs:g}")
    print(f"{'total':14s} changed={total_changed}")
    return total_changed


def engine_eval_children(
    engine: Path,
    net: Path,
    row: dict[str, Any],
    moves: list[str],
    timeout: float,
) -> dict[str, float]:
    board = chess.Board(str(row["fen"]))
    commands = [
        "uci",
        "isready",
        f"evalnet load {net.expanduser()}",
    ]
    for uci in moves:
        child = board.copy(stack=False)
        child.push(chess.Move.from_uci(uci))
        commands.append(f"position fen {child.fen()}")
        commands.append("evalnet")
    commands.append("quit")
    result = subprocess.run(
        [str(engine.expanduser())],
        input="\n".join(commands) + "\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    if "uciok" not in result.stdout:
        raise RuntimeError("engine did not report uciok")
    load_lines = [
        line for line in result.stdout.splitlines()
        if line.startswith("evalnet load ")
    ]
    if not load_lines or " ok " not in load_lines[-1]:
        raise RuntimeError(load_lines[-1] if load_lines else "missing evalnet load")

    eval_lines = [
        line for line in result.stdout.splitlines()
        if re.match(r"^evalnet -?\d+ cp ", line)
    ]
    if len(eval_lines) != len(moves):
        raise RuntimeError(
            f"expected {len(moves)} evalnet lines, got {len(eval_lines)}")
    values: dict[str, float] = {}
    for uci, line in zip(moves, eval_lines):
        match = re.search(r"^evalnet (-?\d+) cp ", line)
        if match is None:
            raise RuntimeError(line)
        values[uci] = float(match.group(1))
    return values


def report_from_engine(
    values: dict[str, float],
    base_report: TargetReport,
    score_mode: str,
) -> TargetReport:
    best_pred = values[base_report.best_move]
    order = sorted(
        base_report.moves,
        key=lambda move: values[move.uci],
        reverse=(score_mode == "root-high"),
    )
    selected = order[0]
    bad_pred = values[base_report.bad_move]
    all_margins = [
        margin_vs_best(best_pred, values[move.uci], score_mode)
        for move in base_report.moves
        if move.uci != base_report.best_move
    ]
    return TargetReport(
        target_id=base_report.target_id,
        best_move=base_report.best_move,
        bad_move=base_report.bad_move,
        selected_move=selected.uci,
        selected_gap_cp=selected.gap_cp,
        bad_margin_cp=margin_vs_best(best_pred, bad_pred, score_mode),
        worst_margin_cp=min(all_margins) if all_margins else 0.0,
        moves=[
            MoveReport(
                uci=move.uci,
                rank=move.rank,
                gap_cp=move.gap_cp,
                policy=move.policy,
                pred_cp=values[move.uci],
                is_best=move.is_best,
                is_bad=move.is_bad,
            )
            for move in base_report.moves
        ],
    )


def check_broad_sanity(
    candidate: EnyoNNUE,
    reference: EnyoNNUE,
    args: argparse.Namespace,
) -> int:
    broad_path = Path(args.broad_targets)
    raw_rows = raw_row_map(broad_path, limit=args.broad_limit)
    cand = score_model(
        candidate,
        broad_path,
        raw_rows=raw_rows,
        device=args.device,
        forward=args.forward,
        score_mode=args.search_score_mode,
        batch_size=args.batch_size,
        target_limit=args.broad_limit,
        max_moves=args.broad_max_moves,
        max_gap_cp=args.max_gap_cp,
    )
    ref = score_model(
        reference,
        broad_path,
        raw_rows=raw_rows,
        device=args.device,
        forward=args.forward,
        score_mode=args.search_score_mode,
        batch_size=args.batch_size,
        target_limit=args.broad_limit,
        max_moves=args.broad_max_moves,
        max_gap_cp=args.max_gap_cp,
    )
    worst = max(
        (cand_report.selected_gap_cp - ref_report.selected_gap_cp
         for cand_report, ref_report in zip(cand, ref)),
        default=0.0,
    )
    cand_top1 = sum(1 for report in cand if report.selected_gap_cp == 0.0)
    ref_top1 = sum(1 for report in ref if report.selected_gap_cp == 0.0)
    print("[broad-sanity]")
    print(f"targets={len(cand)}")
    print(f"candidate_top1={cand_top1}/{len(cand)}")
    print(f"reference_top1={ref_top1}/{len(ref)}")
    print(f"worst_new_regression_cp={worst:.0f}")
    return 1 if worst > args.broad_max_new_regression_cp else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--net", required=True, help="Exported candidate .nn")
    parser.add_argument("--pt", default="", help="Optional candidate .pt")
    parser.add_argument("--reference-net", default="")
    parser.add_argument("--engine", default="")
    parser.add_argument("--forward", default="quantized", choices=["float", "quantized"])
    parser.add_argument("--search-score-mode", default="child-low",
                        choices=["child-low", "root-high"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--target-limit", type=int, default=1)
    parser.add_argument("--max-moves", type=int, default=0)
    parser.add_argument("--max-gap-cp", type=float, default=800.0)
    parser.add_argument("--pt-min-bad-margin-cp", type=float, default=100.0)
    parser.add_argument("--nn-min-bad-margin-cp", type=float, default=50.0)
    parser.add_argument("--pt-min-worst-margin-cp", type=float, default=-25.0)
    parser.add_argument("--nn-min-worst-margin-cp", type=float, default=-25.0)
    parser.add_argument("--engine-min-bad-margin-cp", type=float, default=50.0)
    parser.add_argument("--engine-nn-tolerance-cp", type=float, default=5.0)
    parser.add_argument("--engine-timeout", type=float, default=10.0)
    parser.add_argument("--fail-if-net-identical", action="store_true")
    parser.add_argument("--broad-targets", default="")
    parser.add_argument("--broad-limit", type=int, default=100)
    parser.add_argument("--broad-max-moves", type=int, default=12)
    parser.add_argument("--broad-max-new-regression-cp", type=float, default=300.0)
    parser.add_argument("--out-csv", default="")
    args = parser.parse_args()

    rc = 0
    target_path = Path(args.targets)
    raw_rows = raw_row_map(target_path, limit=args.target_limit)
    first_raw = next(iter(raw_rows.values()))

    if args.pt:
        pt_model = load_pt_model(Path(args.pt), args.device)
        pt_reports = score_model(
            pt_model,
            target_path,
            raw_rows=raw_rows,
            device=args.device,
            forward=args.forward,
            score_mode=args.search_score_mode,
            batch_size=args.batch_size,
            target_limit=args.target_limit,
            max_moves=args.max_moves,
            max_gap_cp=args.max_gap_cp,
        )
        pt_report = pt_reports[0]
        print_report("pt", pt_report)
        if args.out_csv:
            write_reports_csv(Path(args.out_csv), "pt", pt_reports)
        if pt_report.bad_margin_cp < args.pt_min_bad_margin_cp:
            print(
                "FAIL pt bad margin "
                f"{pt_report.bad_margin_cp:.1f} < {args.pt_min_bad_margin_cp:.1f}")
            rc = 1
        if pt_report.worst_margin_cp < args.pt_min_worst_margin_cp:
            print(
                "FAIL pt worst margin "
                f"{pt_report.worst_margin_cp:.1f} < {args.pt_min_worst_margin_cp:.1f}")
            rc = 1

    net_model = load_model_from_nn(args.net, device=args.device)
    net_reports = score_model(
        net_model,
        target_path,
        raw_rows=raw_rows,
        device=args.device,
        forward=args.forward,
        score_mode=args.search_score_mode,
        batch_size=args.batch_size,
        target_limit=args.target_limit,
        max_moves=args.max_moves,
        max_gap_cp=args.max_gap_cp,
    )
    net_report = net_reports[0]
    print_report("nn", net_report)
    if args.out_csv:
        write_reports_csv(Path(args.out_csv), "nn", net_reports)
    if net_report.bad_margin_cp < args.nn_min_bad_margin_cp:
        print(
            "FAIL nn bad margin "
            f"{net_report.bad_margin_cp:.1f} < {args.nn_min_bad_margin_cp:.1f}")
        rc = 1
    if net_report.worst_margin_cp < args.nn_min_worst_margin_cp:
        print(
            "FAIL nn worst margin "
            f"{net_report.worst_margin_cp:.1f} < {args.nn_min_worst_margin_cp:.1f}")
        rc = 1

    if args.reference_net:
        changed = count_net_diffs(Path(args.net), Path(args.reference_net))
        if args.fail_if_net_identical and changed == 0:
            print("FAIL net diff is identical")
            rc = 1

    if args.engine:
        values = engine_eval_children(
            Path(args.engine),
            Path(args.net),
            first_raw,
            [move.uci for move in net_report.moves],
            args.engine_timeout,
        )
        engine_report = report_from_engine(values, net_report, args.search_score_mode)
        print_report("engine", engine_report)
        if args.out_csv:
            write_reports_csv(Path(args.out_csv), "engine", [engine_report])
        if engine_report.bad_margin_cp < args.engine_min_bad_margin_cp:
            print(
                "FAIL engine bad margin "
                f"{engine_report.bad_margin_cp:.1f} < "
                f"{args.engine_min_bad_margin_cp:.1f}")
            rc = 1
        for move in engine_report.moves:
            nn_pred = next(item.pred_cp for item in net_report.moves if item.uci == move.uci)
            if abs(move.pred_cp - nn_pred) > args.engine_nn_tolerance_cp:
                print(
                    "FAIL engine/nn mismatch "
                    f"{move.uci}: engine={move.pred_cp:.1f} nn={nn_pred:.1f}")
                rc = 1

    if args.broad_targets and args.reference_net:
        ref_model = load_model_from_nn(args.reference_net, device=args.device)
        rc |= check_broad_sanity(net_model, ref_model, args)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
