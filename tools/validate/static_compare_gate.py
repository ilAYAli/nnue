#!/usr/bin/env python3
"""Compare static dataset metrics between candidate and reference nets."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

from lib.nnue_dataset import load_score_dataset
from lib.nnue_model import load_model_from_nn
from train_impl import MPE_EXPONENT, MPE_SCALE


BUCKETS = (
    (0.0, 50.0),
    (50.0, 100.0),
    (100.0, 300.0),
    (300.0, 800.0),
    (800.0, 1600.0),
    (1600.0, float("inf")),
)


def bucket_name(lo: float, hi: float) -> str:
    if math.isinf(hi):
        return f"{lo:.0f}+"
    return f"{lo:.0f}-{hi:.0f}"


def empty_stats() -> dict[str, float]:
    return {
        "n": 0,
        "mae": 0.0,
        "mse": 0.0,
        "mpe": 0.0,
        "sign": 0,
        "sign_n": 0,
        "pred_sum": 0.0,
        "target_sum": 0.0,
    }


def update_stats(stats: dict[str, float], pred: torch.Tensor,
                 target: torch.Tensor) -> None:
    err = pred - target
    sign_mask = target != 0
    stats["n"] += len(target)
    stats["mae"] += float(err.abs().sum())
    stats["mse"] += float((err * err).sum())
    stats["mpe"] += float(
        ((torch.sigmoid(pred * MPE_SCALE)
          - torch.sigmoid(target * MPE_SCALE)).abs() ** MPE_EXPONENT).sum())
    stats["sign"] += int(
        ((pred[sign_mask] > 0) == (target[sign_mask] > 0)).sum())
    stats["sign_n"] += int(sign_mask.sum())
    stats["pred_sum"] += float(pred.sum())
    stats["target_sum"] += float(target.sum())


def finalize(stats: dict[str, float]) -> dict[str, float]:
    n = max(1.0, stats["n"])
    sign_n = max(1.0, stats["sign_n"])
    return {
        "rows": stats["n"],
        "mae": stats["mae"] / n,
        "mse": stats["mse"] / n,
        "rmse": math.sqrt(stats["mse"] / n),
        "mpe25": stats["mpe"] / n,
        "sign": stats["sign"] / sign_n,
        "wrong_sign": stats["sign_n"] - stats["sign"],
        "sign_n": stats["sign_n"],
        "pred_mean": stats["pred_sum"] / n,
        "target_mean": stats["target_sum"] / n,
    }


def bucket_stats() -> list[dict[str, float]]:
    return [empty_stats() for _ in BUCKETS]


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--reference-net", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--rows", type=int, default=100000)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--target-clamp", type=float, default=0.0)
    ap.add_argument("--fail-if-mae-regression-gt", type=float, default=-1.0)
    ap.add_argument("--fail-if-sign-drop-gt", type=float, default=-1.0,
                    help="Maximum allowed sign drop in percentage points.")
    ap.add_argument("--fail-if-near-zero-sign-drop-gt", type=float,
                    default=-1.0,
                    help="Maximum allowed 0-50cp bucket sign drop in points.")
    args = ap.parse_args()

    ds, collate_fn = load_score_dataset(
        args.data, limit=args.rows, skip=args.skip)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn)

    candidate = load_model_from_nn(args.net, device=args.device)
    reference = load_model_from_nn(args.reference_net, device=args.device)
    candidate.eval()
    reference.eval()

    cand_stats = empty_stats()
    ref_stats = empty_stats()
    cand_buckets = bucket_stats()
    ref_buckets = bucket_stats()

    for w, b, w_off, b_off, stm, y, _wdl, phase_scale, _source_ids in loader:
        w = w.to(args.device)
        b = b.to(args.device)
        w_off = w_off.to(args.device)
        b_off = b_off.to(args.device)
        stm = stm.to(args.device)
        y = y.to(args.device)
        phase_scale = phase_scale.to(args.device)
        if args.target_clamp > 0:
            y = torch.clamp(y, -args.target_clamp, args.target_clamp)

        cand_pred = candidate(w, b, w_off, b_off, stm, phase_scale)
        ref_pred = reference(w, b, w_off, b_off, stm, phase_scale)
        update_stats(cand_stats, cand_pred, y)
        update_stats(ref_stats, ref_pred, y)

        abs_y = y.abs()
        for idx, (lo, hi) in enumerate(BUCKETS):
            mask = (abs_y >= lo) if math.isinf(hi) else (
                (abs_y >= lo) & (abs_y < hi))
            if int(mask.sum()) == 0:
                continue
            update_stats(cand_buckets[idx], cand_pred[mask], y[mask])
            update_stats(ref_buckets[idx], ref_pred[mask], y[mask])

    cand = finalize(cand_stats)
    ref = finalize(ref_stats)
    mae_delta = cand["mae"] - ref["mae"]
    sign_drop = (ref["sign"] - cand["sign"]) * 100.0

    print(
        f"candidate rows={int(cand['rows'])} mae={cand['mae']:.3f} "
        f"sign={cand['sign'] * 100:.2f}% "
        f"wrong_sign={int(cand['wrong_sign'])}/{int(cand['sign_n'])}")
    print(
        f"reference rows={int(ref['rows'])} mae={ref['mae']:.3f} "
        f"sign={ref['sign'] * 100:.2f}% "
        f"wrong_sign={int(ref['wrong_sign'])}/{int(ref['sign_n'])}")
    print(f"delta mae={mae_delta:+.3f} sign_drop_pp={sign_drop:+.2f}")

    near_zero_drop = 0.0
    print("bucket candidate_mae reference_mae candidate_sign reference_sign sign_drop_pp")
    for idx, (bounds, cand_raw, ref_raw) in enumerate(
            zip(BUCKETS, cand_buckets, ref_buckets)):
        cand_item = finalize(cand_raw)
        ref_item = finalize(ref_raw)
        drop = (ref_item["sign"] - cand_item["sign"]) * 100.0
        if idx == 0:
            near_zero_drop = drop
        print(
            f"{bucket_name(*bounds):>9}"
            f" {cand_item['mae']:13.2f}"
            f" {ref_item['mae']:13.2f}"
            f" {cand_item['sign'] * 100:13.2f}%"
            f" {ref_item['sign'] * 100:14.2f}%"
            f" {drop:+12.2f}")

    failed = False
    if (args.fail_if_mae_regression_gt >= 0
            and mae_delta > args.fail_if_mae_regression_gt):
        print(
            "FAIL: mae regression "
            f"{mae_delta:.3f} > {args.fail_if_mae_regression_gt:.3f}",
            file=sys.stderr)
        failed = True
    if args.fail_if_sign_drop_gt >= 0 and sign_drop > args.fail_if_sign_drop_gt:
        print(
            "FAIL: sign drop "
            f"{sign_drop:.2f} > {args.fail_if_sign_drop_gt:.2f}",
            file=sys.stderr)
        failed = True
    if (args.fail_if_near_zero_sign_drop_gt >= 0
            and near_zero_drop > args.fail_if_near_zero_sign_drop_gt):
        print(
            "FAIL: near-zero sign drop "
            f"{near_zero_drop:.2f} > "
            f"{args.fail_if_near_zero_sign_drop_gt:.2f}",
            file=sys.stderr)
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
