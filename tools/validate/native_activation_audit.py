#!/usr/bin/env python3
"""Audit Enyo native NNUE activations on a FEN set.

This uses the repo's Python loader/reference model for Enyo-format .nn files. It
is intentionally not a strength test; it exposes saturation, clamp rate, phase
scaling, and output bucket distribution so training/export/runtime scale issues
are visible before spending SPRT time.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from tools.lib import enyo_nnue as nn2
from tools.lib.nnue_model import load_model_from_nn


@dataclass(frozen=True)
class Row:
    fen: str
    material_bucket: str
    piece_count: int
    output_bucket: int
    phase_scale: float
    raw: float
    scaled: float
    clamped: float
    clamped_flag: bool
    acc_min: float
    acc_max: float
    acc_mean: float
    acc_std: float
    input_zero_frac: float
    input_clip_frac: float
    x1_zero_frac: float
    x1_mean: float
    x1_max: float
    x2_zero_frac: float
    x2_mean: float
    x2_max: float


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def stdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def material_bucket(piece_count: int) -> str:
    if piece_count >= 28:
        return "opening"
    if piece_count >= 18:
        return "middlegame"
    if piece_count >= 10:
        return "late"
    return "endgame"


def fen_from_json_line(line: str) -> str | None:
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return None
    fen = row.get("fen")
    return fen if isinstance(fen, str) else None


def load_fens(path: Path | None, structural_json: Path | None, limit: int) -> list[str]:
    if structural_json is not None:
        payload = json.loads(structural_json.read_text(encoding="utf-8"))
        return list(payload["fens"][:limit])
    if path is None:
        raise SystemExit("provide --fen-file or --structural-json")
    fens: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fen = fen_from_json_line(stripped) if stripped.startswith("{") else stripped
            if path.suffix.lower() == ".epd" and fen:
                fen = " ".join(fen.split()[:4]) + " 0 1"
            if fen:
                fens.append(fen)
            if len(fens) >= limit:
                break
    return fens


def tensor_for(values: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor(values, dtype=torch.long), torch.tensor([0], dtype=torch.long)


def audit_fen(model, fen: str) -> Row:
    pieces, stm = nn2.parse_fen(fen)
    w = nn2.features_from_pieces(
        pieces, nn2.WHITE, model.input_buckets, model.feature_channels, model.full_threats)
    b = nn2.features_from_pieces(
        pieces, nn2.BLACK, model.input_buckets, model.feature_channels, model.full_threats)
    w_feats, w_offsets = tensor_for(w)
    b_feats, b_offsets = tensor_for(b)
    stm_t = torch.tensor([stm], dtype=torch.long)
    phase = nn2.phase_scale_from_pieces(pieces)
    piece_count = len(pieces)
    output_bucket = nn2.output_bucket_from_pieces(pieces, model.output_buckets)
    output_bucket_t = torch.tensor([output_bucket], dtype=torch.long)
    head_features = None
    if model.output_head_features:
        head_features = torch.tensor(
            [nn2.material_head_features_from_pieces(pieces)], dtype=torch.float32)

    with torch.no_grad():
        w_acc = model.accumulator(w_feats, w_offsets)
        b_acc = model.accumulator(b_feats, b_offsets)
        stm_f = stm_t.unsqueeze(-1).float()
        us = (1.0 - stm_f) * w_acc + stm_f * b_acc
        them = stm_f * w_acc + (1.0 - stm_f) * b_acc
        acc = torch.cat([us, them], dim=-1)
        x0 = model._quantized_input_relu(acc)
        if model.full_heads:
            l1_weight = model.l1_weight[output_bucket_t]
            l1_bias = model.l1_bias[output_bucket_t]
            x1 = torch.relu(torch.bmm(l1_weight, x0.unsqueeze(-1)).squeeze(-1) + l1_bias)
            l2_weight = model.l2_weight[output_bucket_t]
            l2_bias = model.l2_bias[output_bucket_t]
            x2 = torch.relu(torch.bmm(l2_weight, x1.unsqueeze(-1)).squeeze(-1) + l2_bias)
        else:
            x1 = torch.relu(x0 @ model.l1_weight.t() + model.l1_bias)
            x2 = torch.relu(model.l2(x1))
        if head_features is not None:
            x2_out = torch.cat([x2, head_features], dim=-1)
        else:
            x2_out = x2
        raw_all = model.output(x2_out) / nn2.EVAL_DIVISOR
        raw = raw_all[:, output_bucket].reshape(()).item() if model.output_buckets > 1 else raw_all.reshape(()).item()

    scaled = raw * phase
    clamped = max(-2045.0, min(2045.0, scaled))
    cap = float(127 << nn2.QUANT1_BITS)
    return Row(
        fen=fen,
        material_bucket=material_bucket(piece_count),
        piece_count=piece_count,
        output_bucket=output_bucket,
        phase_scale=phase,
        raw=raw,
        scaled=scaled,
        clamped=clamped,
        clamped_flag=abs(scaled) >= 2045.0,
        acc_min=float(acc.min().item()),
        acc_max=float(acc.max().item()),
        acc_mean=float(acc.mean().item()),
        acc_std=float(acc.std(unbiased=False).item()),
        input_zero_frac=float((acc <= 0.0).float().mean().item()),
        input_clip_frac=float((acc >= cap).float().mean().item()),
        x1_zero_frac=float((x1 <= 0.0).float().mean().item()),
        x1_mean=float(x1.mean().item()),
        x1_max=float(x1.max().item()),
        x2_zero_frac=float((x2 <= 0.0).float().mean().item()),
        x2_mean=float(x2.mean().item()),
        x2_max=float(x2.max().item()),
    )


def summarize(label: str, rows: list[Row]) -> None:
    if not rows:
        return
    print(
        f"{label:12} rows={len(rows):6d}"
        f" raw_mean={mean([r.raw for r in rows]):8.2f} raw_sd={stdev([r.raw for r in rows]):8.2f}"
        f" scaled_sd={stdev([r.scaled for r in rows]):8.2f}"
        f" clamp={sum(r.clamped_flag for r in rows):5d}"
        f" in0={100.0 * mean([r.input_zero_frac for r in rows]):6.2f}%"
        f" incap={100.0 * mean([r.input_clip_frac for r in rows]):6.2f}%"
        f" x1z={100.0 * mean([r.x1_zero_frac for r in rows]):6.2f}%"
        f" x2z={100.0 * mean([r.x2_zero_frac for r in rows]):6.2f}%"
    )


def finite(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, type=Path)
    ap.add_argument("--fen-file", type=Path)
    ap.add_argument("--structural-json", type=Path)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    fens = load_fens(args.fen_file, args.structural_json, args.limit)
    if not fens:
        raise SystemExit("no FENs loaded")

    model = load_model_from_nn(args.net)
    model.eval()
    rows = []
    skipped = []
    for fen in fens:
        try:
            rows.append(audit_fen(model, fen))
        except (StopIteration, ValueError) as exc:
            skipped.append({"fen": fen, "error": f"{type(exc).__name__}: {exc}"})
    if not rows:
        raise SystemExit("no FENs could be encoded")

    print(
        f"net={args.net} rows={len(rows)} skipped={len(skipped)} buckets={model.output_buckets}"
        f" input_buckets={model.input_buckets} channels={model.feature_channels}"
        f" full_threats={model.full_threats} full_heads={model.full_heads}"
    )
    summarize("all", rows)
    for bucket in ("opening", "middlegame", "late", "endgame"):
        summarize(bucket, [r for r in rows if r.material_bucket == bucket])
    for bucket in sorted({r.output_bucket for r in rows}):
        summarize(f"out{bucket}", [r for r in rows if r.output_bucket == bucket])

    worst = sorted(rows, key=lambda r: (r.clamped_flag, abs(r.scaled)), reverse=True)[:10]
    print("\nWorst scaled/clamped positions")
    for r in worst:
        print(
            f"scaled={r.scaled:9.2f} raw={r.raw:9.2f} phase={r.phase_scale:.4f}"
            f" pc={r.piece_count:2d} out={r.output_bucket} clamp={int(r.clamped_flag)} {r.fen}"
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        serializable = [
            {k: finite(v) if isinstance(v, float) else v for k, v in r.__dict__.items()}
            for r in rows
        ]
        args.json_out.write_text(
            json.dumps({"net": str(args.net), "rows": serializable, "skipped": skipped}, indent=2) + "\n",
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
