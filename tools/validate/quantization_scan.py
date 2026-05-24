#!/usr/bin/env python3
"""Inspect whether float checkpoint movement crosses exported NNUE boundaries."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.enyo_nnue import load_net


@dataclass(frozen=True)
class IntTensorSpec:
    name: str
    state_key: str
    reference_attr: str
    dtype: np.dtype


@dataclass(frozen=True)
class FloatTensorSpec:
    name: str
    state_key: str
    reference_attr: str


INT_TENSORS = (
    IntTensorSpec("input_weights", "embed.weight", "input_weights", np.dtype(np.int16)),
    IntTensorSpec("input_biases", "input_bias", "input_biases", np.dtype(np.int16)),
    IntTensorSpec("l1_weights", "l1_weight", "l1_weights", np.dtype(np.int8)),
    IntTensorSpec("l1_biases", "l1_bias", "l1_biases", np.dtype(np.int32)),
)

FLOAT_TENSORS = (
    FloatTensorSpec("l2_weights", "l2.weight", "l2_weights"),
    FloatTensorSpec("l2_biases", "l2.bias", "l2_biases"),
    FloatTensorSpec("output_weights", "output.weight", "output_weights"),
    FloatTensorSpec("output_bias", "output.bias", "output_bias"),
)


def load_state(path: str | Path) -> dict[str, np.ndarray]:
    raw = torch.load(Path(path).expanduser(), map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    if not isinstance(raw, dict):
        raise SystemExit(f"{path}: expected a PyTorch state_dict")
    state: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        if not torch.is_tensor(value):
            continue
        state[key] = value.detach().cpu().numpy().astype(np.float32, copy=False)
    return state


def round_clip(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    info = np.iinfo(dtype)
    rounded = np.rint(values)
    return np.clip(rounded, info.min, info.max).astype(dtype)


def reference_tensor(
    net,
    spec: IntTensorSpec | FloatTensorSpec,
    candidate_shape: tuple[int, ...],
) -> np.ndarray:
    values = getattr(net, spec.reference_attr)
    array = np.asarray(values, dtype=np.float32)
    if array.shape == candidate_shape:
        return array
    if array.ndim == len(candidate_shape) + 1 and array.shape[1:] == candidate_shape:
        return array[0]
    if candidate_shape == (1,) and array.ndim == 1 and array.size > 1:
        return array[:1]
    return array


def pct(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def int_tensor_report(
    name: str,
    candidate: np.ndarray,
    dtype: np.dtype,
    *,
    reference: np.ndarray | None,
    atol: float,
) -> dict[str, float | int | str]:
    candidate = np.asarray(candidate, dtype=np.float32)
    candidate_q = round_clip(candidate, dtype)
    total = int(candidate.size)
    report: dict[str, float | int | str] = {"name": name, "total": total}

    if reference is None:
        nearest = np.rint(candidate)
        frac_distance = np.abs(candidate - nearest)
        boundary_remaining = np.maximum(0.0, 0.5 - frac_distance)
        exported_nonzero = int(np.count_nonzero(candidate_q))
        near_005 = int(np.count_nonzero(boundary_remaining <= 0.05))
        near_010 = int(np.count_nonzero(boundary_remaining <= 0.10))
        report.update({
            "exported_nonzero": exported_nonzero,
            "exported_nonzero_pct": pct(exported_nonzero, total),
            "near_boundary_0p05": near_005,
            "near_boundary_0p10": near_010,
            "near_boundary_0p05_pct": pct(near_005, total),
            "near_boundary_0p10_pct": pct(near_010, total),
        })
        report.update({f"abs_value_{k}": v for k, v in quantiles(np.abs(candidate)).items()})
        return report

    reference = np.asarray(reference, dtype=np.float32)
    if candidate.shape != reference.shape:
        raise SystemExit(
            f"{name}: shape mismatch state={candidate.shape} reference={reference.shape}")

    reference_q = round_clip(reference, dtype)
    delta = candidate - reference
    abs_delta = np.abs(delta)
    float_changed_mask = abs_delta > atol
    exported_changed_mask = candidate_q != reference_q
    changed_values = abs_delta[float_changed_mask]
    reference_center = reference_q.astype(np.float32)
    boundary_remaining = np.maximum(0.0, 0.5 - np.abs(candidate - reference_center))
    blocked_mask = float_changed_mask & ~exported_changed_mask

    float_changed = int(np.count_nonzero(float_changed_mask))
    exported_changed = int(np.count_nonzero(exported_changed_mask))
    blocked = int(np.count_nonzero(blocked_mask))
    blocked_near_005 = int(np.count_nonzero(blocked_mask & (boundary_remaining <= 0.05)))
    blocked_near_010 = int(np.count_nonzero(blocked_mask & (boundary_remaining <= 0.10)))

    report.update({
        "float_changed": float_changed,
        "float_changed_pct": pct(float_changed, total),
        "exported_changed": exported_changed,
        "exported_changed_pct": pct(exported_changed, total),
        "blocked_changed": blocked,
        "blocked_changed_pct": pct(blocked, total),
        "blocked_near_boundary_0p05": blocked_near_005,
        "blocked_near_boundary_0p10": blocked_near_010,
        "blocked_near_boundary_0p05_pct": pct(blocked_near_005, total),
        "blocked_near_boundary_0p10_pct": pct(blocked_near_010, total),
    })
    report.update({f"abs_delta_{k}": v for k, v in quantiles(changed_values).items()})
    return report


def float_tensor_report(
    name: str,
    candidate: np.ndarray,
    *,
    reference: np.ndarray | None,
    atol: float,
) -> dict[str, float | int | str]:
    candidate = np.asarray(candidate, dtype=np.float32).reshape(-1)
    total = int(candidate.size)
    report: dict[str, float | int | str] = {"name": name, "total": total}
    if reference is None:
        report.update({f"abs_value_{k}": v for k, v in quantiles(np.abs(candidate)).items()})
        return report

    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    if candidate.shape != reference.shape:
        raise SystemExit(
            f"{name}: shape mismatch state={candidate.shape} reference={reference.shape}")
    abs_delta = np.abs(candidate - reference)
    changed = int(np.count_nonzero(abs_delta > atol))
    report.update({
        "changed": changed,
        "changed_pct": pct(changed, total),
    })
    report.update({f"abs_delta_{k}": v for k, v in quantiles(abs_delta[abs_delta > atol]).items()})
    return report


def print_report(report: list[dict[str, float | int | str]]) -> None:
    for row in report:
        name = str(row["name"])
        total = int(row["total"])
        if "exported_changed" in row:
            print(
                f"{name:14s} exported_changed={row['exported_changed']}/{total} "
                f"({row['exported_changed_pct']:.3f}%) "
                f"float_changed={row['float_changed']}/{total} "
                f"blocked={row['blocked_changed']}/{total} "
                f"near_blocked_0.05={row['blocked_near_boundary_0p05']} "
                f"abs_delta_p95={row['abs_delta_p95']:.6g} "
                f"abs_delta_max={row['abs_delta_max']:.6g}")
        elif "exported_nonzero" in row:
            print(
                f"{name:14s} exported_nonzero={row['exported_nonzero']}/{total} "
                f"({row['exported_nonzero_pct']:.3f}%) "
                f"near_boundary_0.05={row['near_boundary_0p05']} "
                f"abs_value_p95={row['abs_value_p95']:.6g} "
                f"abs_value_max={row['abs_value_max']:.6g}")
        else:
            print(
                f"{name:14s} changed={row.get('changed', 0)}/{total} "
                f"({row.get('changed_pct', 0.0):.3f}%) "
                f"abs_delta_p95={row.get('abs_delta_p95', row.get('abs_value_p95', 0.0)):.6g} "
                f"abs_delta_max={row.get('abs_delta_max', row.get('abs_value_max', 0.0)):.6g}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, help="PyTorch state_dict checkpoint")
    parser.add_argument("--reference", help="Reference exported .nn")
    parser.add_argument("--json", help="Optional JSON report path")
    parser.add_argument("--float-atol", type=float, default=0.0)
    parser.add_argument("--fail-if-no-sparse-export-movement", action="store_true")
    parser.add_argument("--min-input-exported-changed", type=int, default=0)
    parser.add_argument("--min-l1-exported-changed", type=int, default=0)
    args = parser.parse_args()

    state = load_state(args.state)
    reference = load_net(args.reference) if args.reference else None
    report: list[dict[str, float | int | str]] = []

    for spec in INT_TENSORS:
        if spec.state_key not in state:
            raise SystemExit(f"{args.state}: missing state tensor {spec.state_key}")
        candidate = state[spec.state_key]
        ref = (
            reference_tensor(reference, spec, candidate.shape)
            if reference is not None else None
        )
        report.append(
            int_tensor_report(
                spec.name,
                candidate,
                spec.dtype,
                reference=ref,
                atol=args.float_atol,
            ))

    for spec in FLOAT_TENSORS:
        if spec.state_key not in state:
            raise SystemExit(f"{args.state}: missing state tensor {spec.state_key}")
        candidate = state[spec.state_key]
        ref = (
            reference_tensor(reference, spec, candidate.shape)
            if reference is not None else None
        )
        report.append(
            float_tensor_report(
                spec.name,
                candidate,
                reference=ref,
                atol=args.float_atol,
            ))

    print_report(report)

    if args.json:
        path = Path(args.json).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    by_name = {str(row["name"]): row for row in report}
    input_changed = (
        int(by_name["input_weights"].get("exported_changed", 0))
        + int(by_name["input_biases"].get("exported_changed", 0))
    )
    l1_changed = (
        int(by_name["l1_weights"].get("exported_changed", 0))
        + int(by_name["l1_biases"].get("exported_changed", 0))
    )
    if args.fail_if_no_sparse_export_movement and input_changed + l1_changed == 0:
        return 1
    if input_changed < args.min_input_exported_changed:
        return 1
    if l1_changed < args.min_l1_exported_changed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
