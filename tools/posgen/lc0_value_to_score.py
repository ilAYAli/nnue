#!/usr/bin/env python3
"""Convert LCZero value labels into Enyo scored JSONL rows."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def q_to_cp(q_value: float, *, scale: float, clamp: float) -> int:
    q = max(-clamp, min(clamp, q_value))
    return int(round(scale * math.atanh(q)))


def append_tag(row: dict[str, Any], tag: str) -> None:
    tags = row.get("tags")
    if not isinstance(tags, list):
        tags = []
    if tag not in tags:
        tags.append(tag)
    row["tags"] = tags


def convert_row(row: dict[str, Any], *, q_field: str, scale: float,
                clamp: float, invert: bool) -> dict[str, Any]:
    lc0 = row.get("lc0")
    if not isinstance(lc0, dict):
        raise ValueError("missing lc0 object")
    q = float(lc0[q_field])
    if invert:
        q = -q

    score = q_to_cp(q, scale=scale, clamp=clamp)
    if "score" in row:
        row["source_score"] = row["score"]
    row["score"] = score
    row["wdl"] = max(0.0, min(1.0, (q + 1.0) * 0.5))
    row["teacher"] = f"lc0_{q_field}"
    row["teacher_q"] = q
    row["teacher_q_field"] = q_field
    row["teacher_cp_scale"] = scale
    row["teacher_cp_transform"] = "scale_atanh_q"
    append_tag(row, f"label:lc0_{q_field}")
    append_tag(row, "score:lc0_value")
    return row


def convert(input_path: Path, output_path: Path, *, q_field: str, scale: float,
            clamp: float, invert: bool, max_abs_cp: int, limit: int,
            progress: int) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "input": str(input_path),
        "output": str(output_path),
        "q_field": q_field,
        "scale": scale,
        "clamp": clamp,
        "invert": invert,
        "max_abs_cp": max_abs_cp,
        "limit": limit,
        "read": 0,
        "written": 0,
        "skipped_bad": 0,
        "skipped_cp": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8", errors="replace") as src, \
            output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            stats["read"] += 1
            try:
                row = convert_row(
                    json.loads(line),
                    q_field=q_field,
                    scale=scale,
                    clamp=clamp,
                    invert=invert,
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                stats["skipped_bad"] += 1
                continue
            if max_abs_cp > 0 and abs(int(row["score"])) > max_abs_cp:
                stats["skipped_cp"] += 1
                continue
            dst.write(json.dumps(row, separators=(",", ":")))
            dst.write("\n")
            stats["written"] += 1
            if progress > 0 and stats["written"] % progress == 0:
                print(f"converted {stats['written']}", flush=True)
            if limit > 0 and stats["written"] >= limit:
                break
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert LCZero root value labels into Enyo scored JSONL.")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--q-field", default="root_q")
    ap.add_argument("--scale", type=float, default=300.0)
    ap.add_argument("--clamp", type=float, default=0.999)
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--max-abs-cp", type=int, default=800)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress", type=int, default=100000)
    args = ap.parse_args()
    if not (0.0 < args.clamp < 1.0):
        raise SystemExit("--clamp must be between 0 and 1")
    stats = convert(
        args.input.expanduser(),
        args.output.expanduser(),
        q_field=args.q_field,
        scale=args.scale,
        clamp=args.clamp,
        invert=args.invert,
        max_abs_cp=args.max_abs_cp,
        limit=args.limit,
        progress=args.progress,
    )
    stats_path = args.output.expanduser().with_suffix(
        args.output.expanduser().suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    print(json.dumps(stats, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
