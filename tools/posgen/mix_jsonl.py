#!/usr/bin/env python3
"""Deterministically mix multiple JSONL datasets by per-file row quotas.

This script is designed for large training sets. It counts rows first, then
streams an exact-size deterministic sample from each source while interleaving
the selected rows, so it does not hold the mixed dataset in memory.

Source specs are either PATH:ROWS or PATH:ROWS:MAX_ABS_CP. The optional
MAX_ABS_CP filter is applied to row["score"] before sampling.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Source:
    path: Path
    requested: int
    total_rows: int
    eligible_rows: int
    selected_rows: int
    max_abs_cp: int
    written: int = 0


def keep_line(line: str, max_abs_cp: int) -> bool:
    if max_abs_cp <= 0:
        return True
    try:
        score = float(json.loads(line)["score"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return abs(score) <= max_abs_cp


def count_rows(path: Path, max_abs_cp: int) -> tuple[int, int]:
    rows = 0
    eligible = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows += 1
            if keep_line(line, max_abs_cp):
                eligible += 1
    return rows, eligible


def parse_source_spec(spec: str) -> tuple[Path, int, int]:
    try:
        path_s, rows_s, max_abs_cp_s = spec.rsplit(":", 2)
        requested = int(rows_s)
        max_abs_cp = int(max_abs_cp_s)
    except ValueError:
        path_s, rows_s = spec.rsplit(":", 1)
        requested = int(rows_s)
        max_abs_cp = 0
    if requested < 0:
        raise ValueError(f"{spec}: ROWS must be >= 0")
    if max_abs_cp < 0:
        raise ValueError(f"{spec}: MAX_ABS_CP must be >= 0")
    return Path(path_s).expanduser(), requested, max_abs_cp


def selected_lines(source: Source, rng: random.Random):
    remaining_available = source.eligible_rows
    remaining_needed = source.selected_rows
    with source.path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if not keep_line(line, source.max_abs_cp):
                continue
            if remaining_needed <= 0:
                break
            if rng.randrange(remaining_available) < remaining_needed:
                yield line
                remaining_needed -= 1
            remaining_available -= 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--source", action="append", required=True,
                    help=(
                        "Source spec PATH:ROWS or PATH:ROWS:MAX_ABS_CP. "
                        "Repeat for each dataset."))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--progress", type=int, default=250000)
    ap.add_argument("--unique-fen", action="store_true",
                    help="Rejected for large streaming mixes; dedupe inputs first.")
    args = ap.parse_args()
    if args.unique_fen:
        raise SystemExit(
            "--unique-fen is not supported by the streaming mixer; "
            "dedupe inputs before mixing")

    rng = random.Random(args.seed)
    stats = {
        "output": args.output,
        "seed": args.seed,
        "streaming": True,
        "sources": [],
    }

    sources: list[Source] = []
    for spec in args.source:
        try:
            path, requested, max_abs_cp = parse_source_spec(spec)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        total_rows, eligible_rows = count_rows(path, max_abs_cp)
        selected_rows = min(requested, eligible_rows)
        source = Source(path=path, requested=requested,
                        total_rows=total_rows, eligible_rows=eligible_rows,
                        selected_rows=selected_rows, max_abs_cp=max_abs_cp)
        sources.append(source)
        stats["sources"].append({
            "path": str(path),
            "requested": requested,
            "total_rows": total_rows,
            "eligible_rows": eligible_rows,
            "filtered_rows": total_rows - eligible_rows,
            "selected_rows": selected_rows,
            "max_abs_cp": max_abs_cp,
        })

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    select_rngs = [random.Random(rng.randrange(1 << 63)) for _ in sources]
    iterators = [
        selected_lines(source, source_rng)
        for source, source_rng in zip(sources, select_rngs)
    ]
    remaining = [source.selected_rows for source in sources]
    total_remaining = sum(remaining)
    written = 0

    with out.open("w", encoding="utf-8") as handle:
        while total_remaining > 0:
            pick = rng.randrange(total_remaining)
            cumulative = 0
            source_idx = 0
            for i, count in enumerate(remaining):
                cumulative += count
                if pick < cumulative:
                    source_idx = i
                    break

            line = next(iterators[source_idx])
            handle.write(line)
            handle.write("\n")
            sources[source_idx].written += 1
            remaining[source_idx] -= 1
            total_remaining -= 1
            written += 1
            if args.progress > 0 and written % args.progress == 0:
                print(f"mixed {written}", flush=True)

    for source, item in zip(sources, stats["sources"]):
        item["written"] = source.written

    stats["written_rows"] = written
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
