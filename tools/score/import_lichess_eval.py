#!/usr/bin/env python3
"""Convert filtered Lichess eval DB rows to Enyo NNUE JSONL rows."""

from __future__ import annotations

import argparse
import bz2
import gzip
import io
import json
import lzma
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any


@dataclass
class Bucket:
    name: str
    sign: str
    lo: float
    hi: float
    rows: int
    seen: int = 0
    accepted: int = 0
    reservoir: list[str] = field(default_factory=list)

    def contains(self, score: float) -> bool:
        if self.sign == "pos" and score <= 0:
            return False
        if self.sign == "neg" and score >= 0:
            return False
        if self.sign == "zero" and score != 0:
            return False
        abs_score = abs(score)
        return self.lo <= abs_score < self.hi


def parse_hi(value: str) -> float:
    if value in ("inf", "+inf"):
        return float("inf")
    return float(value)


def parse_bucket(spec: str) -> Bucket:
    parts = spec.split(":")
    if len(parts) != 5:
        raise ValueError(
            "bucket must be NAME:SIGN:LO:HI:ROWS, "
            f"where SIGN is any|pos|neg|zero, got {spec}")
    name, sign, lo, hi, rows = parts
    if sign not in ("any", "pos", "neg", "zero"):
        raise ValueError(f"invalid bucket sign '{sign}' in {spec}")
    return Bucket(name=name, sign=sign, lo=float(lo),
                  hi=parse_hi(hi), rows=int(rows))


def maybe_take(bucket: Bucket, line: str, rng: random.Random) -> None:
    bucket.accepted += 1
    if len(bucket.reservoir) < bucket.rows:
        bucket.reservoir.append(line)
        return
    idx = rng.randrange(bucket.accepted)
    if idx < bucket.rows:
        bucket.reservoir[idx] = line


def open_text(path: Path) -> IO[str]:
    suffixes = path.suffixes
    if suffixes and suffixes[-1] == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    if suffixes and suffixes[-1] == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    if suffixes and suffixes[-1] in (".xz", ".lzma"):
        return lzma.open(path, "rt", encoding="utf-8", errors="replace")
    if suffixes and suffixes[-1] == ".zst":
        try:
            import zstandard as zstd  # type: ignore
        except ImportError:
            proc = subprocess.Popen(
                ["zstdcat", str(path)],
                stdout=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if proc.stdout is None:
                raise RuntimeError("zstdcat did not provide stdout")
            return proc.stdout
        dctx = zstd.ZstdDecompressor()
        raw = path.open("rb")
        stream = dctx.stream_reader(raw)
        return io.TextIOWrapper(stream, encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def pick_eval(row: dict[str, Any], *, min_depth: int,
              min_knodes: int) -> dict[str, Any] | None:
    evals = row.get("evals")
    if not isinstance(evals, list):
        return None

    best = None
    best_key = (-1, -1)
    for entry in evals:
        if not isinstance(entry, dict):
            continue
        depth = int(entry.get("depth") or 0)
        knodes = int(entry.get("knodes") or 0)
        if depth < min_depth or knodes < min_knodes:
            continue
        key = (depth, knodes)
        if key > best_key:
            best = entry
            best_key = key
    return best


def score_from_eval(entry: dict[str, Any]) -> int | None:
    pvs = entry.get("pvs")
    if not isinstance(pvs, list) or not pvs:
        return None
    first = pvs[0]
    if not isinstance(first, dict):
        return None
    cp = first.get("cp")
    if cp is None:
        return None
    return int(cp)


def side_to_move_score(fen: str, white_pov_score: int) -> int:
    parts = fen.split()
    if len(parts) < 2:
        raise ValueError(f"invalid FEN: {fen}")
    return white_pov_score if parts[1] == "w" else -white_pov_score


def valid_standard_material(fen: str) -> bool:
    parts = fen.split()
    if len(parts) < 2 or parts[1] not in ("w", "b"):
        return False
    ranks = parts[0].split("/")
    if len(ranks) != 8:
        return False

    white = 0
    black = 0
    white_kings = 0
    black_kings = 0
    for rank in ranks:
        files = 0
        for ch in rank:
            if ch.isdigit():
                files += int(ch)
                continue
            if ch.lower() not in "pnbrqk":
                return False
            files += 1
            if ch.isupper():
                white += 1
                white_kings += ch == "K"
            else:
                black += 1
                black_kings += ch == "k"
        if files != 8:
            return False

    return white <= 16 and black <= 16 and white_kings == 1 and black_kings == 1


def result_wdl(score: int, scale: float) -> float:
    # Lichess eval rows do not carry game result. Use the cp target as the
    # neutral WDL proxy, so WDL mixing does not add unrelated game-result noise.
    import math
    return 1.0 / (1.0 + math.exp(-score / scale))


def phase_scale_from_fen(fen: str) -> float:
    board = fen.split()[0]
    minors = sum(1 for ch in board if ch in "NnBb")
    rooks = sum(1 for ch in board if ch in "Rr")
    queens = sum(1 for ch in board if ch in "Qq")
    phase = 3 * minors + 5 * rooks + 10 * queens
    return (128.0 + float(phase)) / 128.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--rows", type=int, default=0)
    ap.add_argument("--min-depth", type=int, default=18)
    ap.add_argument("--min-knodes", type=int, default=0)
    ap.add_argument("--max-abs-cp", type=int, default=1600)
    ap.add_argument("--wdl-scale", type=float, default=400.0)
    ap.add_argument("--unique-fen", action="store_true")
    ap.add_argument("--bucket", action="append",
                    help="Optional signed bucket spec NAME:SIGN:LO:HI:ROWS. "
                         "When present, rows are reservoir-sampled while "
                         "streaming instead of written sequentially.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-input-rows", type=int, default=0,
                    help="Stop after this many input rows. Default scans all.")
    ap.add_argument("--stop-when-full", action="store_true",
                    help="Stop once all bucket reservoirs are full. Faster, "
                         "but biased toward earlier input rows.")
    ap.add_argument("--progress", type=int, default=100000)
    ap.add_argument("--output-format", choices=("jsonl", "bullet-text"),
                    default="jsonl")
    ap.add_argument("--enyo-runtime-target", action="store_true",
                    help="For bullet-text output, pre-divide cp labels by "
                         "Enyo's runtime phase scale.")
    args = ap.parse_args()
    output_format = str(args.output_format)
    if args.bucket and args.rows:
        raise SystemExit("--rows is only valid without --bucket; put row "
                         "counts in the bucket specs")

    src = Path(args.input).expanduser()
    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    buckets = [parse_bucket(spec) for spec in args.bucket or []]
    rng = random.Random(args.seed)

    stats = {
        "input": str(src),
        "output": str(out),
        "seen": 0,
        "written": 0,
        "skipped_no_eval": 0,
        "skipped_invalid_fen": 0,
        "skipped_no_cp": 0,
        "skipped_extreme": 0,
        "skipped_duplicate": 0,
        "min_depth": args.min_depth,
        "min_knodes": args.min_knodes,
        "max_abs_cp": args.max_abs_cp,
        "unique_fen": args.unique_fen,
        "bucket_mode": bool(buckets),
        "seed": args.seed,
        "eligible": 0,
        "skipped_no_bucket": 0,
        "output_format": output_format,
        "enyo_runtime_target": bool(args.enyo_runtime_target),
    }
    seen_fens: set[str] = set()

    writer = None if buckets else out.open("w", encoding="utf-8")
    try:
        with open_text(src) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                stats["seen"] += 1
                if (args.max_input_rows > 0
                        and stats["seen"] > args.max_input_rows):
                    break
                row = json.loads(line)
                fen = row.get("fen")
                if not isinstance(fen, str):
                    stats["skipped_no_eval"] += 1
                    continue
                if not valid_standard_material(fen):
                    stats["skipped_invalid_fen"] += 1
                    continue
                if args.unique_fen:
                    if fen in seen_fens:
                        stats["skipped_duplicate"] += 1
                        continue
                    seen_fens.add(fen)

                entry = pick_eval(row, min_depth=args.min_depth,
                                  min_knodes=args.min_knodes)
                if entry is None:
                    stats["skipped_no_eval"] += 1
                    continue

                white_score = score_from_eval(entry)
                if white_score is None:
                    stats["skipped_no_cp"] += 1
                    continue
                score = side_to_move_score(fen, white_score)
                if abs(score) > args.max_abs_cp:
                    stats["skipped_extreme"] += 1
                    continue

                out_row = {
                    "fen": fen,
                    "score": score,
                    "wdl": result_wdl(score, args.wdl_scale),
                    "white_pov_score": white_score,
                    "source": str(src),
                    "source_type": "lichess_eval",
                    "teacher": "lichess_eval_db",
                    "teacher_depth": int(entry.get("depth") or 0),
                    "teacher_knodes": int(entry.get("knodes") or 0),
                }
                if output_format == "bullet-text":
                    bullet_score = white_score
                    if args.enyo_runtime_target:
                        bullet_score = int(round(
                            bullet_score / phase_scale_from_fen(fen)))
                    if white_score > 0:
                        bullet_result = 1.0
                    elif white_score < 0:
                        bullet_result = 0.0
                    else:
                        bullet_result = 0.5
                    out_line = f"{fen} | {bullet_score} | {bullet_result:.1f}"
                else:
                    out_line = json.dumps(out_row, separators=(",", ":"))
                stats["eligible"] += 1

                if buckets:
                    matched = False
                    for bucket in buckets:
                        if bucket.contains(score):
                            bucket.seen += 1
                            maybe_take(bucket, out_line, rng)
                            matched = True
                            break
                    if not matched:
                        stats["skipped_no_bucket"] += 1
                    if (args.stop_when_full
                            and all(len(b.reservoir) >= b.rows
                                    for b in buckets)):
                        break
                else:
                    assert writer is not None
                    writer.write(out_line)
                    writer.write("\n")
                    stats["written"] += 1

                if args.progress > 0 and stats["seen"] % args.progress == 0:
                    progress = dict(stats)
                    if buckets:
                        progress["buckets"] = [
                            {
                                "name": b.name,
                                "seen": b.seen,
                                "written": len(b.reservoir),
                                "requested": b.rows,
                            }
                            for b in buckets
                        ]
                    print(json.dumps(progress), flush=True)
                if (not buckets and args.rows > 0
                        and stats["written"] >= args.rows):
                    break
    finally:
        if writer is not None:
            writer.close()

    if buckets:
        selected: list[str] = []
        for bucket in buckets:
            selected.extend(bucket.reservoir)
        rng.shuffle(selected)
        with out.open("w", encoding="utf-8") as writer:
            for line in selected:
                writer.write(line)
                writer.write("\n")
        stats["written"] = len(selected)
        stats["buckets"] = [
            {
                "name": bucket.name,
                "sign": bucket.sign,
                "lo": bucket.lo,
                "hi": bucket.hi,
                "requested": bucket.rows,
                "seen": bucket.seen,
                "written": len(bucket.reservoir),
            }
            for bucket in buckets
        ]

    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(f"missing dependency or file: {exc}") from exc
