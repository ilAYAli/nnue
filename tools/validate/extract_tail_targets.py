#!/usr/bin/env python3
"""Extract repeated tail-regression positions from replay failure-suite CSVs."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


FEN_RE = re.compile(r"search_position start: fen=(.*?), legal_moves=")
POSITION_STARTPOS_RE = re.compile(r"^position startpos(?: moves (.*))?$")
POSITION_FEN_RE = re.compile(r"^position fen (.*?)(?: moves (.*))?$")


def canonical_log_name(value: str) -> str:
    stem = Path(value).stem
    return "".join(ch.lower() for ch in stem if ch.isalnum())


def int_field(row: dict[str, str], key: str) -> int:
    try:
        return int(row.get(key, "0"))
    except ValueError:
        return 0


def parse_named_csv(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "CSV inputs must use NAME=PATH, for example kingbucket=out.csv")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("CSV input name is empty")
    return name, Path(path).expanduser()


@dataclass
class Hit:
    candidate_name: str
    log_name: str
    fullmove: str
    ply: str
    side: str
    candidate_move: str
    reference_move: str
    oracle_best: str
    diff: int
    candidate_loss: str
    reference_loss: str


def find_bug_logs(bug_dir: Path) -> dict[str, Path]:
    logs: dict[str, Path] = {}
    for path in bug_dir.expanduser().glob("*.log"):
        logs[canonical_log_name(path.name)] = path
    return logs


def fen_map_for_log(path: Path) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return result
    try:
        import chess  # type: ignore
    except ImportError:
        chess = None
    for line in lines:
        line = line.strip()
        if line.startswith(">> "):
            line = line[3:]
        match = FEN_RE.search(line)
        if match:
            fen = match.group(1)
            parts = fen.split()
            if len(parts) < 6:
                continue
            side = "White" if parts[1] == "w" else "Black"
            fullmove = parts[5]
            result.setdefault((fullmove, side), fen)
            continue
        if chess is None:
            continue
        match = POSITION_STARTPOS_RE.match(line)
        if match:
            board = chess.Board()
            moves = match.group(1)
        else:
            match = POSITION_FEN_RE.match(line)
            if not match:
                continue
            board = chess.Board(match.group(1))
            moves = match.group(2)
        if moves:
            try:
                for move in moves.split():
                    board.push_uci(move)
            except ValueError:
                continue
        side = "White" if board.turn == chess.WHITE else "Black"
        result.setdefault((str(board.fullmove_number), side), board.fen())
    return result


def read_hits(
        csv_inputs: list[tuple[str, Path]], max_diff: int) -> dict[
            tuple[str, str, str], list[Hit]]:
    groups: dict[tuple[str, str, str], list[Hit]] = defaultdict(list)
    for candidate_name, path in csv_inputs:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                diff = int_field(row, "diff")
                if diff > max_diff:
                    continue
                log_name = row.get("log", "")
                fullmove = row.get("fullmove", "")
                side = row.get("side", "")
                key = (canonical_log_name(log_name), fullmove, side)
                groups[key].append(Hit(
                    candidate_name=candidate_name,
                    log_name=Path(log_name).name,
                    fullmove=fullmove,
                    ply=row.get("ply", ""),
                    side=side,
                    candidate_move=row.get("candidate_move", ""),
                    reference_move=row.get("reference_move", ""),
                    oracle_best=row.get("oracle_best", ""),
                    diff=diff,
                    candidate_loss=row.get("candidate_loss", ""),
                    reference_loss=row.get("reference_loss", ""),
                ))
    return groups


def write_targets(
        groups: dict[tuple[str, str, str], list[Hit]], bug_dir: Path,
        output_path: Path, min_hits: int) -> int:
    bug_logs = find_bug_logs(bug_dir)
    fen_cache: dict[str, dict[tuple[str, str], str]] = {}
    fieldnames = [
        "log",
        "fullmove",
        "ply",
        "side",
        "fen",
        "hits",
        "worst_diff",
        "candidates",
        "candidate_moves",
        "reference_moves",
        "oracle_moves",
        "candidate_losses",
        "reference_losses",
    ]
    rows: list[dict[str, str]] = []
    for key, hits in groups.items():
        if len(hits) < min_hits:
            continue
        canonical_name, fullmove, side = key
        bug_log = bug_logs.get(canonical_name)
        fen = ""
        if bug_log:
            fen_cache.setdefault(canonical_name, fen_map_for_log(bug_log))
            fen = fen_cache[canonical_name].get((fullmove, side), "")
        rows.append({
            "log": hits[0].log_name,
            "fullmove": fullmove,
            "ply": hits[0].ply,
            "side": side,
            "fen": fen,
            "hits": str(len(hits)),
            "worst_diff": str(min(hit.diff for hit in hits)),
            "candidates": ";".join(hit.candidate_name for hit in hits),
            "candidate_moves": ";".join(hit.candidate_move for hit in hits),
            "reference_moves": ";".join(hit.reference_move for hit in hits),
            "oracle_moves": ";".join(hit.oracle_best for hit in hits),
            "candidate_losses": ";".join(hit.candidate_loss for hit in hits),
            "reference_losses": ";".join(hit.reference_loss for hit in hits),
        })
    rows.sort(key=lambda row: (int(row["worst_diff"]), row["log"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a small target set from positions that caused repeated "
            "large failure-suite regressions."))
    parser.add_argument(
        "--csv", action="append", required=True, type=parse_named_csv,
        help="Candidate replay CSV in NAME=PATH form. Can be repeated.")
    parser.add_argument("--bug-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--max-diff", type=int, default=-100,
        help="Only count rows with diff <= this value.")
    parser.add_argument(
        "--min-hits", type=int, default=2,
        help="Only emit positions that occur in at least this many CSVs.")
    args = parser.parse_args()

    missing = [str(path) for _, path in args.csv if not path.exists()]
    if missing:
        for path in missing:
            print(f"missing CSV: {path}", file=sys.stderr)
        return 2
    groups = read_hits(args.csv, args.max_diff)
    count = write_targets(
        groups, args.bug_dir.expanduser(), args.out.expanduser(),
        args.min_hits)
    print(f"wrote {count} targets to {args.out.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
