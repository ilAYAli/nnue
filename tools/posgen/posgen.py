#!/usr/bin/env python3
"""Generate and select Enyo NNUE training positions.

This tool is intentionally limited to the position-generation side of the
pipeline. Teacher/oracle labeling remains separate.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCORE_RE = re.compile(
    r"(?P<score>[+-]?(?:M\d+|\d+(?:\.\d+)?))/(?P<depth>\d+)"
    r"(?:\s+(?P<time>\d+(?:\.\d+)?)s)?"
)


SIGNED_BALANCED_V1 = (
    "any0:any:0:25:400000",
    "p25:pos:25:75:250000",
    "n25:neg:25:75:250000",
    "p75:pos:75:150:300000",
    "n75:neg:75:150:300000",
    "p150:pos:150:300:350000",
    "n150:neg:150:300:350000",
    "p300:pos:300:600:250000",
    "n300:neg:300:600:250000",
    "p600:pos:600:1200:150000",
    "n600:neg:600:1200:150000",
)


def expand_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def script_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_command(command: list[str]) -> int:
    print(" ".join(command), flush=True)
    proc = subprocess.Popen(command)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        raise


def parse_score(comment: str, mate_score_cp: int) -> dict[str, Any] | None:
    match = SCORE_RE.search(comment)
    if not match:
        return None

    raw = match.group("score")
    depth = int(match.group("depth"))
    time_s = float(match.group("time")) if match.group("time") else None

    if "M" in raw:
        sign = -1 if raw.startswith("-") else 1
        mate_ply = int(raw.lstrip("+-")[1:])
        return {
            "score_cp": sign * mate_score_cp,
            "mate_ply": sign * mate_ply,
            "depth": depth,
            "time_s": time_s,
            "raw_score": raw,
        }

    return {
        "score_cp": int(round(float(raw) * 100.0)),
        "mate_ply": None,
        "depth": depth,
        "time_s": time_s,
        "raw_score": raw,
    }


def wdl_for_side(result: str, turn: Any) -> float | None:
    import chess

    if result == "1/2-1/2":
        return 0.5
    if result == "1-0":
        return 1.0 if turn == chess.WHITE else 0.0
    if result == "0-1":
        return 1.0 if turn == chess.BLACK else 0.0
    return None


def result_reason(headers: Any) -> str:
    return headers.get("Termination") or headers.get("Result", "*")


def extract_rows(
    pgn_path: Path,
    *,
    skip_plies: int,
    min_depth: int,
    max_abs_cp: int,
    include_mates: bool,
    mate_score_cp: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    try:
        import chess
        import chess.pgn
    except ImportError as exc:
        raise SystemExit(
            "python-chess is required: python3 -m pip install chess"
        ) from exc

    rows: list[dict[str, Any]] = []
    stats = {
        "games": 0,
        "rows": 0,
        "skipped_no_score": 0,
        "skipped_depth": 0,
        "skipped_mate": 0,
        "skipped_cp": 0,
        "skipped_unknown_result": 0,
    }

    with pgn_path.open(encoding="utf-8", errors="replace") as handle:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break

            stats["games"] += 1
            result = game.headers.get("Result", "*")
            game_wdl_known = result in {"1-0", "0-1", "1/2-1/2"}
            board = game.board()
            ply = 0
            game_id = (
                game.headers.get("GameId")
                or game.headers.get("Round")
                or str(stats["games"])
            )

            for node in game.mainline():
                move = node.move
                if move is None:
                    continue

                turn = board.turn
                score = parse_score(node.comment, mate_score_cp)
                wdl = wdl_for_side(result, turn)

                if ply >= skip_plies:
                    if score is None:
                        stats["skipped_no_score"] += 1
                    elif score["depth"] < min_depth:
                        stats["skipped_depth"] += 1
                    elif score["mate_ply"] is not None and not include_mates:
                        stats["skipped_mate"] += 1
                    elif abs(score["score_cp"]) > max_abs_cp:
                        stats["skipped_cp"] += 1
                    elif not game_wdl_known or wdl is None:
                        stats["skipped_unknown_result"] += 1
                    else:
                        rows.append(
                            {
                                "fen": board.fen(en_passant="fen"),
                                "move": board.san(move),
                                "move_uci": move.uci(),
                                "score": score["score_cp"],
                                "wdl": wdl,
                                "result": result,
                                "side": "white" if turn == chess.WHITE else "black",
                                "ply": ply,
                                "fullmove": board.fullmove_number,
                                "depth": score["depth"],
                                "time_s": score["time_s"],
                                "mate_ply": score["mate_ply"],
                                "game_id": game_id,
                                "termination": result_reason(game.headers),
                                "source": str(pgn_path),
                            }
                        )
                        stats["rows"] += 1

                board.push(move)
                ply += 1

    return rows, stats


def cmd_extract(args: argparse.Namespace) -> int:
    pgn = expand_path(args.pgn)
    output = expand_path(args.output)
    rows, stats = extract_rows(
        pgn,
        skip_plies=args.skip_plies,
        min_depth=args.min_depth,
        max_abs_cp=args.max_abs_cp,
        include_mates=args.include_mates,
        mate_score_cp=args.mate_score_cp,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, separators=(",", ":")) + "\n")

    payload = {"input": str(pgn), "output": str(output), **stats}
    if args.stats:
        stats_path = expand_path(args.stats)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


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
    if value in {"inf", "+inf"}:
        return float("inf")
    return float(value)


def parse_bucket(spec: str) -> Bucket:
    parts = spec.split(":")
    if len(parts) != 5:
        raise ValueError(
            "bucket must be NAME:SIGN:LO:HI:ROWS, "
            f"where SIGN is any|pos|neg|zero, got {spec}"
        )
    name, sign, lo, hi, rows = parts
    if sign not in {"any", "pos", "neg", "zero"}:
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


def bucket_specs(args: argparse.Namespace) -> list[str]:
    specs: list[str] = []
    if args.preset == "signed-balanced-v1":
        specs.extend(SIGNED_BALANCED_V1)
    elif args.preset != "none":
        raise ValueError(f"unknown preset: {args.preset}")
    specs.extend(args.bucket or [])
    if not specs:
        raise ValueError("sample requires --preset or at least one --bucket")
    return specs


def cmd_sample(args: argparse.Namespace) -> int:
    buckets = [parse_bucket(spec) for spec in bucket_specs(args)]
    rng = random.Random(args.seed)
    seen_fens: set[str] = set()
    total_seen = 0
    total_used = 0
    skipped_duplicate = 0
    skipped_missing_score = 0
    skipped_no_bucket = 0

    input_path = expand_path(args.input)
    output_path = expand_path(args.output)
    with input_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total_seen += 1
            row = json.loads(line)

            if args.unique_fen:
                fen = row.get("fen")
                if not isinstance(fen, str):
                    continue
                if fen in seen_fens:
                    skipped_duplicate += 1
                    continue
                seen_fens.add(fen)

            if args.score_field not in row:
                skipped_missing_score += 1
                continue
            score = float(row[args.score_field])

            matched = False
            for bucket in buckets:
                if bucket.contains(score):
                    bucket.seen += 1
                    maybe_take(bucket, line, rng)
                    total_used += 1
                    matched = True
                    break
            if not matched:
                skipped_no_bucket += 1

            if args.progress > 0 and total_seen % args.progress == 0:
                print(
                    f"seen={total_seen} used={total_used} "
                    + " ".join(
                        f"{b.name}:{len(b.reservoir)}/{b.rows}"
                        for b in buckets),
                    flush=True,
                )

    selected: list[str] = []
    for bucket in buckets:
        selected.extend(bucket.reservoir)
    rng.shuffle(selected)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for line in selected:
            handle.write(line)
            handle.write("\n")

    stats = {
        "input": str(input_path),
        "output": str(output_path),
        "score_field": args.score_field,
        "seed": args.seed,
        "unique_fen": args.unique_fen,
        "seen": total_seen,
        "written": len(selected),
        "skipped_duplicate": skipped_duplicate,
        "skipped_missing_score": skipped_missing_score,
        "skipped_no_bucket": skipped_no_bucket,
        "buckets": [
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
        ],
    }
    output_path.with_suffix(output_path.suffix + ".stats.json").write_text(
        json.dumps(stats, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2), flush=True)
    return 0


def cmd_selfplay(args: argparse.Namespace) -> int:
    runner_script = expand_path(args.runner_script)
    if not runner_script.exists():
        raise SystemExit(f"selfplay runner not found: {runner_script}")

    command = [
        str(runner_script),
        "--engine", str(expand_path(args.engine)),
        "--output", str(expand_path(args.output)),
        "--games", str(args.games),
        "--shard-games", str(args.shard_games),
        "--concurrency", str(args.concurrency),
        "--threads", str(args.threads),
        "--book", str(expand_path(args.book)),
        "--book-format", args.book_format,
        "--order", args.order,
        "--restart", args.restart,
    ]
    if args.runner:
        command += ["--runner", str(expand_path(args.runner))]
    if args.nnue_file:
        command += ["--nnue-file", str(expand_path(args.nnue_file))]
    if args.tc:
        command += ["--tc", args.tc]
    else:
        command += ["--depth", str(args.depth)]
    if args.srand:
        command += ["--srand", str(args.srand)]
    for option in args.engine_option or []:
        command += ["--engine-option", option]

    return run_command(command)


def cmd_instability(args: argparse.Namespace) -> int:
    sampler = script_root() / "posgen" / "sample_search_instability.py"
    if not sampler.exists():
        raise SystemExit(f"instability sampler not found: {sampler}")

    command = [
        sys.executable,
        str(sampler),
        "--input", str(expand_path(args.input)),
        "--output", str(expand_path(args.output)),
        "--engine", args.engine,
        "--low-depth", str(args.low_depth),
        "--high-depth", str(args.high_depth),
        "--threads", str(args.threads),
        "--hash", str(args.hash),
        "--shard-count", str(args.shard_count),
        "--shard-index", str(args.shard_index),
        "--max-abs-cp", str(args.max_abs_cp),
        "--min-delta-cp", str(args.min_delta_cp),
        "--max-output", str(args.max_output),
        "--seed", str(args.seed),
        "--limit", str(args.limit),
        "--progress", str(args.progress),
        "--position-timeout-s", str(args.position_timeout_s),
    ]
    if args.bestmove_change:
        command.append("--bestmove-change")
    return run_command(command)


def add_selfplay_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("selfplay", help="Generate self-play PGN with fastchess.")
    parser.add_argument("--runner-script", default=str(script_root() / "posgen" / "run_selfplay.sh"))
    parser.add_argument("--runner")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--nnue-file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--shard-games", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--tc")
    parser.add_argument("--book", required=True)
    parser.add_argument("--book-format", default="epd")
    parser.add_argument("--order", default="random")
    parser.add_argument("--srand")
    parser.add_argument("--restart", choices=("auto", "on", "off"), default="off")
    parser.add_argument("--engine-option", action="append")
    parser.set_defaults(func=cmd_selfplay)


def add_extract_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("extract", help="Extract JSONL rows from annotated PGN.")
    parser.add_argument("pgn")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--stats")
    parser.add_argument("--skip-plies", type=int, default=8)
    parser.add_argument("--min-depth", type=int, default=1)
    parser.add_argument("--max-abs-cp", type=int, default=10000)
    parser.add_argument("--include-mates", action="store_true")
    parser.add_argument("--mate-score-cp", type=int, default=10000)
    parser.set_defaults(func=cmd_extract)


def add_sample_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("sample", help="Sample JSONL rows into score buckets.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preset", default="none",
                        help="Known preset, currently: signed-balanced-v1.")
    parser.add_argument("--bucket", action="append",
                        help="Bucket spec NAME:SIGN:LO:HI:ROWS.")
    parser.add_argument("--score-field", default="score")
    parser.add_argument("--unique-fen", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--progress", type=int, default=1000000)
    parser.set_defaults(func=cmd_sample)


def add_instability_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "instability",
        help="Sample positions where a UCI teacher changes opinion between depths.",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--engine", default="stockfish")
    parser.add_argument("--low-depth", type=int, default=8)
    parser.add_argument("--high-depth", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=128)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-abs-cp", type=int, default=1600)
    parser.add_argument("--min-delta-cp", type=int, default=80)
    parser.add_argument("--bestmove-change", action="store_true")
    parser.add_argument("--max-output", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress", type=int, default=1000)
    parser.add_argument("--position-timeout-s", type=float, default=0)
    parser.set_defaults(func=cmd_instability)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate, extract, and sample Enyo NNUE training positions."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_selfplay_parser(subparsers)
    add_extract_parser(subparsers)
    add_sample_parser(subparsers)
    add_instability_parser(subparsers)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
