#!/usr/bin/env python3
"""Attach cp/WDL/search targets to position rows."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def expand_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def path_or_command(value: str) -> str:
    if "/" not in value and not value.startswith("~"):
        return value
    return str(expand_path(value))


def tools_root() -> Path:
    return Path(__file__).resolve().parents[1]


def cmd_uci(args: argparse.Namespace) -> int:
    score_tools = tools_root() / "score"
    sys.path.insert(0, str(score_tools))
    import label_with_uci  # type: ignore

    forwarded = [
        "label_with_uci.py",
        "--input", str(expand_path(args.input)),
        "--output", str(expand_path(args.output)),
        "--engine", path_or_command(args.engine),
        "--depth", str(args.depth),
        "--threads", str(args.threads),
        "--hash", str(args.hash),
        "--shard-count", str(args.shard_count),
        "--shard-index", str(args.shard_index),
        "--limit", str(args.limit),
        "--max-abs-cp", str(args.max_abs_cp),
        "--progress", str(args.progress),
    ]
    old_argv = sys.argv
    try:
        sys.argv = forwarded
        label_with_uci.main()
    finally:
        sys.argv = old_argv
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score Enyo NNUE position rows with a selected source."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    uci = subparsers.add_parser("uci", help="Score rows with a UCI engine.")
    uci.add_argument("--input", required=True)
    uci.add_argument("--output", required=True)
    uci.add_argument("--engine", default="stockfish")
    uci.add_argument("--depth", type=int, default=12)
    uci.add_argument("--threads", type=int, default=1)
    uci.add_argument("--hash", type=int, default=128)
    uci.add_argument("--shard-count", type=int, default=1)
    uci.add_argument("--shard-index", type=int, default=0)
    uci.add_argument("--limit", type=int, default=0)
    uci.add_argument("--max-abs-cp", type=int, default=1600)
    uci.add_argument("--progress", type=int, default=1000)
    uci.set_defaults(func=cmd_uci)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
