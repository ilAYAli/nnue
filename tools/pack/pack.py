#!/usr/bin/env python3
"""Pack scored position rows into NNUE training tensors."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def expand_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def expand_user(value: str | Path) -> Path:
    return Path(value).expanduser()


def tools_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run(command: list[str]) -> int:
    print(" ".join(command), flush=True)
    proc = subprocess.Popen(command)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        raise


def cmd_build(args: argparse.Namespace) -> int:
    script = tools_root() / "pack" / "pack_dataset.py"
    command = [
        str(expand_user(args.python)),
        str(script),
        "--input", str(expand_path(args.input)),
        "--out-dir", str(expand_path(args.out_dir)),
        "--skip", str(args.skip),
        "--limit", str(args.limit),
        "--max-features", str(args.max_features),
        "--progress", str(args.progress),
    ]
    if args.rows > 0:
        command.extend(["--rows", str(args.rows)])
    if args.rows_file:
        command.extend(["--rows-file", str(expand_path(args.rows_file))])
    return run(command)


def cmd_merge_shards(args: argparse.Namespace) -> int:
    script = tools_root() / "pack" / "pack_dataset.py"
    command = [
        str(expand_user(args.python)),
        str(script),
        "--packed-shard-dir", str(expand_path(args.input_dir)),
        "--packed-shard-pattern", args.pattern,
        "--out-dir", str(expand_path(args.out_dir)),
        "--max-features", str(args.max_features),
        "--progress", str(args.progress),
    ]
    return run(command)


def cmd_inspect(args: argparse.Namespace) -> int:
    path = expand_path(args.path)
    meta_path = path / "meta.json"
    source_map_path = path / "source_map.json"
    if not meta_path.exists():
        raise SystemExit(f"missing {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    print(json.dumps(meta, indent=2, sort_keys=True))
    if source_map_path.exists():
        print("source_map:")
        print(source_map_path.read_text(encoding="utf-8").strip())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pack scored rows for training.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build packed tensor dataset.")
    build.add_argument("--input", required=True)
    build.add_argument("--out-dir", required=True)
    build.add_argument("--skip", type=int, default=0)
    build.add_argument("--limit", type=int, default=0)
    build.add_argument("--rows", type=int, default=0)
    build.add_argument("--rows-file", default="")
    build.add_argument("--max-features", type=int, default=32)
    build.add_argument("--progress", type=int, default=250000)
    build.add_argument("--python", default=sys.executable)
    build.set_defaults(func=cmd_build)

    merge = subparsers.add_parser(
        "merge-shards",
        help="Merge compact packed label shards into a training dataset.",
    )
    merge.add_argument("--input-dir", required=True)
    merge.add_argument("--out-dir", required=True)
    merge.add_argument("--pattern", default="label.*.npz")
    merge.add_argument("--max-features", type=int, default=32)
    merge.add_argument("--progress", type=int, default=250000)
    merge.add_argument("--python", default=sys.executable)
    merge.set_defaults(func=cmd_merge_shards)

    inspect = subparsers.add_parser("inspect", help="Print packed dataset metadata.")
    inspect.add_argument("path")
    inspect.set_defaults(func=cmd_inspect)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
