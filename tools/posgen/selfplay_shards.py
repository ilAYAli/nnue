#!/usr/bin/env python3
"""Generate and merge distributed self-play PGN shards.

This tool keeps CPU-only self-play contributions reproducible.  A worker writes
one PGN plus metadata; the merge step accepts only shards with matching engine,
network, book, and self-play settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCHEMA = "enyo.selfplay_shard.v1"
MERGE_SCHEMA = "enyo.selfplay_shard_merge.v1"


def expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser().resolve()


def tool_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def count_pgn_games(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("[Event "):
                count += 1
    return count


def best_effort_engine_id(engine: Path, timeout_s: float = 5.0) -> str | None:
    try:
        proc = subprocess.Popen(
            [str(engine)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdin is not None
        assert proc.stdout is not None
        out, _ = proc.communicate("uci\nquit\n", timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return None
    except (OSError, subprocess.SubprocessError):
        return None

    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("id name "):
            return stripped[len("id name "):].strip()
        if stripped.startswith("id "):
            return stripped[len("id "):].strip()
    return None


def normalized_engine_options(options: list[str] | None) -> list[str]:
    return sorted(options or [])


def shard_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "concurrency": args.concurrency,
        "threads": args.threads,
        "depth": args.depth if not args.tc else None,
        "tc": args.tc,
        "book_format": args.book_format,
        "order": args.order,
        "restart": args.restart,
        "engine_options": normalized_engine_options(args.engine_option),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_selfplay(args: argparse.Namespace, output_pgn: Path) -> int:
    command = [
        sys.executable,
        str(tool_root() / "posgen" / "posgen.py"),
        "selfplay",
        "--runner", str(expand_path(args.runner)),
        "--engine", str(expand_path(args.engine)),
        "--book", str(expand_path(args.book)),
        "--output", str(output_pgn),
        "--games", str(args.games),
        "--shard-games", "0",
        "--concurrency", str(args.concurrency),
        "--threads", str(args.threads),
        "--srand", str(args.seed),
        "--restart", args.restart,
        "--book-format", args.book_format,
        "--order", args.order,
    ]
    if args.nnue_file:
        command.extend(["--nnue-file", str(expand_path(args.nnue_file))])
    if args.tc:
        command.extend(["--tc", args.tc])
    else:
        command.extend(["--depth", str(args.depth)])
    for option in args.engine_option or []:
        command.extend(["--engine-option", option])

    print(" ".join(command), flush=True)
    return subprocess.call(command)


def build_metadata(args: argparse.Namespace, output_pgn: Path) -> dict[str, Any]:
    engine = expand_path(args.engine)
    book = expand_path(args.book)
    nnue = expand_path(args.nnue_file) if args.nnue_file else None
    runner = expand_path(args.runner)
    return {
        "schema": SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": args.host or socket.gethostname(),
        "seed": args.seed,
        "pgn": str(output_pgn),
        "games_requested": args.games,
        "games_written": count_pgn_games(output_pgn),
        "engine_id": best_effort_engine_id(engine),
        "files": {
            "runner": file_record(runner),
            "engine": file_record(engine),
            "book": file_record(book),
            "nnue": file_record(nnue),
        },
        "settings": shard_settings(args),
    }


def shard_game_count(total_games: int, shards: int, shard_index: int) -> int:
    if total_games <= 0:
        raise ValueError("total_games must be > 0")
    if shards <= 0:
        raise ValueError("shards must be > 0")
    if shard_index < 0 or shard_index >= shards:
        raise ValueError(f"shard_index {shard_index} outside [0, {shards})")
    base = total_games // shards
    remainder = total_games % shards
    return base + (1 if shard_index < remainder else 0)


def shard_seed(base_seed: str, shard_index: int) -> str:
    try:
        return str(int(base_seed) + shard_index)
    except ValueError:
        return f"{base_seed}-{shard_index}"


def resolve_generate_args(args: argparse.Namespace) -> None:
    if args.total_games is not None:
        if args.shards is None or args.shard_index is None:
            raise SystemExit("--total-games requires --shards and --shard-index")
        args.games = shard_game_count(args.total_games, args.shards, args.shard_index)

    if args.base_seed is not None:
        if args.shard_index is None:
            raise SystemExit("--base-seed requires --shard-index")
        args.seed = shard_seed(args.base_seed, args.shard_index)

    if args.games is None:
        raise SystemExit("generate requires --games or --total-games")
    if args.seed is None:
        raise SystemExit("generate requires --seed or --base-seed")
    if args.games <= 0:
        raise SystemExit("resolved shard has no games")


def cmd_generate(args: argparse.Namespace) -> int:
    resolve_generate_args(args)
    output_pgn = expand_path(args.output_pgn)
    metadata = expand_path(args.metadata) if args.metadata else output_pgn.with_suffix(
        output_pgn.suffix + ".meta.json"
    )
    output_pgn.parent.mkdir(parents=True, exist_ok=True)
    rc = run_selfplay(args, output_pgn)
    if rc != 0:
        return rc

    payload = build_metadata(args, output_pgn)
    if payload["games_written"] != args.games:
        raise SystemExit(
            f"{output_pgn}: wrote {payload['games_written']} games, expected {args.games}"
        )
    write_json(metadata, payload)
    print(json.dumps({"pgn": str(output_pgn), "metadata": str(metadata)}, indent=2), flush=True)
    return 0


def load_metadata(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: metadata must be a JSON object")
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unsupported schema {payload.get('schema')!r}")
    return payload


def compatibility_key(payload: dict[str, Any]) -> dict[str, Any]:
    files = payload.get("files")
    settings = payload.get("settings")
    if not isinstance(files, dict) or not isinstance(settings, dict):
        raise ValueError("metadata is missing files/settings")
    return {
        "engine_sha256": (files.get("engine") or {}).get("sha256"),
        "book_sha256": (files.get("book") or {}).get("sha256"),
        "nnue_sha256": (files.get("nnue") or {}).get("sha256"),
        "settings": settings,
    }


def validate_merge_inputs(metadata_paths: list[Path]) -> list[dict[str, Any]]:
    if not metadata_paths:
        raise ValueError("merge requires at least one metadata file")

    payloads = [load_metadata(path) for path in metadata_paths]
    first_key = compatibility_key(payloads[0])
    seen_seeds: set[Any] = set()
    for path, payload in zip(metadata_paths, payloads, strict=True):
        key = compatibility_key(payload)
        if key != first_key:
            raise ValueError(f"{path}: incompatible shard metadata")
        seed = payload.get("seed")
        if seed in seen_seeds:
            raise ValueError(f"{path}: duplicate seed {seed!r}")
        seen_seeds.add(seed)

        pgn = expand_path(str(payload.get("pgn", "")))
        if not pgn.exists():
            raise ValueError(f"{path}: PGN does not exist: {pgn}")
        games_written = int(payload.get("games_written", -1))
        actual_games = count_pgn_games(pgn)
        if games_written != actual_games:
            raise ValueError(
                f"{path}: metadata says {games_written} games, PGN has {actual_games}"
            )

    return payloads


def append_file(src: Path, dst: Path) -> None:
    with src.open("rb") as in_handle, dst.open("ab") as out_handle:
        while True:
            chunk = in_handle.read(1024 * 1024)
            if not chunk:
                break
            out_handle.write(chunk)
        out_handle.write(b"\n")


def cmd_merge(args: argparse.Namespace) -> int:
    metadata_paths = [expand_path(path) for path in args.metadata]
    payloads = validate_merge_inputs(metadata_paths)

    output_pgn = expand_path(args.output_pgn)
    manifest = expand_path(args.manifest) if args.manifest else output_pgn.with_suffix(
        output_pgn.suffix + ".manifest.json"
    )
    output_pgn.parent.mkdir(parents=True, exist_ok=True)
    if output_pgn.exists() and not args.force:
        raise SystemExit(f"{output_pgn}: exists; pass --force to overwrite")
    output_pgn.unlink(missing_ok=True)

    total_games = 0
    shards: list[dict[str, Any]] = []
    for metadata_path, payload in zip(metadata_paths, payloads, strict=True):
        pgn = expand_path(str(payload["pgn"]))
        append_file(pgn, output_pgn)
        games = int(payload["games_written"])
        total_games += games
        shards.append({
            "metadata": str(metadata_path),
            "pgn": str(pgn),
            "seed": payload["seed"],
            "host": payload.get("host"),
            "games": games,
        })

    actual_games = count_pgn_games(output_pgn)
    if actual_games != total_games:
        raise SystemExit(f"{output_pgn}: wrote {actual_games} games, expected {total_games}")

    manifest_payload = {
        "schema": MERGE_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output_pgn": str(output_pgn),
        "games": total_games,
        "compatibility": compatibility_key(payloads[0]),
        "shards": shards,
    }
    write_json(manifest, manifest_payload)
    print(json.dumps({"output_pgn": str(output_pgn), "manifest": str(manifest), "games": total_games}, indent=2), flush=True)
    return 0


def add_generate_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("generate", help="Generate one self-play shard and metadata.")
    parser.add_argument("--runner", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--nnue-file")
    parser.add_argument("--book", required=True)
    parser.add_argument("--output-pgn", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--games", type=int)
    parser.add_argument("--total-games", type=int)
    parser.add_argument("--shards", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--tc")
    parser.add_argument("--book-format", default="epd")
    parser.add_argument("--order", default="random")
    parser.add_argument("--seed")
    parser.add_argument("--base-seed")
    parser.add_argument("--restart", choices=("auto", "on", "off"), default="off")
    parser.add_argument("--engine-option", action="append")
    parser.add_argument("--host")
    parser.set_defaults(func=cmd_generate)


def add_merge_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("merge", help="Merge compatible self-play shards.")
    parser.add_argument("--output-pgn", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("metadata", nargs="+")
    parser.set_defaults(func=cmd_merge)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_generate_parser(subparsers)
    add_merge_parser(subparsers)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
