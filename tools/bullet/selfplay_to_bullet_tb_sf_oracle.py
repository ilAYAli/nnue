#!/usr/bin/env python3
"""Convert Enyo self-play PGN shards into a training-ready Bullet file,
labeled with exact Syzygy WDL for positions with <= --tb-pieces pieces and a
fresh Stockfish static eval for everything else, instead of Enyo's own
per-move search score.

Mirrors tools/bullet/selfplay_to_bullet_sf_oracle.py exactly, swapping only
the relabel step for relabel_with_syzygy_and_stockfish.py.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python3"


def run(cmd: list[str]) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def convert_shard(
    pgn_path: Path,
    tmp_dir: Path,
    skip_plies: int,
    min_depth: int,
    max_abs_cp: int,
    sf_engine: str,
    sf_net: str,
    sf_hash: int,
    tb_pieces: int,
) -> tuple[Path, dict]:
    stem = pgn_path.stem
    rows_jsonl = tmp_dir / f"{stem}.rows.jsonl"
    stats_json = tmp_dir / f"{stem}.stats.json"
    sf_jsonl = tmp_dir / f"{stem}.sf.jsonl"
    bulletfmt = tmp_dir / f"{stem}.tb-sf-oracle.bulletfmt"
    chunk_bullet = tmp_dir / f"{stem}.chunk.bullet"

    run([
        str(VENV_PYTHON), str(REPO_ROOT / "tools/posgen/pgn_to_jsonl.py"),
        str(pgn_path), "-o", str(rows_jsonl), "--stats", str(stats_json),
        "--skip-plies", str(skip_plies),
        "--min-depth", str(min_depth),
        "--max-abs-cp", str(max_abs_cp),
        "--include-mates",
        "--mate-score-cp", "2045",
    ])

    run([
        str(VENV_PYTHON), str(REPO_ROOT / "tools/posgen/relabel_with_syzygy_and_stockfish.py"),
        "--input", str(rows_jsonl), "--output", str(sf_jsonl),
        "--engine", sf_engine, "--net", sf_net, "--hash", str(sf_hash),
        "--max-abs-cp", str(max_abs_cp),
        "--tb-pieces", str(tb_pieces),
    ])

    bullet_stats_json = tmp_dir / f"{stem}.bullet_stats.json"
    convert_cmd = [
        str(VENV_PYTHON), str(REPO_ROOT / "tools/bullet/jsonl_to_bullet_text.py"),
        "--input", str(sf_jsonl), "--output", str(bulletfmt),
        "--max-abs-cp", str(max_abs_cp),
    ]
    with bullet_stats_json.open("w") as stats_out:
        subprocess.run(convert_cmd, check=True, stdout=stats_out)

    run([
        str(VENV_PYTHON), str(REPO_ROOT / "tools/bullet/bullet.py"), "format",
        "--input", str(bulletfmt), "--output", str(chunk_bullet), "--validate",
    ])

    pgn_stats = json.loads(stats_json.read_text()) if stats_json.exists() else {}
    bullet_stats = json.loads(bullet_stats_json.read_text()) if bullet_stats_json.exists() else {}
    skipped = {
        "shard": pgn_path.name,
        "pgn_skipped_mate": pgn_stats.get("skipped_mate"),
        "pgn_skipped_cp": pgn_stats.get("skipped_cp"),
        "pgn_skipped_depth": pgn_stats.get("skipped_depth"),
        "bullet_skipped_cp": bullet_stats.get("skipped_cp"),
        "bullet_written": bullet_stats.get("written"),
    }

    rows_jsonl.unlink(missing_ok=True)
    sf_jsonl.unlink(missing_ok=True)
    bulletfmt.unlink(missing_ok=True)
    stats_json.unlink(missing_ok=True)
    bullet_stats_json.unlink(missing_ok=True)
    return chunk_bullet, skipped


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return {"processed_shards": [], "total_rows": 0}


def save_state(state_path: Path, state: dict) -> None:
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@contextmanager
def locked(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pgn-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--skip-plies", type=int, default=8)
    ap.add_argument("--min-depth", type=int, default=1)
    ap.add_argument("--max-abs-cp", type=int, default=10000)
    ap.add_argument("--sf-engine", default=str(Path.home() / "assets/engines/candidate"))
    ap.add_argument("--sf-net", default=str(Path.home() / "assets/nets/nn-0ee0657fb25e.nnue"))
    ap.add_argument("--sf-hash", type=int, default=64)
    ap.add_argument("--tb-pieces", type=int, default=6)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--limit-shards", type=int, default=0, help="Process at most N new shards this run (0 = all)")
    ap.add_argument(
        "--shard-slice", default=None,
        help="I/N: deterministically assign every Nth shard (by sorted "
             "index, offset I) to this worker, for running N parallel "
             "workers against N separate --output files with no lock "
             "contention. Partitions the full shard list, not just the "
             "unprocessed ones, so it stays stable across runs.",
    )
    args = ap.parse_args()

    work_dir = args.output.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    tmp_tag = args.output.stem
    tmp_dir = work_dir / f".tmp-tb-sf-oracle-{tmp_tag}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    state_path = work_dir / f"{args.output.stem}.state.json"
    lock_path = work_dir / f"{args.output.stem}.lock"

    with locked(lock_path):
        if args.reset:
            state_path.unlink(missing_ok=True)
            args.output.unlink(missing_ok=True)
        state = load_state(state_path)

        all_shards = sorted(args.pgn_dir.glob("*.pgn"))
        if args.shard_slice:
            offset_str, count_str = args.shard_slice.split("/")
            offset, count = int(offset_str), int(count_str)
            all_shards = [s for i, s in enumerate(all_shards) if i % count == offset]
        processed = set(state["processed_shards"])
        new_shards = [s for s in all_shards if s.name not in processed]
        if args.limit_shards:
            new_shards = new_shards[: args.limit_shards]

        if not new_shards:
            print(f"no new shards in {args.pgn_dir} (already processed {len(processed)})")
            return 0

        skip_log_path = work_dir / f"{args.output.stem}.skips.jsonl"
        with args.output.open("ab") as out, skip_log_path.open("a") as skip_log:
            for index, shard in enumerate(new_shards, start=1):
                chunk, skipped = convert_shard(
                    shard, tmp_dir,
                    args.skip_plies, args.min_depth, args.max_abs_cp,
                    args.sf_engine, args.sf_net, args.sf_hash, args.tb_pieces,
                )
                chunk_bytes = chunk.stat().st_size
                with chunk.open("rb") as src:
                    out.write(src.read())
                chunk.unlink()
                out.flush()
                skip_log.write(json.dumps(skipped) + "\n")
                skip_log.flush()
                processed.add(shard.name)
                state["processed_shards"] = sorted(processed)
                state["total_rows"] = state.get("total_rows", 0) + chunk_bytes // 32
                save_state(state_path, state)
                print(
                    f"[{index}/{len(new_shards)}] {shard.name} -> +{chunk_bytes // 32} rows "
                    f"(pgn_skipped_mate={skipped['pgn_skipped_mate']} "
                    f"bullet_skipped_cp={skipped['bullet_skipped_cp']})"
                )

        print(
            f"label_mode=syzygy-tb-stockfish-oracle output={args.output} "
            f"new_shards={len(new_shards)} total_shards={len(state['processed_shards'])} "
            f"total_rows={state['total_rows']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
