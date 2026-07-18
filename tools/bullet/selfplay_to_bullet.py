#!/usr/bin/env python3
"""Convert Enyo self-play PGN shards into a training-ready Bullet file.

Label mode is an explicit, required choice, not an ad-hoc flag:

- self-distillation: keep Enyo's own per-move search score, blended with the
  real game WDL result. Matches Berserk's own self-play recipe. Scores remain
  in runtime units because the Enyo trainer applies phase normalization once
  when it loads every Bullet record.
- outcome-only: discard all engine scores; train only on game result
  (score=0, wdl=1.0 at the training-config level). For the zero-signal
  lineage only.

With --pgn-dir, each shard file already on disk is converted individually
and appended as raw bytes to --output (Bullet's binary format is a flat
sequence of fixed-size records with no header, so append is valid). Nothing
is ever merged into a duplicate copy of the PGN data first: shard content is
read directly from its existing file. A state file next to --output tracks
which shards were already converted, so re-running only processes new ones.
Per-shard intermediates (JSONL/bulletfmt/chunk) are deleted immediately
after each shard, so at most one shard's worth of temp data ever exists.
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
    label_mode: str,
    skip_plies: int,
    min_depth: int,
    max_abs_cp: int,
) -> tuple[Path, dict]:
    stem = pgn_path.stem
    rows_jsonl = tmp_dir / f"{stem}.rows.jsonl"
    stats_json = tmp_dir / f"{stem}.stats.json"
    bulletfmt = tmp_dir / f"{stem}.{label_mode}.bulletfmt"
    chunk_bullet = tmp_dir / f"{stem}.chunk.bullet"

    run([
        str(VENV_PYTHON), str(REPO_ROOT / "tools/posgen/pgn_to_jsonl.py"),
        str(pgn_path), "-o", str(rows_jsonl), "--stats", str(stats_json),
        "--skip-plies", str(skip_plies),
        "--min-depth", str(min_depth),
        "--max-abs-cp", str(max_abs_cp),
        # Without this, any position where the engine found a forced mate is
        # dropped entirely (pgn_to_jsonl's default). Those are exactly the
        # most decisive positions -- a crushing material advantage is often
        # a trivial mate for even a shallow search -- so omitting them left
        # training data with "somewhat ahead" examples but none of the
        # "completely over" ones, and the net never learned the extreme end
        # of the eval range. mate-score-cp matches Enyo's runtime ScaleEval
        # clamp (NNUE.md), the actual ceiling the network will ever need to
        # produce, so training targets stay inside the range the runtime
        # can express instead of chasing an arbitrary larger number.
        "--include-mates",
        "--mate-score-cp", "2045",
    ])

    bullet_stats_json = tmp_dir / f"{stem}.bullet_stats.json"
    convert_cmd = [
        str(VENV_PYTHON), str(REPO_ROOT / "tools/bullet/jsonl_to_bullet_text.py"),
        "--input", str(rows_jsonl), "--output", str(bulletfmt),
        # Must match pgn_to_jsonl.py's own --max-abs-cp, or this tool's
        # separate default (1600) silently re-drops rows the first stage
        # already accepted -- including every KQK-style position at the
        # runtime's actual +/-2045 ceiling, since 2045 divided by even a
        # mild phase scale still lands above 1600.
        "--max-abs-cp", str(max_abs_cp),
    ]
    if label_mode == "outcome-only":
        convert_cmd.append("--zero-score")
    with bullet_stats_json.open("w") as stats_out:
        result = subprocess.run(convert_cmd, check=True, stdout=stats_out)

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
    """Serialize concurrent invocations for the same --output.

    Forge's --task-complete-command is fire-and-forget by default, so many
    shards completing close together can trigger overlapping invocations of
    this script. Without locking, each one reads a stale state.json, so
    updates race and clobber each other -- the on-disk .bullet file can end
    up missing shards from state tracking (which then get reprocessed and
    duplicated on the next run) or, worse, appended twice. This lock makes
    concurrent invocations queue up and run one at a time instead.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pgn-dir", type=Path, required=True, help="Directory of *.pgn shards (e.g. a Forge selfplay run's outputs/ dir); incremental across re-runs")
    ap.add_argument("--output", type=Path, required=True, help="Final .bullet path, appended to incrementally")
    ap.add_argument("--label-mode", required=True, choices=["self-distillation", "outcome-only"])
    ap.add_argument("--skip-plies", type=int, default=8)
    ap.add_argument("--min-depth", type=int, default=1)
    ap.add_argument("--max-abs-cp", type=int, default=10000)
    ap.add_argument("--reset", action="store_true", help="Ignore/clear prior incremental state and reconvert everything")
    args = ap.parse_args()

    work_dir = args.output.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = work_dir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    state_path = work_dir / f"{args.output.stem}.state.json"
    lock_path = work_dir / f"{args.output.stem}.lock"

    with locked(lock_path):
        if args.reset:
            state_path.unlink(missing_ok=True)
            args.output.unlink(missing_ok=True)
        state = load_state(state_path)
        if state.get("label_mode") not in (None, args.label_mode):
            raise SystemExit(
                f"{state_path} was built with label_mode={state['label_mode']!r}; "
                f"pass --reset to switch to {args.label_mode!r}"
            )
        state["label_mode"] = args.label_mode

        all_shards = sorted(args.pgn_dir.glob("*.pgn"))
        processed = set(state["processed_shards"])
        new_shards = [s for s in all_shards if s.name not in processed]

        if not new_shards:
            print(f"no new shards in {args.pgn_dir} (already processed {len(processed)})")
            return 0

        skip_log_path = work_dir / f"{args.output.stem}.skips.jsonl"
        with args.output.open("ab") as out, skip_log_path.open("a") as skip_log:
            for index, shard in enumerate(new_shards, start=1):
                chunk, skipped = convert_shard(
                    shard, tmp_dir, args.label_mode,
                    args.skip_plies, args.min_depth, args.max_abs_cp,
                )
                chunk_bytes = chunk.stat().st_size
                with chunk.open("rb") as src:
                    out.write(src.read())
                chunk.unlink()
                out.flush()
                skip_log.write(json.dumps(skipped) + "\n")
                skip_log.flush()
                processed.add(shard.name)
                # Save after every shard, not just at the end: if this
                # process is killed mid-run, already-appended shards must
                # stay marked processed so a later run never re-appends them.
                state["processed_shards"] = sorted(processed)
                state["total_rows"] = state.get("total_rows", 0) + chunk_bytes // 32
                save_state(state_path, state)
                print(
                    f"[{index}/{len(new_shards)}] {shard.name} -> +{chunk_bytes // 32} rows "
                    f"(pgn_skipped_mate={skipped['pgn_skipped_mate']} "
                    f"bullet_skipped_cp={skipped['bullet_skipped_cp']})"
                )

        tmp_dir.rmdir()
        print(
            f"label_mode={args.label_mode} output={args.output} "
            f"new_shards={len(new_shards)} total_shards={len(state['processed_shards'])} "
            f"total_rows={state['total_rows']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
