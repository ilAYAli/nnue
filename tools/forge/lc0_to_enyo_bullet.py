#!/usr/bin/env python3
"""Run the complete LC0 -> Stockfish -> Enyo-runtime -> Bullet Forge pipeline.

See LC0_CONVERSION.md for the full contract this wrapper implements: a
persistent, resumable work directory under
``<output-dir>/.<output-stem>.work/`` records four gates per batch (Plan,
Launch, Shard, Batch) so an interrupted run can resume without recomputing
already-valid work, and the final merge is only published after independent
validation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.lib.hashing import sha256_file  # noqa: E402
from tools.validate.validate_bullet_results import validate_and_merge  # noqa: E402


DEFAULT_INPUT = Path.home() / "assets/training/lc0/test91-forge-input"
DEFAULT_OUTPUT = Path.home() / "assets/training/bullet/lc0-stockfish/test91-stockfish-enyo.bullet"
DEFAULT_ENGINE = Path.home() / "assets/engines/reference"
DEFAULT_NET = Path.home() / "assets/nets/nn-1a298aa575a0.nnue"
LEGACY_OUTPUT_DIR = Path.home() / "assets/training/bullet/lc0/test91"
MIN_ARCHIVES = 100
MIN_BYTES = 100_000_000_000
DEFAULT_BATCH_BYTES = 20_000_000_000
DEFAULT_TARGET_TASK_BYTES = 100_000_000
DEFAULT_ENGINE_TIMEOUT_S = 120
DEFAULT_MIN_SCORE_SPREAD = 1
DEFAULT_LEASE_SECONDS = 600
DEFAULT_POLL_INTERVAL_S = 20.0
DEFAULT_WORKER_STALE_SECONDS = 30.0
# forge worker del only releases the coordinator-side claim; it does not
# kill the worker's actual OS process. Quarantining a worker that is merely
# slow (not dead) therefore frees its task for another worker to redo while
# the original process keeps running unmonitored, silently duplicating work
# and orphaning a process every cycle. Forge's own lease-based reap
# (lease_seconds, default 600s) is the safe fallback for genuinely abandoned
# work; this grace period must stay close to that, not far below it, so
# quarantine only fires meaningfully ahead of the safe reap rather than
# preempting it by many minutes on nothing more than a busy worker.
DEFAULT_QUARANTINE_GRACE_S = 480.0
FORGE_UNPACKED = Path.home() / ".cache/forge/unpacked-lc0"
FORGE_INPUTS = Path.home() / ".cache/forge/inputs"
FORGE_TASK_INPUTS = Path.home() / ".cache/forge/task-inputs"
WORK_STATE_SCHEMA = "enyo.lc0-stockfish-enyo-bullet.work-state.v1"


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json_if_valid(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def archive_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.tar")
        if path.is_file() and path.name.startswith(("training.", "training-"))
    )


def preflight_source(root: Path, *, allow_small: bool) -> tuple[int, int]:
    if not root.is_dir():
        raise SystemExit(f"LC0 input is not a directory: {root}")
    archives = archive_paths(root)
    total_bytes = sum(path.stat().st_size for path in archives)
    if not archives:
        raise SystemExit(f"LC0 input contains no training.*.tar archives: {root}")
    if not allow_small and (len(archives) < MIN_ARCHIVES or total_bytes < MIN_BYTES):
        raise SystemExit(
            "refusing undersized LC0 input (this catches the four-archive fixture): "
            f"archives={len(archives):,}, bytes={total_bytes:,}; "
            "pass --allow-small-input only for an intentional small test"
        )
    return len(archives), total_bytes


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} does not exist or is not a file: {path}")


def cleanup_old_outputs(root: Path, *, keep: Path | None = None) -> list[Path]:
    """Remove only named conversion products, never the LC0 source or static corpus."""
    removed: list[Path] = []
    if not root.is_dir():
        return removed
    patterns = (
        "lc0-root-*.bullet",
        "lc0-root-*.bullet.*",
        "lc0-root-*.calibration.json",
        "lc0-root-*.validation.json",
    )
    for pattern in patterns:
        for path in root.glob(pattern):
            if keep is not None and path == keep:
                continue
            if path.is_file() or path.is_symlink():
                path.unlink()
                removed.append(path)
    for path in root.glob("lc0-root-*-chunk*"):
        if keep is not None and path == keep:
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
            removed.append(path)
    return removed


# ---------------------------------------------------------------------------
# Batch partitioning: byte-weighted batches, each sized into ~100MB tasks
# ---------------------------------------------------------------------------


def partition_archives(archives: list[Path], max_bytes: int) -> list[list[Path]]:
    if max_bytes <= 0:
        raise ValueError("batch byte limit must be positive")
    batches: list[list[Path]] = []
    current: list[Path] = []
    current_bytes = 0
    for archive in archives:
        size = archive.stat().st_size
        if current and current_bytes + size > max_bytes:
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(archive)
        current_bytes += size
    if current:
        batches.append(current)
    return batches


def batch_shard_count(batch_bytes: int, target_task_bytes: int, *, override: int | None) -> int:
    if override is not None:
        return max(1, override)
    return max(1, math.ceil(batch_bytes / target_task_bytes))


def link_batch(archives: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in destination.iterdir()}
    for index, archive in enumerate(archives):
        # Keep the training.* prefix required by Forge while making names
        # unique even if a future source directory contains duplicate basenames.
        name = f"training.{index:05d}-{archive.name.removeprefix('training.')}"
        if name in existing:
            continue
        (destination / name).symlink_to(archive)


def cleanup_new_cache_entries(before: set[Path], root: Path) -> None:
    if not root.is_dir():
        return
    for entry in root.iterdir():
        if entry not in before and entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry, ignore_errors=True)


# ---------------------------------------------------------------------------
# Source identity: cheap drift detection, not a content re-hash of the corpus
# ---------------------------------------------------------------------------


def archive_identity(path: Path) -> list[Any]:
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


def archives_digest(archives: list[Path]) -> str:
    payload = json.dumps([archive_identity(path) for path in archives], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Forge partition preflight (unchanged verification, still gates every launch)
# ---------------------------------------------------------------------------


def verify_forge_partition(command: list[str]) -> tuple[dict[str, object], dict[str, object]]:
    """Build a manifest without launching workers and reject overlapping tasks."""
    result = subprocess.run(
        [*command, "--print-manifest"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise SystemExit(
            "Forge partition preflight failed: "
            + (detail[-1] if detail else f"rc={result.returncode}")
        )
    try:
        manifest = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit("Forge partition preflight returned invalid JSON") from exc
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise SystemExit("Forge partition preflight produced no tasks")
    match = re.search(r"found\s+([0-9,]+)\s+files", result.stderr)
    if match is None:
        raise SystemExit("Forge partition preflight did not report its inventory size")
    expected_file_count = int(match.group(1).replace(",", ""))

    seen: set[tuple[str, str, int]] = set()
    seen_paths: set[str] = set()
    for task in tasks:
        inputs = [item for item in task.get("inputs", []) if item.get("tree") == "lc0-inventory"]
        if len(inputs) != 1:
            raise SystemExit(f"{task.get('id', '?')}: expected exactly one LC0 task inventory")
        item = inputs[0]
        task_inventory_path = Path(str(item.get("path", ""))).expanduser() / "inventory.json"
        try:
            payload = json.loads(task_inventory_path.read_text(encoding="utf-8"))
            entries = payload["files"]
            task_keys = {
                (str(entry["path"]), str(entry["sha256"]), int(entry["size"]))
                for entry in entries
            }
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read task inventory: {task_inventory_path}") from exc
        if len(task_keys) != len(entries):
            raise SystemExit(f"{task.get('id', '?')}: duplicate entries inside task inventory")
        paths = {key[0] for key in task_keys}
        if seen_paths & paths:
            overlap = sorted(seen_paths & paths)[0]
            raise SystemExit(f"Forge task inventories overlap at {overlap}")
        seen.update(task_keys)
        seen_paths.update(paths)
        if int(item.get("files", -1)) != len(task_keys):
            raise SystemExit(f"{task.get('id', '?')}: manifest file count disagrees with inventory")
    if len(seen) != expected_file_count:
        raise SystemExit(
            "Forge task inventories do not exactly cover the coordinator inventory: "
            f"tasks={len(seen):,} expected={expected_file_count:,}"
        )
    return (
        {
            "tasks": len(tasks),
            "files": len(seen),
            "inventory_files": expected_file_count,
        },
        manifest,
    )


def build_command(args: argparse.Namespace, template: Path, *, batch_input: Path,
                   batch_output_dir: Path, shard_count: int, net_sha256: str) -> list[str]:
    # Deliberately no --engine-sha256: each worker compiles its own engine
    # binary for its own CPU, so pinning the requirement to the coordinator's
    # hash would make Forge's mismatched-input auto-sync overwrite every
    # worker's binary with the coordinator's (see the template's engine
    # parameter help text for the full incident this caused).
    return [
        "forge", "run", str(template),
        "--input", str(batch_input),
        "--output", str(batch_output_dir),
        "--engine", str(args.engine),
        "--net", str(args.net),
        "--net-sha256", net_sha256,
        "--depth", str(args.depth),
        "--threads", str(args.threads),
        "--hash", str(args.hash),
        "--engine-timeout-s", str(args.engine_timeout_s),
        "--max-records", "0",
        "--split-records", str(shard_count),
        "--min-ply", str(args.min_ply),
        "--shards", str(shard_count),
    ] + (["--quiet-only"] if args.quiet_only else ["--no-quiet-only"])


def forge_workers_arg() -> str:
    """Mirror Forge's default worker-file selection for ``forge start``."""
    if value := os.environ.get("FORGE_WORKERS"):
        return value
    config_path = os.environ.get("FORGE_CONFIG")
    if config_path:
        return str(Path(config_path).expanduser().parent / "workers.json")
    return "~/.config/forge/workers.json"


def build_start_command(manifest: Path) -> list[str]:
    """Launch an already-expanded manifest without replanning its tasks."""
    return ["forge", "start", str(manifest), "--workers", forge_workers_arg()]


def build_resume_command(run_name: str) -> list[str]:
    """Relaunch workers and requeue failed tasks against an existing run."""
    return ["forge", "resume", run_name, "--workers", forge_workers_arg()]


def materialized_manifest_path(run_name: str) -> Path:
    """Resolve the coordinator's materialized manifest for a launched run."""
    result = subprocess.run(
        ["forge", "status", run_name, "--json"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        detail = (result.stderr or result.stdout).strip()
        raise SystemExit(f"cannot inspect materialized Forge run {run_name}: {detail}")
    try:
        payload = json.loads(result.stdout)
        path = Path(str(payload["manifest"])).expanduser()
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Forge status did not return a manifest for {run_name}") from exc
    if not path.is_file():
        raise SystemExit(f"materialized Forge manifest is missing: {path}")
    return path


def forge_status_json(run_name: str, *, lease_seconds: int) -> dict[str, Any]:
    result = subprocess.run(
        ["forge", "status", run_name, "--json", "--lease-seconds", str(lease_seconds)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        detail = (result.stderr or result.stdout).strip()
        raise SystemExit(f"cannot inspect Forge run {run_name}: {detail}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Forge status returned invalid JSON for {run_name}") from exc


def validate_materialized_partition(
    expected: dict[str, object], actual: dict[str, object]
) -> None:
    """Reject a coordinator manifest whose task inputs differ from preflight."""
    expected_tasks = expected.get("tasks")
    actual_tasks = actual.get("tasks")
    if not isinstance(expected_tasks, list) or not isinstance(actual_tasks, list):
        raise SystemExit("Forge materialized manifest has no task list")
    if len(expected_tasks) != len(actual_tasks):
        raise SystemExit(
            "Forge materialized task count changed after preflight: "
            f"expected={len(expected_tasks)} actual={len(actual_tasks)}"
        )
    actual_by_id = {
        str(task.get("id")): task
        for task in actual_tasks
        if isinstance(task, dict)
    }
    for expected_task in expected_tasks:
        if not isinstance(expected_task, dict):
            raise SystemExit("Forge preflight returned an invalid task")
        task_id = str(expected_task.get("id"))
        actual_task = actual_by_id.get(task_id)
        if actual_task is None:
            raise SystemExit(f"Forge materialized manifest is missing task {task_id}")
        expected_inputs = [
            item for item in expected_task.get("inputs", [])
            if isinstance(item, dict) and item.get("tree") == "lc0-inventory"
        ]
        actual_inputs = [
            item for item in actual_task.get("inputs", [])
            if isinstance(item, dict) and item.get("tree") == "lc0-inventory"
        ]
        if len(expected_inputs) != 1 or len(actual_inputs) != 1:
            raise SystemExit(f"{task_id}: materialized LC0 input is missing")
        expected_input = expected_inputs[0]
        actual_input = actual_inputs[0]
        for key in ("digest", "files", "bytes", "path", "inventory_source"):
            if actual_input.get(key) != expected_input.get(key):
                raise SystemExit(
                    f"{task_id}: materialized LC0 partition changed ({key})"
                )


# ---------------------------------------------------------------------------
# Worker quarantine: detect a stale worker and force-release its claims
# ---------------------------------------------------------------------------


def quarantine_stale_workers(
    run_name: str,
    *,
    lease_seconds: int,
    worker_stale_seconds: float,
    grace_seconds: float,
    stale_since: dict[str, float],
    quarantined: dict[str, dict[str, Any]],
    quarantine_dir: Path,
) -> dict[str, Any]:
    """Poll once; quarantine any worker stale past the grace period.

    Returns the raw `forge status --json` payload so the caller can decide
    whether to keep polling.
    """
    status = forge_status_json(run_name, lease_seconds=lease_seconds)
    now = time.monotonic()
    for row in status.get("worker_processes", []) or []:
        worker_id = str(row.get("worker_id") or "").strip()
        if not worker_id or worker_id in quarantined:
            continue
        heartbeat_age = row.get("heartbeat_age_s")
        is_stale = row.get("state") == "stale" or (
            isinstance(heartbeat_age, (int, float)) and heartbeat_age > worker_stale_seconds
        )
        if not is_stale:
            stale_since.pop(worker_id, None)
            continue
        first_seen = stale_since.setdefault(worker_id, now)
        if now - first_seen < grace_seconds:
            continue
        result = subprocess.run(
            ["forge", "worker", "del", worker_id, run_name, "--lease-seconds", str(lease_seconds)],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        record = {
            "worker_id": worker_id,
            "run_name": run_name,
            "reason": "heartbeat stale",
            "heartbeat_age_s": heartbeat_age,
            "detected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "forge_worker_del_rc": result.returncode,
            "forge_worker_del_output": (result.stdout + result.stderr).strip(),
        }
        atomic_write_json(quarantine_dir / f"{worker_id}.json", record)
        quarantined[worker_id] = record
        print(json.dumps({"quarantined": record}), flush=True)
    return status


def wait_with_quarantine(
    run_name: str,
    manifest_path: Path,
    *,
    lease_seconds: int,
    poll_interval_s: float,
    worker_stale_seconds: float,
    grace_seconds: float,
    quarantine_dir: Path,
) -> None:
    stale_since: dict[str, float] = {}
    quarantined: dict[str, dict[str, Any]] = {}
    while True:
        status = quarantine_stale_workers(
            run_name,
            lease_seconds=lease_seconds,
            worker_stale_seconds=worker_stale_seconds,
            grace_seconds=grace_seconds,
            stale_since=stale_since,
            quarantined=quarantined,
            quarantine_dir=quarantine_dir,
        )
        counts = status.get("counts") or {}
        outstanding = sum(int(counts.get(key) or 0) for key in ("pending", "running", "claimed", "stale"))
        if outstanding == 0:
            break
        time.sleep(poll_interval_s)
    # All tasks are terminal; this is Forge's authoritative pass/fail check
    # (execution checksum + declared outputs), and should return immediately.
    subprocess.run(
        ["forge", "wait", "--manifest", str(manifest_path), "--lease-seconds", str(lease_seconds)],
        cwd=REPO_ROOT,
        check=True,
    )


# ---------------------------------------------------------------------------
# Work directory: persistent, resumable state across the four gates
# ---------------------------------------------------------------------------


@dataclass
class Batch:
    index: int
    archives: list[Path]
    bytes: int
    shard_count: int

    @property
    def name(self) -> str:
        return f"{self.index:04d}"


def work_dir_for_output(output: Path) -> Path:
    return output.parent / f".{output.stem}.work"


def batch_dir(work_dir: Path, batch: Batch) -> Path:
    return work_dir / "batches" / batch.name


def build_batches(archives: list[Path], *, batch_bytes: int, target_task_bytes: int,
                   shard_override: int | None) -> list[Batch]:
    raw_batches = partition_archives(archives, batch_bytes)
    batches = []
    for index, batch_archives in enumerate(raw_batches):
        total = sum(path.stat().st_size for path in batch_archives)
        shard_count = batch_shard_count(total, target_task_bytes, override=shard_override)
        batches.append(Batch(index=index, archives=batch_archives, bytes=total, shard_count=shard_count))
    return batches


def init_or_verify_work_state(work_dir: Path, *, args: argparse.Namespace,
                               archives: list[Path], batches: list[Batch]) -> dict[str, Any]:
    state_path = work_dir / "work-state.json"
    corpus_digest = archives_digest(archives)
    identity = {
        "input": str(args.input),
        "output": str(args.output),
        "engine": str(args.engine),
        "net": str(args.net),
        "depth": args.depth,
        "threads": args.threads,
        "hash": args.hash,
        "min_ply": args.min_ply,
        "quiet_only": args.quiet_only,
        "batch_bytes": args.batch_bytes,
        "target_task_bytes": args.target_task_bytes,
    }
    fresh = {
        "schema": WORK_STATE_SCHEMA,
        "identity": identity,
        "archive_count": len(archives),
        "archive_bytes": sum(path.stat().st_size for path in archives),
        "corpus_digest": corpus_digest,
        "batch_count": len(batches),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    existing = read_json_if_valid(state_path)
    if existing is None:
        atomic_write_json(state_path, fresh)
        return fresh
    drift = []
    if existing.get("identity") != identity:
        drift.append("identity")
    if existing.get("corpus_digest") != corpus_digest:
        drift.append("corpus_digest")
    if existing.get("batch_count") != len(batches):
        drift.append("batch_count")
    if drift:
        raise SystemExit(
            "refusing to resume: source or configuration drifted since the last run "
            f"({', '.join(drift)}); the raw LC0 input and prior work state are untouched. "
            f"Work directory: {work_dir}"
        )
    return existing


# ---------------------------------------------------------------------------
# Gate 1 (Plan) / Gate 2 (Launch)
# ---------------------------------------------------------------------------


def batch_plan_digest(batch: Batch) -> str:
    payload = json.dumps(
        {"archives": [archive_identity(path) for path in batch.archives], "shard_count": batch.shard_count},
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_batch_plan(work_dir: Path, batch: Batch) -> dict[str, Any]:
    plan_path = batch_dir(work_dir, batch) / "plan.json"
    plan = {
        "batch": batch.index,
        "archives": [str(path) for path in batch.archives],
        "archive_bytes": batch.bytes,
        "shard_count": batch.shard_count,
        "digest": batch_plan_digest(batch),
    }
    atomic_write_json(plan_path, plan)
    return plan


def ensure_batch_launched(
    args: argparse.Namespace, template: Path, batch: Batch, this_batch_dir: Path,
    *, net_sha256: str,
) -> tuple[str, Path]:
    """Idempotently reach Gate 2 (Launch) for one batch; return (run_name, manifest_path)."""
    launch_path = this_batch_dir / "launch.json"
    materialized_copy = this_batch_dir / "materialized.manifest.json"
    existing_launch = read_json_if_valid(launch_path)
    if existing_launch is not None:
        run_name = str(existing_launch["run_name"])
        manifest_path = materialized_manifest_path(run_name)
        print(json.dumps({"batch": batch.index, "resuming_run": run_name}), flush=True)
        subprocess.run(build_resume_command(run_name), cwd=REPO_ROOT, check=True)
        return run_name, manifest_path

    batch_input = this_batch_dir / "input"
    batch_output_dir = this_batch_dir / "output"
    link_batch(batch.archives, batch_input)
    # Forge's own preflight refuses to plan into an output directory that
    # already exists; it creates this itself during materialization.
    command = build_command(
        args, template, batch_input=batch_input, batch_output_dir=batch_output_dir,
        shard_count=batch.shard_count, net_sha256=net_sha256,
    )
    unpacked_before = set(FORGE_UNPACKED.iterdir()) if FORGE_UNPACKED.is_dir() else set()
    inputs_before = set(FORGE_INPUTS.iterdir()) if FORGE_INPUTS.is_dir() else set()
    task_inputs_before = set(FORGE_TASK_INPUTS.iterdir()) if FORGE_TASK_INPUTS.is_dir() else set()
    try:
        partition, manifest = verify_forge_partition(command)
        print(json.dumps({"batch": batch.index, "partition": partition}), flush=True)
        run_name = str(manifest.get("name") or "")
        if not run_name:
            raise SystemExit("Forge preflight manifest has no run name")
        # The preflight manifest is the plan we validated.  Starting the
        # original `forge run` command here would expand the template a
        # second time and could produce a different task partition
        # (especially when workers/configuration differ).  `forge start`
        # materializes this exact manifest instead.
        with tempfile.TemporaryDirectory(prefix="forge-validated-") as tmp:
            manifest_path = Path(tmp) / "validated.manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
            )
            subprocess.run(build_start_command(manifest_path), cwd=REPO_ROOT, check=True)
            actual_manifest_path = materialized_manifest_path(run_name)
            actual_manifest = json.loads(actual_manifest_path.read_text(encoding="utf-8"))
            try:
                validate_materialized_partition(manifest, actual_manifest)
            except SystemExit:
                subprocess.run(
                    ["forge", "stop", run_name, "--force", "--no-wait"],
                    cwd=REPO_ROOT, check=False,
                )
                raise
            shutil.copyfile(actual_manifest_path, materialized_copy)
            atomic_write_json(launch_path, {
                "batch": batch.index,
                "run_name": run_name,
                "materialized_manifest_sha256": sha256_file(actual_manifest_path),
                "launched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            })
            return run_name, actual_manifest_path
    finally:
        cleanup_new_cache_entries(unpacked_before, FORGE_UNPACKED)
        cleanup_new_cache_entries(inputs_before, FORGE_INPUTS)
        cleanup_new_cache_entries(task_inputs_before, FORGE_TASK_INPUTS)


# ---------------------------------------------------------------------------
# Gate 3 (Shard) / Gate 4 (Batch)
# ---------------------------------------------------------------------------


def validate_shards(
    this_batch_dir: Path, manifest: dict[str, Any], *,
    expected_net_sha256: str,
) -> list[Path]:
    """Run the Shard gate for every task; return the batch's bullet shard paths."""
    state_dir = Path(str(manifest.get("state_dir") or "state")).expanduser()
    log_dir = Path(str(manifest.get("log_dir") or "logs")).expanduser()
    shards_dir = this_batch_dir / "shards"
    logs_dir = this_batch_dir / "logs"
    bullet_paths: list[Path] = []
    for task in manifest.get("tasks", []):
        task_id = str(task["id"])
        outputs = task.get("outputs") or []
        if len(outputs) != 2:
            raise SystemExit(f"{task_id}: expected [bullet, stats] outputs, got {outputs!r}")
        bullet_path = Path(outputs[0]).expanduser()
        stats_path = Path(outputs[1]).expanduser()
        done_path = state_dir / f"{task_id}.done.json"
        fail_path = state_dir / f"{task_id}.fail.json"
        done = read_json_if_valid(done_path)
        if done is None:
            fail = read_json_if_valid(fail_path)
            raise SystemExit(f"{task_id}: no done state found (fail={fail})")
        stats = read_json_if_valid(stats_path)
        if stats is None:
            raise SystemExit(f"{task_id}: stats.json missing or invalid: {stats_path}")

        reasons = []
        if stats.get("score_source") != "uci":
            reasons.append(f"score_source={stats.get('score_source')!r}")
        if stats.get("error") is not None:
            reasons.append(f"error={stats.get('error')!r}")
        if stats.get("net_load_confirmed") is not True:
            reasons.append(f"net_load_confirmed={stats.get('net_load_confirmed')!r}")
        # engine_sha256 is recorded but not gated on a single expected value:
        # each worker legitimately compiles its own engine binary for its own
        # CPU, so there is no one correct hash to compare against. Only
        # require that label.py successfully hashed *some* engine binary.
        if not stats.get("engine_sha256"):
            reasons.append(f"engine_sha256={stats.get('engine_sha256')!r}")
        if stats.get("net_sha256") != expected_net_sha256:
            reasons.append(f"net_sha256={stats.get('net_sha256')!r}")
        if not bullet_path.is_file():
            reasons.append(f"missing bullet output: {bullet_path}")

        task_log_src = log_dir / f"{task_id}.log"
        logs_dir.mkdir(parents=True, exist_ok=True)
        task_log_dst = logs_dir / f"{task_id}.log"
        if task_log_src.is_file():
            shutil.copyfile(task_log_src, task_log_dst)

        shard_record = {
            "task": task_id,
            "bullet": str(bullet_path),
            "stats": stats,
            "rc": done.get("rc"),
            "elapsed_s": done.get("elapsed_s"),
            "task_execution_sha256": done.get("task_execution_sha256"),
            "log": str(task_log_dst) if task_log_src.is_file() else None,
            "status": "invalid" if reasons else "valid",
            "reasons": reasons,
        }
        atomic_write_json(shards_dir / f"{task_id}.json", shard_record)
        if reasons:
            raise SystemExit(f"{task_id}: shard gate failed: {'; '.join(reasons)}")
        bullet_paths.append(bullet_path)
    return bullet_paths


def validate_batch(this_batch_dir: Path, shard_paths: list[Path], *,
                    min_score_spread: int | None) -> dict[str, Any]:
    output_path = this_batch_dir / "output.bullet"
    validation = validate_and_merge(
        shard_paths,
        merge_output=output_path,
        replace=True,
        require_win_loss=True,
        min_score_spread=min_score_spread,
    )
    atomic_write_json(this_batch_dir / "batch-validation.json", {**validation, "valid": True})
    return validation


UNPACKED_LC0_CACHE_ROOT = (Path.home() / ".cache/forge/unpacked-lc0").resolve()


def cleanup_unpacked_lc0_sources(manifest: dict[str, Any]) -> list[str]:
    """Remove this batch's extracted-tar cache once its shards are validated.

    ``unpack_lc0_archives`` (forge_lib/labeling.py) extracts each batch's raw
    .tar archives into ~/.cache/forge/unpacked-lc0/<digest>/ roughly 1:1 in
    size with the source archive, and nothing else ever cleans it up. Left
    alone across a full-corpus run (many batches), that cache accumulates to
    approximately the full corpus size on top of the batch outputs and final
    merge, which does not fit in the available disk headroom. Once a batch's
    shards are copied out and its output is merged, the extracted source is
    no longer needed, so it is safe to remove here.
    """
    sources: set[Path] = set()
    for task in manifest.get("tasks", []):
        for item in task.get("inputs", []):
            if item.get("tree") != "lc0-inventory":
                continue
            source = item.get("source")
            if source:
                sources.add(Path(source).expanduser().resolve())
    removed = []
    for source in sources:
        if UNPACKED_LC0_CACHE_ROOT not in source.parents:
            continue
        if source.is_dir():
            shutil.rmtree(source, ignore_errors=True)
            removed.append(str(source))
    return removed


# ---------------------------------------------------------------------------
# Final merge and provenance
# ---------------------------------------------------------------------------


def write_provenance(path: Path, *, args: argparse.Namespace, archive_count: int,
                      archive_bytes: int, validation: dict[str, object],
                      removed: Iterable[Path], engine_sha256: str, net_sha256: str) -> Path:
    manifest = path.with_name(path.name + ".manifest.json")
    payload = {
        "schema": "enyo.lc0-stockfish-enyo-bullet.v2",
        "pipeline": [
            "LC0 V6 decode",
            "Stockfish UCI search labels via EvalFile",
            "Enyo runtime clamp and phase normalization",
            "BulletFormat serialization",
        ],
        "input": str(args.input),
        "input_archive_count": archive_count,
        "input_archive_bytes": archive_bytes,
        "engine": str(args.engine),
        "engine_sha256": engine_sha256,
        "net": str(args.net),
        "net_sha256": net_sha256,
        "depth": args.depth,
        "threads": args.threads,
        "hash": args.hash,
        "engine_timeout_s": args.engine_timeout_s,
        "min_ply": args.min_ply,
        "quiet_only": args.quiet_only,
        "target_task_bytes": args.target_task_bytes,
        "output": str(path),
        "output_sha256": sha256_file(path),
        "validation": validation,
        "cleaned_before_run": [str(item) for item in removed],
    }
    temporary = manifest.with_name(f".{manifest.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--net", type=Path, default=DEFAULT_NET)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=128)
    parser.add_argument("--engine-timeout-s", type=int, default=DEFAULT_ENGINE_TIMEOUT_S,
                        help="Maximum seconds allowed for one Stockfish position")
    parser.add_argument("--shards", type=int, default=None,
                        help="Override the byte-weighted per-batch task count for every batch")
    parser.add_argument("--batch-bytes", type=int, default=DEFAULT_BATCH_BYTES,
                        help="Maximum compressed archive bytes staged per sequential Forge run")
    parser.add_argument("--target-task-bytes", type=int, default=DEFAULT_TARGET_TASK_BYTES,
                        help="Target archive bytes per Forge task, applied independently per batch")
    parser.add_argument("--min-ply", type=int, default=16)
    parser.add_argument("--quiet-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-score-spread", type=int, default=DEFAULT_MIN_SCORE_SPREAD,
                        help="Reject a batch/corpus whose max-min centipawn score is below this")
    parser.add_argument("--lease-seconds", type=int, default=DEFAULT_LEASE_SECONDS)
    parser.add_argument("--poll-interval-s", type=float, default=DEFAULT_POLL_INTERVAL_S)
    parser.add_argument("--worker-stale-seconds", type=float, default=DEFAULT_WORKER_STALE_SECONDS)
    parser.add_argument("--quarantine-grace-s", type=float, default=DEFAULT_QUARANTINE_GRACE_S)
    parser.add_argument("--allow-small-input", action="store_true")
    parser.add_argument("--clean-old", action="store_true", help="Remove old lc0-root conversion products before launch")
    parser.add_argument("--clean-only", action="store_true", help="Remove old conversion products and exit")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.input = args.input.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.engine = args.engine.expanduser().resolve()
    args.net = args.net.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    archive_count, archive_bytes = preflight_source(args.input, allow_small=args.allow_small_input)
    archives = archive_paths(args.input)
    require_file(args.engine, "Stockfish engine")
    require_file(args.net, "Stockfish net")
    if (args.depth <= 0 or args.threads <= 0 or args.hash <= 0
            or args.engine_timeout_s <= 0 or args.batch_bytes <= 0 or args.target_task_bytes <= 0
            or (args.shards is not None and args.shards <= 0)):
        raise SystemExit(
            "depth, threads, hash, engine-timeout-s, batch-bytes, target-task-bytes, "
            "and shards (if given) must be positive"
        )
    template = Path(__file__).resolve().with_name("label-lc0-stockfish-enyo.template.json")
    removed: list[Path] = []
    if args.clean_only:
        removed = cleanup_old_outputs(args.output.parent)
        if LEGACY_OUTPUT_DIR != args.output.parent:
            removed.extend(cleanup_old_outputs(LEGACY_OUTPUT_DIR))
        print(json.dumps({"cleaned": [str(path) for path in removed]}, indent=2))
        return 0
    if args.clean_old and not args.dry_run:
        removed = cleanup_old_outputs(args.output.parent)
        if LEGACY_OUTPUT_DIR != args.output.parent:
            removed.extend(cleanup_old_outputs(LEGACY_OUTPUT_DIR))
    if args.output.exists() and not args.dry_run:
        raise SystemExit(f"output already exists; refusing overwrite: {args.output}")

    batches = build_batches(
        archives, batch_bytes=args.batch_bytes, target_task_bytes=args.target_task_bytes,
        shard_override=args.shards,
    )
    work_dir = work_dir_for_output(args.output)
    print(json.dumps({
        "input": str(args.input),
        "input_archive_count": archive_count,
        "input_archive_bytes": archive_bytes,
        "output": str(args.output),
        "work_dir": str(work_dir),
        "batch_count": len(batches),
        "batch_bytes_limit": args.batch_bytes,
        "target_task_bytes": args.target_task_bytes,
        "batch_shards": [batch.shard_count for batch in batches],
        "cleaned": [str(path) for path in removed],
    }, indent=2), flush=True)
    if args.dry_run:
        return 0

    engine_sha256 = sha256_file(args.engine)
    net_sha256 = sha256_file(args.net)
    work_dir.mkdir(parents=True, exist_ok=True)
    init_or_verify_work_state(work_dir, args=args, archives=archives, batches=batches)
    quarantine_dir = work_dir / "quarantine"

    batch_outputs: list[Path] = []
    for batch in batches:
        this_batch_dir = batch_dir(work_dir, batch)
        this_batch_dir.mkdir(parents=True, exist_ok=True)
        validation_path = this_batch_dir / "batch-validation.json"
        existing_validation = read_json_if_valid(validation_path)
        if existing_validation is not None and existing_validation.get("valid") is True:
            print(json.dumps({"batch": batch.index, "reused_valid": True}), flush=True)
            batch_outputs.append(this_batch_dir / "output.bullet")
            continue

        stored_plan = read_json_if_valid(this_batch_dir / "plan.json")
        plan = write_batch_plan(work_dir, batch)
        if stored_plan is not None and stored_plan.get("digest") != plan["digest"]:
            raise SystemExit(
                f"batch {batch.index}: source archives changed since the last run; "
                f"refusing to silently re-plan. Work directory: {this_batch_dir}"
            )
        print(json.dumps({"batch": batch.index, "batches": len(batches), "plan": plan}), flush=True)

        run_name, manifest_path = ensure_batch_launched(
            args, template, batch, this_batch_dir,
            net_sha256=net_sha256,
        )
        wait_with_quarantine(
            run_name, manifest_path,
            lease_seconds=args.lease_seconds,
            poll_interval_s=args.poll_interval_s,
            worker_stale_seconds=args.worker_stale_seconds,
            grace_seconds=args.quarantine_grace_s,
            quarantine_dir=quarantine_dir,
        )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shard_paths = validate_shards(
            this_batch_dir, manifest,
            expected_net_sha256=net_sha256,
        )
        validate_batch(this_batch_dir, shard_paths, min_score_spread=args.min_score_spread)
        removed_sources = cleanup_unpacked_lc0_sources(manifest)
        if removed_sources:
            print(json.dumps({"batch": batch.index, "cleaned_unpacked_lc0": removed_sources}), flush=True)
        batch_outputs.append(this_batch_dir / "output.bullet")

    final_dir = work_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(final_dir / "merge-plan.json", {
        "batches": [str(path) for path in batch_outputs],
    })
    validation = validate_and_merge(
        batch_outputs,
        merge_output=args.output,
        require_win_loss=True,
        min_score_spread=args.min_score_spread,
    )
    manifest_path_out = write_provenance(
        args.output, args=args, archive_count=archive_count, archive_bytes=archive_bytes,
        validation=validation, removed=removed, engine_sha256=engine_sha256, net_sha256=net_sha256,
    )
    print(json.dumps({
        "output": str(args.output), "manifest": str(manifest_path_out), "validation": validation,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
