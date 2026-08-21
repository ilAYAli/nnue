#!/usr/bin/env python3
"""Launch and validate a guarded two- or three-host Bullet local-SGD smoke test.

Run this from pwa-llm's existing nnue_cmd session.  It never edits build.json,
creates tmux sessions, or terminates remote jobs.  The default is a dry run;
--launch is required to start the test.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
from typing import TextIO


ROOT = Path(__file__).resolve().parents[2]
TRAINER_MANIFEST = "tools/bullet/spike_trainer/Cargo.toml"
TRAINER_BIN = "tools/bullet/spike_trainer/target/release/train"
FINGERPRINT_FILES = (
    "tools/bullet/spike_trainer/Cargo.toml",
    "tools/bullet/spike_trainer/Cargo.lock",
    "tools/bullet/spike_trainer/src/lib.rs",
    "tools/bullet/spike_trainer/src/main.rs",
    "tools/bullet/spike_trainer/src/bin/train.rs",
    "tools/bullet/spike_trainer/src/distributed/checkpoint.rs",
    "tools/bullet/spike_trainer/src/distributed/coordinator.rs",
    "tools/bullet/spike_trainer/src/distributed/mod.rs",
    "tools/bullet/spike_trainer/src/distributed/protocol.rs",
    "tools/bullet/spike_trainer/src/distributed/worker.rs",
    "tools/bullet/bullet-patched/crates/bullet_lib/src/value.rs",
)


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def remote_path(path: str) -> str:
    if path == "~":
        return "$HOME"
    if path.startswith("~/"):
        return "$HOME/" + shlex.quote(path[2:])
    return shlex.quote(path)


def remote(host: str, repo: str, command: str, *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    remote_command = f"cd {remote_path(repo)} && {command}"
    return subprocess.run(
        ["ssh", host, f"bash -lc {shlex.quote(remote_command)}"],
        check=False,
        text=True,
        capture_output=capture,
    )


def fingerprint(host: str, repo: str) -> str:
    command = "sha256sum " + shell_join(list(FINGERPRINT_FILES)) + " | sha256sum"
    result = remote(host, repo, command, capture=True)
    if result.returncode:
        fail(f"cannot fingerprint {host}: {result.stderr.strip() or result.stdout.strip()}")
    fields = result.stdout.split()
    if not fields:
        fail(f"{host} returned an empty source fingerprint")
    return fields[0]


def remote_file_hash(host: str, repo: str, path: str) -> str:
    result = remote(host, repo, f"sha256sum {remote_path(path)}", capture=True)
    if result.returncode:
        fail(f"cannot hash {path} on {host}: {result.stderr.strip() or result.stdout.strip()}")
    fields = result.stdout.split()
    if not fields:
        fail(f"{host} returned an empty hash for {path}")
    return fields[0]


def require_remote_file(host: str, repo: str, path: str) -> None:
    result = remote(host, repo, f"test -s {remote_path(path)}", capture=True)
    if result.returncode:
        fail(f"missing or empty training shard on {host}: {path}")


def active_trainings(host: str, repo: str) -> str:
    result = remote(
        host,
        repo,
        "pgrep -af 'tools/bullet/spike_trainer/target/release/train|tools/bullet/train' || true",
        capture=True,
    )
    return result.stdout.strip()


def build_command() -> str:
    return shell_join([
        "cargo", "build", "--release", "--manifest-path", TRAINER_MANIFEST,
        "--bin", "train", "--features", "cuda",
    ])


def build_run_command(
    *,
    role: str,
    run: str,
    node: str,
    data: str,
    build: str,
    coordinator: str,
    port: int,
    peers: int,
    sync_every: int,
    timeout: int,
) -> str:
    env = {
        "ENYO_BULLET_DISTRIBUTED_ROLE": role,
        "ENYO_BULLET_DISTRIBUTED_RUN_ID": run,
        "ENYO_BULLET_DISTRIBUTED_NODE_ID": node,
        "ENYO_BULLET_DISTRIBUTED_DATA": data,
        "ENYO_BULLET_DISTRIBUTED_SYNC_EVERY": str(sync_every),
        "ENYO_BULLET_DISTRIBUTED_TIMEOUT_SECS": str(timeout),
    }
    if role == "coordinator":
        env["ENYO_BULLET_DISTRIBUTED_LISTEN_ADDR"] = f"0.0.0.0:{port}"
        env["ENYO_BULLET_DISTRIBUTED_NUM_PEERS"] = str(peers)
    else:
        env["ENYO_BULLET_DISTRIBUTED_COORDINATOR_ADDR"] = f"{coordinator}:{port}"

    exports = " ".join(f"{name}={shlex.quote(value)}" for name, value in env.items())
    run_cmd = shell_join([TRAINER_BIN, "run", "--build", build])
    return f"exec env {exports} {run_cmd}"


def stream_output(host: str, process: subprocess.Popen[str], log: TextIO) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        log.write(line)
        log.flush()
        print(f"[{host}] {line}", end="", flush=True)


def latest_checkpoint_hash(host: str, repo: str, run: str) -> tuple[str, str]:
    run_path = shlex.quote(f"runs/{run}")
    command = (
        f"latest=$(find {run_path} -type f -name quantised.bin -print | sort | tail -n 1); "
        "test -n \"$latest\" && sha256sum \"$latest\""
    )
    result = remote(host, repo, command, capture=True)
    if result.returncode:
        fail(f"cannot find final checkpoint on {host}: {result.stderr.strip() or result.stdout.strip()}")
    fields = result.stdout.split(maxsplit=1)
    if len(fields) != 2:
        fail(f"invalid final checkpoint hash from {host}: {result.stdout.strip()}")
    return fields[0], fields[1].strip()


def validate_logs(logs: dict[str, Path], coordinator: str, workers: list[str], final_superbatch: int) -> None:
    for host, path in logs.items():
        text = path.read_text(encoding="utf-8", errors="replace")
        required = "distributed coordinator ready" if host == coordinator else "distributed worker ready"
        if required not in text or f"distributed sync: run=" not in text or f"superbatch={final_superbatch}" not in text:
            fail(f"{host} did not complete the distributed startup/sync evidence; inspect {path}")
    for host in workers:
        if host == coordinator:
            fail("the coordinator must not also be listed as a worker")


def validate_workers(coordinator: str, workers: list[str]) -> None:
    if not workers or len(set(workers)) != len(workers):
        fail("supply one or more distinct --worker hosts")
    if coordinator in workers:
        fail("the coordinator cannot be a worker")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default="build.json", help="Same build file on every host.")
    parser.add_argument("--repo", default="~/code/chess/nnue", help="Repository path on every host.")
    parser.add_argument("--coordinator", default="pwa-llm")
    parser.add_argument("--worker", action="append", default=[], help="Worker host; repeat for additional workers.")
    parser.add_argument("--coordinator-data", required=True, help="Prepared local shard on the coordinator.")
    parser.add_argument("--worker-data", action="append", default=[], metavar="HOST=PATH")
    parser.add_argument("--port", type=int, default=9219)
    parser.add_argument("--sync-every", type=int, default=1)
    parser.add_argument("--timeout-secs", type=int, default=180)
    parser.add_argument("--max-superbatches", type=int, default=4, help="Refuse a non-smoke build above this dose.")
    parser.add_argument("--launch", action="store_true", help="Actually build and start the remote trainers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workers = args.worker or ["pwa-hak"]
    validate_workers(args.coordinator, workers)
    if args.port <= 0 or args.port > 65535 or args.sync_every <= 0 or args.timeout_secs <= 0 or args.max_superbatches <= 0:
        fail("--port, --sync-every, --timeout-secs, and --max-superbatches must be positive")

    build_path = Path(args.build)
    if not build_path.is_absolute():
        build_path = ROOT / build_path
    try:
        build_config = json.loads(build_path.read_text(encoding="utf-8"))
        run = build_config["run"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        fail(f"cannot read a run name from {build_path}: {error}")
    if not isinstance(run, str) or not run:
        fail(f"{build_path} has no non-empty run name")
    defaults = json.loads((ROOT / "defaults.json").read_text(encoding="utf-8"))
    superbatches = build_config.get("superbatches", defaults.get("superbatches"))
    if not isinstance(superbatches, int) or superbatches <= 0:
        fail("the resolved superbatches value must be a positive integer")
    if superbatches > args.max_superbatches:
        fail(
            f"refusing {superbatches} superbatches for a smoke test; "
            f"prepare a short build or raise --max-superbatches explicitly"
        )
    lineage = (ROOT / "LINEAGE.md").read_text(encoding="utf-8")
    if not any(line.startswith("Reserved:") and f"`{run}`" in line for line in lineage.splitlines()):
        fail(f"{run} is not reserved in LINEAGE.md")

    data_by_host: dict[str, str] = {args.coordinator: args.coordinator_data}
    for item in args.worker_data:
        host, separator, data = item.partition("=")
        if not separator or not host or not data:
            fail(f"--worker-data must be HOST=PATH, got {item!r}")
        if host in data_by_host:
            fail(f"duplicate shard for {host}")
        data_by_host[host] = data
    if set(data_by_host) != {args.coordinator, *workers}:
        fail("provide one --worker-data HOST=PATH for each worker")

    hosts = [args.coordinator, *workers]
    for host in hosts:
        require_remote_file(host, args.repo, data_by_host[host])
    fingerprints = {host: fingerprint(host, args.repo) for host in hosts}
    if len(set(fingerprints.values())) != 1:
        detail = ", ".join(f"{host}={digest}" for host, digest in fingerprints.items())
        fail(f"trainer source differs across hosts: {detail}")
    build_hashes = {host: remote_file_hash(host, args.repo, args.build) for host in hosts}
    if len(set(build_hashes.values())) != 1:
        detail = ", ".join(f"{host}={digest}" for host, digest in build_hashes.items())
        fail(f"build configuration differs across hosts: {detail}")
    print(
        f"run={run}\nsuperbatches={superbatches}\n"
        f"source_sha256={next(iter(fingerprints.values()))}\n"
        f"build_sha256={next(iter(build_hashes.values()))}"
    )

    commands = {
        args.coordinator: build_run_command(
            role="coordinator", run=run, node=args.coordinator, data=data_by_host[args.coordinator],
            build=args.build, coordinator=args.coordinator, port=args.port, peers=len(workers),
            sync_every=args.sync_every, timeout=args.timeout_secs,
        ),
        **{
            worker: build_run_command(
                role="worker", run=run, node=worker, data=data_by_host[worker], build=args.build,
                coordinator=args.coordinator, port=args.port, peers=len(workers),
                sync_every=args.sync_every, timeout=args.timeout_secs,
            )
            for worker in workers
        },
    }
    for host in hosts:
        remote_command = f"cd {remote_path(args.repo)} && {commands[host]}"
        print(f"\n[{host}]\nssh {shlex.quote(host)} bash -lc {shlex.quote(remote_command)}")
    if not args.launch:
        print("\ndry run only; rerun from the existing nnue_cmd session with --launch to start the smoke test")
        return
    if not os.environ.get("TMUX"):
        fail("--launch must run from the existing nnue_cmd tmux session")
    session = subprocess.run(
        ["tmux", "display-message", "-p", "#{session_name}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if session.returncode or session.stdout.strip() != "nnue_cmd":
        fail("--launch must run in the nnue_cmd tmux session")
    for host in hosts:
        active = active_trainings(host, args.repo)
        if active:
            fail(f"refusing to overlap an active Bullet trainer on {host}: {active}")
    for host in hosts:
        print(f"building trainer on {host}", flush=True)
        result = remote(host, args.repo, build_command())
        if result.returncode:
            fail(f"cannot build the trainer on {host}")

    log_dir = ROOT / "runs" / run / "distributed-smoke" / dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    log_dir.mkdir(parents=True, exist_ok=False)
    processes: dict[str, subprocess.Popen[str]] = {}
    logs: dict[str, Path] = {}
    log_files: list[TextIO] = []
    threads: list[threading.Thread] = []
    for host in hosts:
        log_path = log_dir / f"{host}.log"
        logs[host] = log_path
        process = subprocess.Popen(
            ["ssh", host, f"bash -lc {shlex.quote(f'cd {remote_path(args.repo)} && {commands[host]}')}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes[host] = process
        log = log_path.open("w", encoding="utf-8")
        log_files.append(log)
        thread = threading.Thread(target=stream_output, args=(host, process, log), daemon=True)
        thread.start()
        threads.append(thread)
    statuses = {host: process.wait() for host, process in processes.items()}
    for thread in threads:
        thread.join()
    for log in log_files:
        log.close()
    if any(statuses.values()):
        fail(f"distributed smoke failed: statuses={statuses}; logs={log_dir}")

    validate_logs(logs, args.coordinator, workers, superbatches)
    hashes = {host: latest_checkpoint_hash(host, args.repo, run) for host in hosts}
    if len({digest for digest, _ in hashes.values()}) != 1:
        detail = ", ".join(f"{host}={digest} ({path})" for host, (digest, path) in hashes.items())
        fail(f"final checkpoints differ: {detail}; logs={log_dir}")
    digest = next(iter(hashes.values()))[0]
    print(f"\nPASS: synchronized final checkpoint sha256={digest}\nlogs={log_dir}")


if __name__ == "__main__":
    main()
