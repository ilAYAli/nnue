#!/usr/bin/env python3
"""Count NNUE training positions recorded in git history."""

from __future__ import annotations

import argparse
import json
import re
import subprocess


CHECKPOINT_RE = re.compile(r"trained pos:\s*([0-9,]+)", re.IGNORECASE)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def json_at(commit: str, path: str) -> dict[str, object]:
    return json.loads(git("show", f"{commit}:{path}"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count trained positions since the repository's scratch root"
    )
    parser.add_argument("--verbose", action="store_true", help="show each added run")
    args = parser.parse_args()

    checkpoint = ""
    baseline = 0
    for line in git("log", "--reverse", "--format=%H%x00%s").splitlines():
        commit, subject = line.split("\0", 1)
        match = CHECKPOINT_RE.search(subject)
        if match:
            checkpoint = commit
            baseline = int(match.group(1).replace(",", ""))

    if not checkpoint:
        raise SystemExit("no 'trained pos' checkpoint found in git history")

    total = baseline
    seen: set[str] = set()
    entries: list[tuple[str, int]] = []
    history = git("log", "--reverse", "--format=%H%x00%s", f"{checkpoint}..HEAD")
    for line in history.splitlines():
        commit, subject = line.split("\0", 1)
        try:
            build = json_at(commit, "build.json")
            defaults = json_at(commit, "defaults.json")
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            continue

        run = build.get("run")
        if not isinstance(run, str) or run in seen or not subject.startswith(run):
            continue

        superbatches = int(build.get("superbatches", defaults["superbatches"]))
        batch_size = int(build.get("batch_size", defaults["batch_size"]))
        batches = int(build.get("batches", defaults["batches"]))
        positions = superbatches * batch_size * batches
        seen.add(run)
        entries.append((run, positions))
        total += positions

    if args.verbose:
        print(f"checkpoint {checkpoint[:12]}: {baseline:,}")
        for run, positions in entries:
            print(f"{run}: {positions:,}")
    print(f"{total:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
