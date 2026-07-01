#!/usr/bin/env python3
"""Count NNUE training positions recorded in git history."""

from __future__ import annotations

import argparse
import json
import re
import subprocess


CHECKPOINT_RE = re.compile(r"trained pos:\s*([0-9,]+)", re.IGNORECASE)
ACCEPTED_RE = re.compile(r"^(.+?)(?:\.nn)?:\s+(?:Elo\b|accept(?:ed)?\b)")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def json_at(commit: str, path: str) -> dict[str, object]:
    return json.loads(git("show", f"{commit}:{path}"))


def parent_run(build: dict[str, object]) -> str | None:
    value = build.get("initialize_from") or build.get("continue_from")
    if not isinstance(value, str) or not value:
        return None
    return value.rsplit("/", 1)[-1].removesuffix(".nn")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count trained positions since the repository's scratch root"
    )
    parser.add_argument("--verbose", action="store_true", help="show each added run")
    parser.add_argument("--run", help="show the lineage ending at this run")
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

    checkpoint_run = json_at(checkpoint, "build.json").get("run")
    if not isinstance(checkpoint_run, str):
        raise SystemExit(f"{checkpoint}: build.json has no run")

    total = baseline
    seen: set[str] = set()
    entries: dict[str, tuple[str | None, int]] = {}
    accepted_run = ""
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
        entries[run] = (parent_run(build), positions)
        total += positions

        match = ACCEPTED_RE.match(subject)
        if match and match.group(1) == run:
            accepted_run = run

    target = args.run or accepted_run
    if not target:
        raise SystemExit("no accepted run found in git history")

    lineage: list[tuple[str, int]] = []
    lineage_total = 0
    run: str | None = target
    while run:
        if run == checkpoint_run:
            lineage.append((run, baseline))
            lineage_total += baseline
            break
        if any(name == run for name, _ in lineage):
            raise SystemExit(f"lineage cycle at {run}")
        try:
            parent, positions = entries[run]
        except KeyError:
            raise SystemExit(f"lineage run not found: {run}") from None
        lineage.append((run, positions))
        lineage_total += positions
        run = parent

    if args.verbose:
        print(f"checkpoint {checkpoint_run} ({checkpoint[:12]}): {baseline:,}")
        print("all experiments:")
        for run, (_, positions) in entries.items():
            print(f"{run}: {positions:,}")
        print(f"accepted lineage ({target}):")
        for run, positions in reversed(lineage):
            print(f"{run}: {positions:,}")
    print(f"total_exposures={total:,}")
    print(f"accepted_run={target}")
    print(f"accepted_lineage_exposures={lineage_total:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
