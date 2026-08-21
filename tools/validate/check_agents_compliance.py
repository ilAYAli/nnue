#!/usr/bin/env python3
"""Pre-flight check for the mechanically-verifiable subset of AGENTS.md.

Most of AGENTS.md is operator policy (never poll, one variable per
iteration, read IMPROVEMENT_PLAN.md first, ...) that no script can check
from repo state alone. This covers only what actually can be checked
mechanically from build.json/architecture.json/LINEAGE.md/git, without
touching the trainer itself:

- rule 8:  build.json sets at most one of continue_from/initialize_from.
- rule 11: run name matches enyo-{A}.{P}.0-rc{N}.
- rule 14: the run name is reserved in LINEAGE.md.
- rule 19: build.json has a non-empty hypothesis.
- rule 41: the latest commit has no AI/bot co-author trailer.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN_NAME_RE = re.compile(r"^enyo-\d+\.\d+\.0-rc\d+$")
COAUTHOR_RE = re.compile(r"Co-Authored-By:.*(claude|copilot|codex|gpt|ai)", re.IGNORECASE)


def check_origin_fields(build: dict) -> list[str]:
    has_continue = "continue_from" in build
    has_init = "initialize_from" in build
    if has_continue and has_init:
        return ["rule 8: build.json sets both continue_from and "
                "initialize_from; exactly one (or neither, for an "
                "approved scratch root) is allowed"]
    return []


def check_run_name(build: dict) -> list[str]:
    run = build.get("run")
    if not run:
        return ["rule 11: build.json is missing \"run\""]
    if not RUN_NAME_RE.match(run):
        return [f"rule 11: run name {run!r} does not match "
                "enyo-{architecture}.{promotion}.0-rc{iteration}"]
    return []


def check_reserved(build: dict, lineage_text: str) -> list[str]:
    run = build.get("run")
    if not run:
        return []
    if run not in lineage_text:
        return [f"rule 14: run name {run!r} is not mentioned in LINEAGE.md "
                "(reserve it before launch)"]
    return []


def check_hypothesis(build: dict) -> list[str]:
    hypothesis = build.get("hypothesis", "").strip()
    if not hypothesis:
        return ["rule 19: build.json is missing a non-empty \"hypothesis\""]
    return []


def check_latest_commit_coauthor() -> list[str]:
    message = subprocess.run(
        ["git", "-C", str(REPO), "log", "-1", "--format=%B"],
        capture_output=True, text=True, check=True).stdout
    if COAUTHOR_RE.search(message):
        return ["rule 41: the latest commit has an AI/bot co-author trailer"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default=str(REPO / "build.json"), type=Path)
    parser.add_argument("--lineage", default=str(REPO / "LINEAGE.md"), type=Path)
    args = parser.parse_args()

    build = json.loads(args.build.read_text())
    lineage_text = args.lineage.read_text()

    findings: list[str] = []
    findings += check_origin_fields(build)
    findings += check_run_name(build)
    findings += check_reserved(build, lineage_text)
    findings += check_hypothesis(build)
    findings += check_latest_commit_coauthor()

    if findings:
        for finding in findings:
            print(f"FAIL: {finding}")
        return 1

    print("PASS: all mechanically-checkable rules satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
