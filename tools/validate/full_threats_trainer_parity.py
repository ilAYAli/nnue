#!/usr/bin/env python3
"""Compare Rust trainer FullThreats indices with the Python reference."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import enyo_nnue as nn2


FENS = (
    "rn1qkbnr/ppp2ppp/3p4/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 5",
    "r3k2r/pp1n1ppp/2pbpn2/q7/3P4/2N1PN2/PPQB1PPP/R3KB1R w KQkq - 3 10",
    "2r2rk1/1bqnbppp/p2ppn2/1p6/3NP3/1BN1B3/PPPQ1PPP/2KR3R w - - 4 14",
    "4k3/8/3p4/2pPp3/2P1P3/8/8/4K3 w - - 0 1",
    "8/8/2k5/2P1p3/4Pp2/5P2/4K3/8 w - - 0 54",
    "rn1qkbnr/ppp2ppp/3p4/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R b KQkq - 0 5",
    "r3k2r/pp1n1ppp/2pbpn2/q7/3P4/2N1PN2/PPQB1PPP/R3KB1R b KQkq - 3 10",
    "2r2rk1/1bqnbppp/p2ppn2/1p6/3NP3/1BN1B3/PPPQ1PPP/2KR3R b - - 4 14",
    "4k3/8/3p4/2pPp3/2P1P3/8/8/4K3 b - - 0 1",
    "8/8/2k5/2P1p3/4Pp2/5P2/4K3/8 b - - 0 54",
)


def python_threats(fen: str) -> tuple[list[int], list[int]]:
    pieces, stm = nn2.parse_fen(fen)
    pieces.sort(key=lambda item: item[2])
    ntm = nn2.BLACK if stm == nn2.WHITE else nn2.WHITE
    return (
        nn2.threat_features_from_pieces(pieces, stm),
        nn2.threat_features_from_pieces(pieces, ntm),
    )


def rust_threats(repo: Path, fens: list[str]) -> list[dict[str, object]]:
    manifest = repo / "tools/bullet/spike_trainer/Cargo.toml"
    command = [
        "cargo",
        "run",
        "--quiet",
        "--manifest-path",
        str(manifest),
        "--bin",
        "dump_enyo_threats",
        "--",
        *fens,
    ]
    result = subprocess.run(
        command,
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def summarize_diff(left: list[int], right: list[int]) -> str:
    left_set = set(left)
    right_set = set(right)
    left_only = sorted(left_set - right_set)[:20]
    right_only = sorted(right_set - left_set)[:20]
    return (
        f"python_len={len(left)} rust_len={len(right)} "
        f"python_only={left_only} rust_only={right_only}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fen", action="append", default=[])
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()

    fens = args.fen or list(FENS)
    rows = rust_threats(args.repo, fens)
    failures = 0
    for fen, row in zip(fens, rows, strict=True):
        py_stm, py_ntm = python_threats(fen)
        rs_stm = row["stm"]
        rs_ntm = row["ntm"]
        ok = py_stm == rs_stm and py_ntm == rs_ntm
        failures += 0 if ok else 1
        print(
            f"{'PASS' if ok else 'FAIL'} stm={len(py_stm)} ntm={len(py_ntm)} fen={fen}",
            flush=True,
        )
        if not ok:
            print(f"  stm {summarize_diff(py_stm, rs_stm)}", flush=True)
            print(f"  ntm {summarize_diff(py_ntm, rs_ntm)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
