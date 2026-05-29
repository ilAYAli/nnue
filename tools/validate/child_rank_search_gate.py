#!/usr/bin/env python3
"""Gate child-ranking targets with root search-selected moves."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import queue
import statistics
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.child_rank_targets import ChildRankGroup, load_groups, target_gap_cp


def resolve_net_path(net: Path) -> Path:
    path = Path(os.path.expandvars(str(net))).expanduser()
    if not path.is_file():
        raise RuntimeError(f"NNUE file not found: {path}")
    return path.resolve()


class SearchEngine:
    def __init__(self, engine: Path, net: Path, threads: int, hash_mb: int):
        net = resolve_net_path(net)
        self.proc = subprocess.Popen(
            [str(engine)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("failed to open engine pipes")
        self.stdin = self.proc.stdin
        self.stdout = self.proc.stdout
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.reader = threading.Thread(target=self._read_stdout, daemon=True)
        self.reader.start()
        self.send("uci")
        self.wait_for("uciok", 10.0)
        self.send(f"setoption name Threads value {threads}")
        self.send(f"setoption name Hash value {hash_mb}")
        self.load_nnue(net)

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.send("quit")
            except Exception:
                pass
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines.put(line.rstrip("\n"))
        self.lines.put(None)

    def send(self, command: str) -> None:
        self.stdin.write(command + "\n")
        self.stdin.flush()

    def read_line(self, timeout: float) -> str:
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty as exc:
            raise RuntimeError("engine read timed out") from exc
        if line is None:
            raise RuntimeError("engine exited")
        return line

    def wait_for(self, token: str, timeout: float) -> None:
        while True:
            line = self.read_line(timeout)
            if token in line:
                return

    def load_nnue(self, net: Path) -> None:
        self.send("debug on")
        self.send(f"setoption name nnue_file value {net}")
        self.send("isready")
        loaded = False
        warnings: list[str] = []
        while True:
            line = self.read_line(20.0)
            if "WARNING: nnue_file" in line:
                warnings.append(line)
            if (
                ("network loaded from" in line or "embedded evaluator loaded from" in line)
                and str(net) in line
            ):
                loaded = True
            if "readyok" in line:
                break
        if warnings:
            raise RuntimeError("; ".join(warnings))
        if not loaded:
            raise RuntimeError(f"engine did not confirm NNUE load: {net}")

    def newgame(self) -> None:
        self.send("ucinewgame")
        self.send("isready")
        self.wait_for("readyok", 20.0)

    def bestmove(self, fen: str, nodes: int, depth: int) -> str | None:
        self.newgame()
        self.send(f"position fen {fen}")
        if nodes > 0:
            self.send(f"go nodes {nodes}")
        else:
            self.send(f"go depth {depth}")
        while True:
            line = self.read_line(120.0)
            if line.startswith("bestmove "):
                move = line.split()[1]
                return None if move == "(none)" else move


def group_result(engine: SearchEngine, group: ChildRankGroup, nodes: int, depth: int) -> dict:
    selected = engine.bestmove(group.fen, nodes, depth)
    move_by_uci = {move.move: move for move in group.moves}
    missing = selected not in move_by_uci
    selected_gap = None
    if selected in move_by_uci:
        selected_gap = target_gap_cp(group, move_by_uci[selected])
    return {
        "group_id": group.group_id,
        "best": group.best_move,
        "selected": selected or "",
        "correct": selected == group.best_move,
        "missing": missing,
        "selected_gap": selected_gap,
    }


def chunks(groups: list[ChildRankGroup], jobs: int) -> list[list[ChildRankGroup]]:
    if jobs <= 1 or len(groups) <= 1:
        return [groups]
    jobs = min(jobs, len(groups))
    size = (len(groups) + jobs - 1) // jobs
    return [groups[i:i + size] for i in range(0, len(groups), size)]


def chunk_results(
    groups: list[ChildRankGroup],
    engine_path: Path,
    net_path: Path,
    threads: int,
    hash_mb: int,
    nodes: int,
    depth: int,
) -> list[dict]:
    engine = SearchEngine(engine_path, net_path, threads, hash_mb)
    try:
        return [group_result(engine, group, nodes, depth) for group in groups]
    finally:
        engine.close()


def evaluate(
    groups: list[ChildRankGroup],
    engine_path: Path,
    net_path: Path,
    threads: int,
    hash_mb: int,
    jobs: int,
    nodes: int,
    depth: int,
) -> list[dict]:
    work = chunks(groups, jobs)
    if len(work) == 1:
        return chunk_results(work[0], engine_path, net_path, threads, hash_mb,
                             nodes, depth)
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(work)) as executor:
        futures = [
            executor.submit(
                chunk_results, chunk, engine_path, net_path, threads, hash_mb,
                nodes, depth)
            for chunk in work
        ]
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
    return results


def summarize(label: str, results: list[dict]) -> dict[str, float]:
    total = len(results)
    top1 = sum(1 for result in results if result["correct"])
    missing = sum(1 for result in results if result["missing"])
    scored = [float(result["selected_gap"]) for result in results
              if result["selected_gap"] is not None]
    sum_gap = sum(scored)
    worst_gap = max(scored, default=0.0)
    print(
        f"{label} search_top1={top1}/{total} missing_selected={missing} "
        f"scored_sum_gap={sum_gap:.0f} worst_gap={worst_gap:.0f}",
        flush=True,
    )
    return {
        "top1": float(top1),
        "total": float(total),
        "missing": float(missing),
        "sum_gap": sum_gap,
        "worst_gap": worst_gap,
    }


def compare(candidate: list[dict], reference: list[dict]) -> dict[str, float]:
    by_id = {str(result["group_id"]): result for result in reference}
    diffs = []
    candidate_better = 0
    reference_better = 0
    equal = 0
    skipped_missing = 0
    for cand in candidate:
        ref = by_id[str(cand["group_id"])]
        if cand["selected_gap"] is None or ref["selected_gap"] is None:
            skipped_missing += 1
            continue
        diff = float(ref["selected_gap"]) - float(cand["selected_gap"])
        diffs.append(diff)
        if diff > 0:
            candidate_better += 1
        elif diff < 0:
            reference_better += 1
        else:
            equal += 1
    median = statistics.median(diffs) if diffs else 0.0
    worst_regression = min(diffs, default=0.0)
    best_gain = max(diffs, default=0.0)
    sum_diff = sum(diffs)
    print(
        "compare "
        f"scored={len(diffs)} skipped_missing={skipped_missing} "
        f"candidate_better={candidate_better} "
        f"reference_better={reference_better} equal={equal} "
        f"sum_diff={sum_diff:.0f} median_diff={median:.0f} "
        f"worst_regression={worst_regression:.0f} best_gain={best_gain:.0f}",
        flush=True,
    )
    return {
        "scored": float(len(diffs)),
        "skipped_missing": float(skipped_missing),
        "candidate_better": float(candidate_better),
        "reference_better": float(reference_better),
        "equal": float(equal),
        "sum_diff": sum_diff,
        "median_diff": median,
        "worst_regression": worst_regression,
        "best_gain": best_gain,
    }


def write_details(
    path: Path,
    groups: list[ChildRankGroup],
    candidate: list[dict],
    reference: list[dict] | None,
) -> None:
    by_group = {group.group_id: group for group in groups}
    reference_by_id = (
        {str(result["group_id"]): result for result in reference}
        if reference is not None else {}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for cand in candidate:
            group = by_group[str(cand["group_id"])]
            ref = reference_by_id.get(str(cand["group_id"]))
            diff = None
            relation = "no_reference"
            if ref is not None:
                if cand["selected_gap"] is None or ref["selected_gap"] is None:
                    relation = "missing"
                else:
                    diff = float(ref["selected_gap"]) - float(cand["selected_gap"])
                    if diff > 0:
                        relation = "candidate_better"
                    elif diff < 0:
                        relation = "reference_better"
                    else:
                        relation = "equal"
            row = {
                "schema": "enyo.child_rank_search_gate.v1",
                "group_id": group.group_id,
                "fen": group.fen,
                "tags": list(group.tags),
                "best_move": group.best_move,
                "candidate": {
                    "selected": cand["selected"],
                    "correct": cand["correct"],
                    "missing": cand["missing"],
                    "gap_cp": cand["selected_gap"],
                },
                "relation": relation,
                "diff_cp": diff,
            }
            if ref is not None:
                row["reference"] = {
                    "selected": ref["selected"],
                    "correct": ref["correct"],
                    "missing": ref["missing"],
                    "gap_cp": ref["selected_gap"],
                }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, type=Path)
    ap.add_argument("--engine", required=True, type=Path)
    ap.add_argument("--net", required=True, type=Path)
    ap.add_argument("--reference-net", type=Path)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--nodes", type=int, default=20000)
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--min-groups", type=int, default=1)
    ap.add_argument("--fail-if-top1-below", type=int, default=-1)
    ap.add_argument("--fail-if-missing-above", type=int, default=-1)
    ap.add_argument("--fail-if-reference-better-above", type=int, default=-1)
    ap.add_argument("--fail-if-sum-diff-below", type=float, default=None)
    ap.add_argument("--out-jsonl", type=Path)
    args = ap.parse_args()

    if args.nodes <= 0 and args.depth <= 0:
        raise SystemExit("either --nodes or --depth must be positive")

    groups = load_groups(args.targets, min_groups=args.min_groups)
    candidate = evaluate(
        groups, args.engine, args.net, args.threads, args.hash,
        max(1, args.jobs), args.nodes, args.depth)
    candidate_summary = summarize("candidate", candidate)
    compare_summary = None
    if args.reference_net:
        reference = evaluate(
            groups, args.engine, args.reference_net, args.threads, args.hash,
            max(1, args.jobs), args.nodes, args.depth)
        summarize("reference", reference)
        compare_summary = compare(candidate, reference)
    else:
        reference = None

    if args.out_jsonl:
        write_details(args.out_jsonl, groups, candidate, reference)

    failed = False
    if (
        args.fail_if_top1_below >= 0
        and candidate_summary["top1"] < args.fail_if_top1_below
    ):
        failed = True
    if (
        args.fail_if_missing_above >= 0
        and candidate_summary["missing"] > args.fail_if_missing_above
    ):
        failed = True
    if (
        compare_summary is not None
        and args.fail_if_reference_better_above >= 0
        and compare_summary["reference_better"] > args.fail_if_reference_better_above
    ):
        failed = True
    if (
        compare_summary is not None
        and args.fail_if_sum_diff_below is not None
        and compare_summary["sum_diff"] < args.fail_if_sum_diff_below
    ):
        failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
