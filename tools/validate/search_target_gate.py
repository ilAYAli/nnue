#!/usr/bin/env python3
"""Run an engine on search-aware JSONL targets."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from pathlib import Path


def send(proc: subprocess.Popen[str], command: str) -> None:
    assert proc.stdin is not None
    proc.stdin.write(command + "\n")
    proc.stdin.flush()


def read_until(proc: subprocess.Popen[str], prefix: str) -> str:
    assert proc.stdout is not None
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(f"engine exited before {prefix!r}")
        line = line.strip()
        if line.startswith(prefix):
            return line


def read_until_lines(proc: subprocess.Popen[str], prefix: str) -> list[str]:
    assert proc.stdout is not None
    lines: list[str] = []
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(f"engine exited before {prefix!r}")
        line = line.strip()
        lines.append(line)
        if line.startswith(prefix):
            return lines


class UciEngine:
    def __init__(self, engine: Path, *, nnue_file: Path | None,
                 threads: int, hash_mb: int,
                 require_native_net_load: bool) -> None:
        self.proc = subprocess.Popen(
            [str(engine)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)
        send(self.proc, "uci")
        read_until(self.proc, "uciok")
        if require_native_net_load:
            send(self.proc, "debug on")
        send(self.proc, f"setoption name Threads value {threads}")
        send(self.proc, f"setoption name Hash value {hash_mb}")
        if nnue_file:
            send(self.proc, f"setoption name nnue_file value {nnue_file}")
        send(self.proc, "isready")
        ready_lines = read_until_lines(self.proc, "readyok")
        if nnue_file and require_native_net_load:
            text = "\n".join(ready_lines)
            if "network loaded from" not in text:
                raise RuntimeError(
                    "engine did not report native NNUE load for "
                    f"{nnue_file}; ready output was:\n{text}")
            if (
                "embedded evaluator loaded from" in text
                or "nnue: loaded network" in text
            ):
                raise RuntimeError(
                    "engine used legacy/embedded NNUE loader for "
                    f"{nnue_file}; ready output was:\n{text}")

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                send(self.proc, "quit")
            except Exception:
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def bestmove(self, fen: str, nodes: int, *, reset_game: bool) -> tuple[str, int]:
        if reset_game:
            send(self.proc, "ucinewgame")
            send(self.proc, "isready")
            read_until(self.proc, "readyok")
        send(self.proc, f"position fen {fen}")
        send(self.proc, f"go nodes {nodes}")
        nodes_seen = 0
        assert self.proc.stdout is not None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine exited before bestmove")
            line = line.strip()
            if line.startswith("info "):
                parts = line.split()
                if "nodes" in parts:
                    try:
                        nodes_seen = int(parts[parts.index("nodes") + 1])
                    except (IndexError, ValueError):
                        pass
            if line.startswith("bestmove "):
                parts = line.split()
                return (parts[1] if len(parts) > 1 else ""), nodes_seen


def load_targets(path: Path, limit: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def move_info(target: dict[str, object], move: str) -> dict[str, object]:
    for item in target.get("moves", []):
        if str(item.get("uci", "")) == move:
            return item
    return {"rank": 999, "score_cp": 0, "gap_cp": 32000, "policy": 0.0}


def run_engine(
    engine: Path, net: Path | None, targets: list[dict[str, object]],
    *, nodes: int, threads: int, hash_mb: int,
    reset_game: bool, require_native_net_load: bool,
) -> list[dict[str, object]]:
    uci = UciEngine(
        engine, nnue_file=net, threads=threads, hash_mb=hash_mb,
        require_native_net_load=require_native_net_load)
    rows: list[dict[str, object]] = []
    try:
        for target in targets:
            move, nodes_seen = uci.bestmove(
                str(target["fen"]), nodes, reset_game=reset_game)
            info = move_info(target, move)
            rows.append({
                "id": target["id"],
                "source": target["source"],
                "log": target.get("log", ""),
                "ply": target.get("ply", ""),
                "fullmove": target.get("fullmove", ""),
                "side": target.get("side", ""),
                "tags": ";".join(str(tag) for tag in target.get("tags", [])),
                "fen": target["fen"],
                "best_move": target.get("best_move", ""),
                "engine_move": move,
                "rank": int(info["rank"]),
                "gap_cp": int(info["gap_cp"]),
                "score_cp": int(info["score_cp"]),
                "policy": float(info.get("policy", 0.0)),
                "nodes": nodes_seen,
            })
    finally:
        uci.close()
    return rows


def split_rows(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    splits: dict[str, list[dict[str, object]]] = {"all": rows}
    for row in rows:
        for tag in str(row.get("tags", "")).split(";"):
            if tag:
                splits.setdefault(tag, []).append(row)
    return splits


def pct(count: int, total: int) -> float:
    return 100.0 * count / max(1, total)


def median_nonzero(values: list[int]) -> float:
    nonzero = [value for value in values if value != 0]
    return float(statistics.median(nonzero)) if nonzero else 0.0


def summarize_rows(name: str, rows: list[dict[str, object]]) -> list[str]:
    total = len(rows)
    gaps = [int(row["gap_cp"]) for row in rows]
    top1 = sum(1 for row in rows if int(row["rank"]) == 1)
    top3 = sum(1 for row in rows if int(row["rank"]) <= 3)
    return [
        f"[{name}]",
        f"targets={total}",
        f"top1={top1}/{total} ({pct(top1, total):.1f}%)",
        f"top3={top3}/{total} ({pct(top3, total):.1f}%)",
        f"sum_gap_cp={sum(gaps)}",
        f"median_gap_cp={statistics.median(gaps) if gaps else 0}",
        f"worst_gap_cp={max(gaps) if gaps else 0}",
    ]


def compare_rows(candidate: list[dict[str, object]], reference: list[dict[str, object]],
                 *, cap: int) -> list[dict[str, object]]:
    by_id = {str(row["id"]): row for row in reference}
    out: list[dict[str, object]] = []
    for cand in candidate:
        ref = by_id.get(str(cand["id"]))
        if ref is None:
            continue
        diff = int(ref["gap_cp"]) - int(cand["gap_cp"])
        row = dict(cand)
        row.update({
            "candidate_move": cand["engine_move"],
            "candidate_rank": cand["rank"],
            "candidate_gap_cp": cand["gap_cp"],
            "reference_move": ref["engine_move"],
            "reference_rank": ref["rank"],
            "reference_gap_cp": ref["gap_cp"],
            "diff_cp": diff,
            "capped_diff_cp": max(-cap, min(cap, diff)),
        })
        out.append(row)
    return out


def summarize_compare(name: str, rows: list[dict[str, object]]) -> list[str]:
    diffs = [int(row["diff_cp"]) for row in rows]
    capped = [int(row["capped_diff_cp"]) for row in rows]
    return [
        f"[{name}]",
        f"targets={len(rows)}",
        f"candidate_better={sum(1 for v in diffs if v > 0)}",
        f"reference_better={sum(1 for v in diffs if v < 0)}",
        f"sum_diff_cp={sum(diffs)}",
        f"capped_sum_diff_cp={sum(capped)}",
        f"median_nonzero_diff_cp={median_nonzero(diffs):.1f}",
        f"worst_regression_cp={min(diffs) if diffs else 0}",
        f"best_gain_cp={max(diffs) if diffs else 0}",
    ]


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run search-aware move-choice gate.")
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--candidate-net", type=Path)
    parser.add_argument("--reference-engine", type=Path)
    parser.add_argument("--reference-net", type=Path)
    parser.add_argument(
        "--reference-csv",
        type=Path,
        help="Reuse a previously written reference.csv instead of rerunning reference search.",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--nodes", type=int, default=100000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cap", type=int, default=200)
    parser.add_argument(
        "--reuse-hash",
        action="store_true",
        help="Do not send ucinewgame/isready before each target.",
    )
    parser.add_argument(
        "--require-native-net-load",
        action="store_true",
        help="Fail if a supplied net is not loaded through the native NNUE loader.",
    )
    args = parser.parse_args()

    if args.reference_csv and (args.reference_net or args.reference_engine):
        raise SystemExit(
            "--reference-csv cannot be combined with --reference-net or "
            "--reference-engine")

    targets = load_targets(args.targets.expanduser(), args.limit)
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate = run_engine(args.engine.expanduser(),
                           args.candidate_net.expanduser() if args.candidate_net else None,
                           targets, nodes=args.nodes, threads=args.threads,
                           hash_mb=args.hash, reset_game=not args.reuse_hash,
                           require_native_net_load=args.require_native_net_load)
    write_csv(out_dir / "candidate.csv", candidate)
    lines: list[str] = []
    for name, rows in sorted(split_rows(candidate).items()):
        if name == "all" or name in {"mate_like", "non_mate"}:
            lines.extend(summarize_rows(name, rows))
            lines.append("")
    reference: list[dict[str, object]] | None = None
    if args.reference_csv:
        reference = read_csv(args.reference_csv.expanduser())
    elif args.reference_net or args.reference_engine:
        ref_engine = args.reference_engine.expanduser() if args.reference_engine else args.engine.expanduser()
        reference = run_engine(ref_engine,
                               args.reference_net.expanduser() if args.reference_net else None,
                               targets, nodes=args.nodes, threads=args.threads,
                               hash_mb=args.hash, reset_game=not args.reuse_hash,
                               require_native_net_load=args.require_native_net_load)

    if reference is not None:
        write_csv(out_dir / "reference.csv", reference)
        comparisons = compare_rows(candidate, reference, cap=args.cap)
        write_csv(out_dir / "compare.csv", comparisons)
        for name, rows in sorted(split_rows(comparisons).items()):
            if name == "all" or name in {"mate_like", "non_mate"}:
                lines.extend(summarize_compare(f"compare:{name}", rows))
                lines.append("")
    text = "\n".join(lines).rstrip() + "\n"
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
