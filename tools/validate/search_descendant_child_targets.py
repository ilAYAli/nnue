#!/usr/bin/env python3
"""Build child-ranking targets from search PV descendant positions."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import queue
import re
import shlex
import subprocess
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import chess


SCORE_RE = re.compile(r"\bscore\s+(cp|mate)\s+(-?\d+)\b")


def mate_to_cp(mate: int) -> int:
    sign = 1 if mate > 0 else -1
    return sign * (32000 - min(abs(mate), 1000))


def phase_tag(board: chess.Board) -> str:
    nonking = max(0, len(board.piece_map()) - 2)
    if nonking <= 8:
        return "phase:endgame"
    if nonking <= 20:
        return "phase:late"
    if nonking <= 28:
        return "phase:midgame"
    return "phase:opening"


def expand(value: str | Path) -> str:
    return str(Path(os.path.expandvars(str(value))).expanduser())


def resolve_net_path(net: str | Path) -> str:
    path = Path(os.path.expandvars(str(net))).expanduser()
    if not path.is_file():
        raise RuntimeError(f"NNUE file not found: {path}")
    return str(path.resolve())


@dataclass(frozen=True)
class SearchResult:
    bestmove: str | None
    pv: tuple[str, ...]


class UciEngine:
    def __init__(
        self,
        command: str,
        *,
        threads: int,
        hash_mb: int,
        net: str = "",
    ) -> None:
        self.proc = subprocess.Popen(
            shlex.split(command),
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
        self.setoption("Threads", str(threads))
        self.setoption("Hash", str(hash_mb))
        if net:
            self.load_nnue(resolve_net_path(net))
        else:
            self.send("isready")
            self.wait_for("readyok", 20.0)

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.send("quit")
            except Exception:
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

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
            if token in self.read_line(timeout):
                return

    def setoption(self, name: str, value: str) -> None:
        self.send(f"setoption name {name} value {value}")

    def load_nnue(self, net: str) -> None:
        self.send("debug on")
        self.setoption("nnue_file", net)
        self.send("isready")
        loaded = False
        warnings: list[str] = []
        while True:
            line = self.read_line(20.0)
            if "WARNING: nnue_file" in line:
                warnings.append(line)
            if (
                ("network loaded from" in line or "embedded evaluator loaded from" in line)
                and net in line
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

    def search(self, fen: str, *, nodes: int, depth: int) -> SearchResult:
        self.newgame()
        self.send(f"position fen {fen}")
        if nodes > 0:
            self.send(f"go nodes {nodes}")
        else:
            self.send(f"go depth {depth}")
        pv: tuple[str, ...] = ()
        while True:
            line = self.read_line(120.0)
            if " pv " in line:
                pv = tuple(line.split(" pv ", 1)[1].split())
            if line.startswith("bestmove "):
                move = line.split()[1]
                return SearchResult(None if move == "(none)" else move, pv)

    def score_stm_cp(self, fen: str, *, nodes: int, depth: int) -> int | None:
        self.newgame()
        self.send(f"position fen {fen}")
        if nodes > 0:
            self.send(f"go nodes {nodes}")
        else:
            self.send(f"go depth {depth}")
        score: int | None = None
        while True:
            line = self.read_line(120.0)
            match = SCORE_RE.search(line)
            if match:
                kind, value = match.groups()
                raw = int(value)
                score = raw if kind == "cp" else mate_to_cp(raw)
            if line.startswith("bestmove "):
                return score


def child_fen(board: chess.Board, move_uci: str) -> str:
    child = board.copy(stack=False)
    child.push(chess.Move.from_uci(move_uci))
    return child.fen()


def score_parent_cp(
    oracle: UciEngine,
    board: chess.Board,
    move_uci: str,
    args: argparse.Namespace,
) -> int | None:
    stm_score = oracle.score_stm_cp(
        child_fen(board, move_uci),
        nodes=args.oracle_nodes,
        depth=args.oracle_depth,
    )
    return None if stm_score is None else -stm_score


def legal_uci(board: chess.Board, move_uci: str | None) -> str | None:
    if not move_uci:
        return None
    try:
        move = chess.Move.from_uci(move_uci)
    except ValueError:
        return None
    return move_uci if move in board.legal_moves else None


def board_after_prefix(fen: str, pv: tuple[str, ...], ply: int) -> chess.Board | None:
    board = chess.Board(fen)
    for move_uci in pv[:ply]:
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            return None
        if move not in board.legal_moves:
            return None
        board.push(move)
    return board


def best_scored_move(scored: list[dict]) -> str:
    return str(max(scored, key=lambda item: int(item["score_cp"]))["move"])


def target_gap(best_score: int, score: int, max_gap_cp: float) -> float:
    return min(max(0, best_score - score), max_gap_cp)


def build_descendant_row(
    root: dict,
    path_label: str,
    path_pv: tuple[str, ...],
    ply: int,
    candidate: UciEngine,
    reference: UciEngine,
    oracle: UciEngine,
    args: argparse.Namespace,
) -> tuple[dict | None, Counter[str]]:
    counters: Counter[str] = Counter()
    board = board_after_prefix(str(root["fen"]), path_pv, ply)
    if board is None:
        counters[f"invalid_path:{path_label}"] += 1
        return None, counters

    fen = board.fen()
    candidate_search = candidate.search(
        fen, nodes=args.descendant_search_nodes, depth=args.descendant_search_depth)
    reference_search = reference.search(
        fen, nodes=args.descendant_search_nodes, depth=args.descendant_search_depth)
    path_move = path_pv[ply] if ply < len(path_pv) else ""

    role_map: dict[str, set[str]] = {}
    for role, move_uci in (
        ("candidate", candidate_search.bestmove),
        ("reference", reference_search.bestmove),
        ("path", path_move),
    ):
        legal = legal_uci(board, move_uci)
        if legal:
            role_map.setdefault(legal, set()).add(role)
        elif move_uci:
            counters[f"illegal_{role}"] += 1

    if len(role_map) < 2:
        counters["skipped_same_moves"] += 1
        return None, counters

    scored = []
    for move_uci, origins in sorted(role_map.items()):
        score = score_parent_cp(oracle, board, move_uci, args)
        if score is None:
            counters["skipped_no_oracle_score"] += 1
            return None, counters
        scored.append({
            "move": move_uci,
            "score_cp": int(score),
            "role": "neighbor",
            "origins": sorted(origins),
        })

    best_move = best_scored_move(scored)
    best_score = max(int(move["score_cp"]) for move in scored)
    second_score = max(
        (int(move["score_cp"]) for move in scored if move["move"] != best_move),
        default=best_score,
    )
    gap = best_score - second_score
    if gap < args.min_oracle_gap_cp:
        counters["skipped_low_oracle_gap"] += 1
        return None, counters

    for move in scored:
        if move["move"] == best_move:
            move["role"] = "best"

    if candidate_search.bestmove == best_move:
        counters["candidate_best"] += 1
    if reference_search.bestmove == best_move:
        counters["reference_best"] += 1
    if path_move == best_move:
        counters[f"path_best:{path_label}"] += 1

    tags = list(dict.fromkeys([
        "source:search_descendant",
        f"path:{path_label}",
        f"descendant_ply:{ply}",
        phase_tag(board),
        "label:oracle_child_cp",
    ]))
    for tag in tags:
        counters[f"tag:{tag}"] += 1

    root_id = str(root.get("id") or root.get("group_id"))
    return {
        "id": f"{root_id}:{path_label}:pv{ply}",
        "fen": fen,
        "best_move": best_move,
        "max_gap_cp": float(args.max_gap_cp),
        "moves": scored,
        "source": "search_descendant",
        "root_id": root_id,
        "root_fen": str(root["fen"]),
        "root_best_move": str(root.get("best_move", "")),
        "root_candidate_move": str(root.get("candidate_move", "")),
        "root_reference_move": str(root.get("reference_move", "")),
        "path": path_label,
        "path_prefix": list(path_pv[:ply]),
        "path_move": path_move,
        "candidate_move": candidate_search.bestmove or "",
        "reference_move": reference_search.bestmove or "",
        "candidate_pv": list(candidate_search.pv),
        "reference_pv": list(reference_search.pv),
        "score_pov": "parent",
        "oracle_top2_gap_cp": gap,
        "oracle": {
            "engine": args.oracle_engine,
            "nodes": args.oracle_nodes,
            "depth": args.oracle_depth,
            "threads": args.oracle_threads,
            "hash": args.oracle_hash,
        },
        "search": {
            "engine": args.engine,
            "nodes": args.descendant_search_nodes,
            "depth": args.descendant_search_depth,
            "threads": args.threads,
            "hash": args.hash,
        },
        "tags": tags,
    }, counters


def process_root(
    root: dict,
    candidate: UciEngine,
    reference: UciEngine,
    oracle: UciEngine,
    args: argparse.Namespace,
) -> tuple[list[dict], Counter[str]]:
    counters: Counter[str] = Counter(input_groups=1)
    root_fen = str(root["fen"])
    root_best = str(root.get("best_move", ""))
    candidate_root = candidate.search(
        root_fen, nodes=args.root_search_nodes, depth=args.root_search_depth)
    reference_root = reference.search(
        root_fen, nodes=args.root_search_nodes, depth=args.root_search_depth)
    counters["root_candidate_best"] += int(candidate_root.bestmove == root_best)
    counters["root_reference_best"] += int(reference_root.bestmove == root_best)
    counters["root_candidate_reference_differ"] += int(
        candidate_root.bestmove != reference_root.bestmove)

    paths = {
        "candidate": candidate_root.pv,
        "reference": reference_root.pv,
    }
    rows: list[dict] = []
    seen_fens: set[tuple[str, str]] = set()
    for path_label, pv in paths.items():
        if not pv:
            counters[f"empty_pv:{path_label}"] += 1
            continue
        for ply in range(args.min_descendant_ply, args.max_descendant_ply + 1):
            if ply >= len(pv):
                counters[f"path_short:{path_label}"] += 1
                break
            board = board_after_prefix(root_fen, pv, ply)
            if board is None:
                counters[f"invalid_path:{path_label}"] += 1
                break
            key = (path_label, board.fen())
            if key in seen_fens:
                counters["skipped_duplicate_path_fen"] += 1
                continue
            seen_fens.add(key)
            row, row_counts = build_descendant_row(
                root, path_label, pv, ply, candidate, reference, oracle, args)
            counters.update(row_counts)
            if row is not None:
                rows.append(row)
                counters["written_groups"] += 1
    return rows, counters


def chunks(items: list[dict], jobs: int) -> list[list[dict]]:
    jobs = max(1, min(jobs, len(items)))
    size = (len(items) + jobs - 1) // jobs
    return [items[i:i + size] for i in range(0, len(items), size)]


def process_chunk(chunk: list[dict], args: argparse.Namespace) -> tuple[list[dict], Counter[str]]:
    candidate = UciEngine(
        args.engine, threads=args.threads, hash_mb=args.hash,
        net=args.candidate_net)
    reference = UciEngine(
        args.engine, threads=args.threads, hash_mb=args.hash,
        net=args.reference_net)
    oracle = UciEngine(
        args.oracle_engine, threads=args.oracle_threads, hash_mb=args.oracle_hash)
    rows: list[dict] = []
    counters: Counter[str] = Counter()
    try:
        for root in chunk:
            root_rows, root_counts = process_root(root, candidate, reference, oracle, args)
            rows.extend(root_rows)
            counters.update(root_counts)
    finally:
        candidate.close()
        reference.close()
        oracle.close()
    return rows, counters


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build oracle-scored child targets from search PV descendants.")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--candidate-net", required=True)
    ap.add_argument("--reference-net", required=True)
    ap.add_argument("--oracle-engine", required=True)
    ap.add_argument("--root-search-nodes", type=int, default=20000)
    ap.add_argument("--root-search-depth", type=int, default=0)
    ap.add_argument("--descendant-search-nodes", type=int, default=10000)
    ap.add_argument("--descendant-search-depth", type=int, default=0)
    ap.add_argument("--oracle-nodes", type=int, default=200000)
    ap.add_argument("--oracle-depth", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--oracle-threads", type=int, default=1)
    ap.add_argument("--oracle-hash", type=int, default=128)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--min-descendant-ply", type=int, default=1)
    ap.add_argument("--max-descendant-ply", type=int, default=4)
    ap.add_argument("--max-gap-cp", type=float, default=800.0)
    ap.add_argument("--min-oracle-gap-cp", type=float, default=20.0)
    ap.add_argument("--min-input-groups", type=int, default=1)
    ap.add_argument("--min-output-groups", type=int, default=1)
    args = ap.parse_args()

    if args.root_search_nodes <= 0 and args.root_search_depth <= 0:
        raise SystemExit("either --root-search-nodes or --root-search-depth must be positive")
    if args.descendant_search_nodes <= 0 and args.descendant_search_depth <= 0:
        raise SystemExit(
            "either --descendant-search-nodes or --descendant-search-depth must be positive")
    if args.oracle_nodes <= 0 and args.oracle_depth <= 0:
        raise SystemExit("either --oracle-nodes or --oracle-depth must be positive")
    if args.min_descendant_ply < 1:
        raise SystemExit("--min-descendant-ply must be at least 1")
    if args.max_descendant_ply < args.min_descendant_ply:
        raise SystemExit("--max-descendant-ply must be >= --min-descendant-ply")

    start = time.monotonic()
    roots: list[dict] = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                roots.append(json.loads(line))
    if len(roots) < args.min_input_groups:
        raise SystemExit(
            f"input_groups={len(roots)} below min_input_groups={args.min_input_groups}")

    rows: list[dict] = []
    counters: Counter[str] = Counter()
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, args.jobs)) as executor:
        futures = [
            executor.submit(process_chunk, chunk, args)
            for chunk in chunks(roots, max(1, args.jobs))
        ]
        for future in concurrent.futures.as_completed(futures):
            chunk_rows, chunk_counts = future.result()
            rows.extend(chunk_rows)
            counters.update(chunk_counts)

    rows.sort(key=lambda row: str(row["id"]))
    if len(rows) < args.min_output_groups:
        raise SystemExit(
            f"written_groups={len(rows)} below min_output_groups={args.min_output_groups}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    written_tags: Counter[str] = Counter()
    search_metrics: Counter[str] = Counter()
    for row in rows:
        written_tags.update(str(tag) for tag in row.get("tags", []))
        scores = {
            str(move["move"]): int(move["score_cp"])
            for move in row.get("moves", [])
        }
        best_move = str(row.get("best_move", ""))
        candidate_move = str(row.get("candidate_move", ""))
        reference_move = str(row.get("reference_move", ""))
        if (
            best_move in scores
            and candidate_move in scores
            and reference_move in scores
        ):
            best_score = scores[best_move]
            candidate_loss = best_score - scores[candidate_move]
            reference_loss = best_score - scores[reference_move]
            diff = reference_loss - candidate_loss
            search_metrics["search_compared"] += 1
            search_metrics["candidate_search_loss_cp"] += candidate_loss
            search_metrics["reference_search_loss_cp"] += reference_loss
            search_metrics["search_sum_diff_cp"] += diff
            if diff > 0:
                search_metrics["candidate_search_better"] += 1
            elif diff < 0:
                search_metrics["reference_search_better"] += 1
            else:
                search_metrics["search_equal"] += 1

    elapsed = time.monotonic() - start
    lines = [
        f"input_groups={counters['input_groups']}",
        f"written_groups={len(rows)}",
        f"root_candidate_best={counters['root_candidate_best']}",
        f"root_reference_best={counters['root_reference_best']}",
        f"root_candidate_reference_differ={counters['root_candidate_reference_differ']}",
        f"candidate_best={counters['candidate_best']}",
        f"reference_best={counters['reference_best']}",
        f"path_best_candidate={counters['path_best:candidate']}",
        f"path_best_reference={counters['path_best:reference']}",
        f"search_compared={search_metrics['search_compared']}",
        f"candidate_search_better={search_metrics['candidate_search_better']}",
        f"reference_search_better={search_metrics['reference_search_better']}",
        f"search_equal={search_metrics['search_equal']}",
        f"search_sum_diff_cp={search_metrics['search_sum_diff_cp']}",
        f"candidate_search_loss_cp={search_metrics['candidate_search_loss_cp']}",
        f"reference_search_loss_cp={search_metrics['reference_search_loss_cp']}",
        f"skipped_same_moves={counters['skipped_same_moves']}",
        f"skipped_low_oracle_gap={counters['skipped_low_oracle_gap']}",
        f"skipped_no_oracle_score={counters['skipped_no_oracle_score']}",
        f"root_search_nodes={args.root_search_nodes}",
        f"descendant_search_nodes={args.descendant_search_nodes}",
        f"oracle_nodes={args.oracle_nodes}",
        f"min_descendant_ply={args.min_descendant_ply}",
        f"max_descendant_ply={args.max_descendant_ply}",
        f"min_oracle_gap_cp={args.min_oracle_gap_cp}",
        f"elapsed_s={elapsed:.3f}",
        "tags=" + ",".join(
            f"{tag}:{count}" for tag, count in sorted(written_tags.items())
        ),
        f"output={args.out}",
    ]
    summary = "\n".join(lines) + "\n"
    print(summary, end="")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(summary, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
