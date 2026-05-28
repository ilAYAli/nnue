from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import chess


EVAL2_RE = re.compile(r"^eval2\s+(-?\d+)\s+cp\s+\(stm=(white|black)\)")
EVALNET_RE = re.compile(r"^evalnet\s+(-?\d+)\s+cp\s+\(stm=(white|black)\)")
EVAL_RE = re.compile(r"^eval\s+(-?\d+)\s*$")


class EngineError(RuntimeError):
    pass


def resolve_net_path(net: Path) -> Path:
    path = Path(os.path.expandvars(str(net))).expanduser()
    if not path.is_file():
        raise EngineError(f"NNUE file not found: {path}")
    return path.resolve()


class EnyoEval2:
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
            raise EngineError("failed to open engine pipes")
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
            raise EngineError("engine read timed out") from exc
        if line is None:
            raise EngineError("engine exited")
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
            raise EngineError("; ".join(warnings))
        if not loaded:
            raise EngineError(f"engine did not confirm NNUE load: {net}")

    def eval_stm_cp(self, fen: str) -> int:
        self.send(f"position fen {fen}")
        self.send("eval2")
        while True:
            line = self.read_line(10.0)
            match = EVAL2_RE.match(line)
            if match:
                return int(match.group(1))
            if line == "unknown command: 'eval2'":
                self.send("evalnet")
                continue
            match = EVALNET_RE.match(line)
            if match:
                return int(match.group(1))
            match = EVAL_RE.match(line)
            if match:
                return int(match.group(1))
            if line.startswith("eval2 error:"):
                raise EngineError(line)


@dataclass
class CaseResult:
    row: dict[str, object]
    baseline_margin: int
    candidate_margin: int

    @property
    def delta_margin(self) -> int:
        return self.candidate_margin - self.baseline_margin

    @property
    def loss_cp(self) -> int:
        return int(self.row["loss_cp"])


def child_fen(fen: str, move_uci: str) -> str:
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        raise ValueError(f"illegal move {move_uci} in {fen}")
    board.push(move)
    return board.fen()


def original_pov_child_eval(engine: EnyoEval2, child: str) -> int:
    # After a legal move, child side-to-move is the opponent, while eval2 is
    # side-to-move POV. Negate it to recover the mover's POV at the root.
    return -engine.eval_stm_cp(child)


def load_cases(path: Path, limit: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def margins(engine: EnyoEval2, rows: list[dict[str, object]]) -> list[int]:
    out: list[int] = []
    for row in rows:
        fen = str(row["fen"])
        best_child = child_fen(fen, str(row["best"]))
        played_child = child_fen(fen, str(row["played"]))
        best_eval = original_pov_child_eval(engine, best_child)
        played_eval = original_pov_child_eval(engine, played_child)
        out.append(best_eval - played_eval)
    return out


def pct(count: int, total: int) -> float:
    return 100.0 * count / max(1, total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--engine", required=True, type=Path)
    ap.add_argument("--baseline-net", required=True, type=Path)
    ap.add_argument("--candidate-net", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    rows = load_cases(args.cases, args.limit)
    if not rows:
        raise SystemExit("no cases")

    baseline = EnyoEval2(args.engine, args.baseline_net, args.threads, args.hash)
    try:
        baseline_margins = margins(baseline, rows)
    finally:
        baseline.close()

    candidate = EnyoEval2(args.engine, args.candidate_net, args.threads, args.hash)
    try:
        candidate_margins = margins(candidate, rows)
    finally:
        candidate.close()

    results = [
        CaseResult(row, base, cand)
        for row, base, cand in zip(rows, baseline_margins, candidate_margins)
    ]
    n = len(results)
    base_correct = sum(r.baseline_margin > 0 for r in results)
    cand_correct = sum(r.candidate_margin > 0 for r in results)
    fixed = sum(r.baseline_margin <= 0 < r.candidate_margin for r in results)
    regressed = sum(r.baseline_margin > 0 >= r.candidate_margin for r in results)
    better = sum(r.delta_margin > 0 for r in results)
    weighted_delta = (
        sum(r.delta_margin * r.loss_cp for r in results)
        / max(1, sum(r.loss_cp for r in results))
    )
    avg_base = sum(r.baseline_margin for r in results) / n
    avg_cand = sum(r.candidate_margin for r in results) / n
    avg_delta = sum(r.delta_margin for r in results) / n

    print(f"cases={n}")
    print(f"baseline_prefers_best={base_correct}/{n} ({pct(base_correct, n):.1f}%)")
    print(f"candidate_prefers_best={cand_correct}/{n} ({pct(cand_correct, n):.1f}%)")
    print(f"fixed={fixed}")
    print(f"regressed={regressed}")
    print(f"candidate_better_margin={better}/{n} ({pct(better, n):.1f}%)")
    print(f"baseline_avg_margin={avg_base:.1f}cp")
    print(f"candidate_avg_margin={avg_cand:.1f}cp")
    print(f"delta_avg_margin={avg_delta:.1f}cp")
    print(f"delta_loss_weighted_margin={weighted_delta:.1f}cp")
    print()
    print("worst candidate deltas:")
    for r in sorted(results, key=lambda item: item.delta_margin)[:10]:
        print(
            f"{r.delta_margin:6d}cp base={r.baseline_margin:6d} "
            f"cand={r.candidate_margin:6d} loss={r.loss_cp:4d} "
            f"{r.row['severity']} {r.row['played']}->{r.row['best']} "
            f"{Path(str(r.row['source'])).name}:{r.row['line']}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            for r in results:
                row = dict(r.row)
                row.update({
                    "baseline_margin": r.baseline_margin,
                    "candidate_margin": r.candidate_margin,
                    "delta_margin": r.delta_margin,
                })
                f.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
