from __future__ import annotations

import argparse
import json
import queue
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess


EVALNET_RE = re.compile(r"^evalnet\s+(-?\d+)\s+cp\s+\(stm=(white|black)\)")


class EngineError(RuntimeError):
    pass


class EnyoEvalNet:
    def __init__(self, engine: Path, net: Path, threads: int, hash_mb: int,
                 timeout: float):
        self.timeout = timeout
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
        self.wait_for("uciok", self.timeout)
        self.send(f"setoption name Threads value {threads}")
        self.send(f"setoption name Hash value {hash_mb}")
        self.send(f"setoption name nnue_file value {net}")
        self.send("isready")
        self.wait_for("readyok", self.timeout)

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

    def eval_stm_cp(self, fen: str) -> int:
        self.send(f"position fen {fen}")
        self.send("evalnet")
        while True:
            line = self.read_line(self.timeout)
            match = EVALNET_RE.match(line)
            if match:
                return int(match.group(1))
            if line.startswith("evalnet error:"):
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


def original_pov_child_eval(engine: EnyoEvalNet, child: str) -> int:
    # After a legal move, child side-to-move is the opponent, while evalnet is
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


def validate_case(row: dict[str, object]) -> None:
    for key in ("fen", "best", "played"):
        if key not in row or not str(row[key]):
            raise ValueError(f"missing {key}: {row}")
    fen = str(row["fen"])
    for key in ("best", "played"):
        move = chess.Move.from_uci(str(row[key]))
        if move not in chess.Board(fen).legal_moves:
            raise ValueError(f"illegal {key}={move} in {fen}")


def margins(engine: EnyoEvalNet, rows: list[dict[str, object]]) -> list[int]:
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


def source_counts(results: list[CaseResult]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for result in results:
        source = str(result.row.get("source_label", result.row.get("source", "")))
        if source not in out:
            out[source] = {
                "cases": 0,
                "baseline_correct": 0,
                "candidate_correct": 0,
                "fixed": 0,
                "regressed": 0,
                "better_margin": 0,
            }
        item = out[source]
        item["cases"] += 1
        item["baseline_correct"] += int(result.baseline_margin > 0)
        item["candidate_correct"] += int(result.candidate_margin > 0)
        item["fixed"] += int(result.baseline_margin <= 0 < result.candidate_margin)
        item["regressed"] += int(result.baseline_margin > 0 >= result.candidate_margin)
        item["better_margin"] += int(result.delta_margin > 0)
    return out


def write_summary_json(path: Path, summary: dict[str, Any],
                       results: list[CaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(summary)
    payload["cases"] = [
        {
            **result.row,
            "baseline_margin": result.baseline_margin,
            "candidate_margin": result.candidate_margin,
            "delta_margin": result.delta_margin,
        }
        for result in results
    ]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--engine", required=True, type=Path)
    ap.add_argument("--baseline-net", required=True, type=Path)
    ap.add_argument("--candidate-net", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--summary-json", type=Path)
    ap.add_argument("--fail-if-candidate-below-baseline", action="store_true")
    ap.add_argument("--fail-if-regressed-above", type=int)
    ap.add_argument("--fail-if-fixed-below", type=int)
    ap.add_argument("--fail-if-delta-below", type=float)
    ap.add_argument("--fail-if-loss-weighted-delta-below", type=float)
    args = ap.parse_args()

    rows = load_cases(args.cases, args.limit)
    if not rows:
        raise SystemExit("no cases")
    for row in rows:
        validate_case(row)

    baseline = EnyoEvalNet(
        args.engine, args.baseline_net, args.threads, args.hash, args.timeout)
    try:
        baseline_margins = margins(baseline, rows)
    finally:
        baseline.close()

    candidate = EnyoEvalNet(
        args.engine, args.candidate_net, args.threads, args.hash, args.timeout)
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
    print("by_source:")
    for source, item in sorted(source_counts(results).items()):
        source_n = item["cases"]
        print(
            f"  {source} cases={source_n} "
            f"baseline={item['baseline_correct']}/{source_n} "
            f"candidate={item['candidate_correct']}/{source_n} "
            f"fixed={item['fixed']} regressed={item['regressed']} "
            f"better_margin={item['better_margin']}/{source_n}"
        )
    print()
    print("worst candidate deltas:")
    for r in sorted(results, key=lambda item: item.delta_margin)[:10]:
        severity = str(r.row.get("severity", r.row.get("source_label", "")))
        line = str(r.row.get("line", r.row.get("record_index", "")))
        print(
            f"{r.delta_margin:6d}cp base={r.baseline_margin:6d} "
            f"cand={r.candidate_margin:6d} loss={r.loss_cp:4d} "
            f"{severity} {r.row['played']}->{r.row['best']} "
            f"{Path(str(r.row.get('source', ''))).name}:{line}"
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

    summary = {
        "cases": n,
        "baseline_prefers_best": base_correct,
        "candidate_prefers_best": cand_correct,
        "fixed": fixed,
        "regressed": regressed,
        "candidate_better_margin": better,
        "baseline_avg_margin": avg_base,
        "candidate_avg_margin": avg_cand,
        "delta_avg_margin": avg_delta,
        "delta_loss_weighted_margin": weighted_delta,
        "by_source": source_counts(results),
    }
    if args.summary_json:
        write_summary_json(args.summary_json, summary, results)

    failures: list[str] = []
    if args.fail_if_candidate_below_baseline and cand_correct < base_correct:
        failures.append(
            f"candidate_prefers_best {cand_correct} < baseline {base_correct}")
    if args.fail_if_regressed_above is not None and regressed > args.fail_if_regressed_above:
        failures.append(
            f"regressed {regressed} > {args.fail_if_regressed_above}")
    if args.fail_if_fixed_below is not None and fixed < args.fail_if_fixed_below:
        failures.append(f"fixed {fixed} < {args.fail_if_fixed_below}")
    if args.fail_if_delta_below is not None and avg_delta < args.fail_if_delta_below:
        failures.append(
            f"delta_avg_margin {avg_delta:.1f} < {args.fail_if_delta_below}")
    if (
        args.fail_if_loss_weighted_delta_below is not None
        and weighted_delta < args.fail_if_loss_weighted_delta_below
    ):
        failures.append(
            "delta_loss_weighted_margin "
            f"{weighted_delta:.1f} < {args.fail_if_loss_weighted_delta_below}"
        )
    if failures:
        raise SystemExit("move gate failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
