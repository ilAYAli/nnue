#!/usr/bin/env python3
"""Evaluate scored JSONL rows through Enyo's exported-net runtime path."""
from __future__ import annotations

import argparse
import json
import math
import queue
import re
import subprocess
import sys
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


EVALNET_RE = re.compile(r"^evalnet\s+(-?\d+)\s+cp\s+\(stm=(white|black)\)")


class EngineError(RuntimeError):
    pass


class EnyoEvalNet:
    def __init__(self, engine: Path, net: Path, *, threads: int,
                 hash_mb: int, timeout: float) -> None:
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
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.reader = threading.Thread(target=self._read_stdout, daemon=True)
        self.reader.start()

        self.send("uci")
        self.wait_for("uciok", timeout)
        self.setoption("Threads", str(threads))
        self.setoption("Hash", str(hash_mb))
        self.setoption("nnue_file", str(net))
        self.send("isready")
        self.wait_for("readyok", timeout)

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.send("quit")
            except Exception:
                pass
            try:
                self.proc.wait(timeout=2.0)
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

    def setoption(self, name: str, value: str) -> None:
        self.send(f"setoption name {name} value {value}")

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
class Stats:
    rows: int = 0
    mae: float = 0.0
    mse: float = 0.0
    sign_ok: int = 0
    sign_total: int = 0
    bias: float = 0.0
    pred_sum: float = 0.0
    target_sum: float = 0.0
    pred_sq_sum: float = 0.0
    target_sq_sum: float = 0.0
    pred_target_sum: float = 0.0

    def update(self, pred: float, target: float) -> None:
        err = pred - target
        self.rows += 1
        self.mae += abs(err)
        self.mse += err * err
        self.bias += err
        self.pred_sum += pred
        self.target_sum += target
        self.pred_sq_sum += pred * pred
        self.target_sq_sum += target * target
        self.pred_target_sum += pred * target
        if target != 0:
            self.sign_total += 1
            self.sign_ok += int((pred > 0) == (target > 0))

    def derived(self) -> dict[str, float]:
        n = max(1, self.rows)
        pred_mean = self.pred_sum / n
        target_mean = self.target_sum / n
        cov = self.pred_target_sum / n - pred_mean * target_mean
        pred_var = self.pred_sq_sum / n - pred_mean * pred_mean
        target_var = self.target_sq_sum / n - target_mean * target_mean
        corr = 0.0
        if pred_var > 0 and target_var > 0:
            corr = cov / math.sqrt(pred_var * target_var)
        slope = cov / target_var if target_var > 0 else 0.0
        return {
            "rows": float(self.rows),
            "mae": self.mae / n,
            "mse": self.mse / n,
            "rmse": math.sqrt(self.mse / n),
            "sign": 100.0 * self.sign_ok / max(1, self.sign_total),
            "wrong_sign": float(self.sign_total - self.sign_ok),
            "sign_total": float(self.sign_total),
            "bias": self.bias / n,
            "pred_mean": pred_mean,
            "target_mean": target_mean,
            "corr": corr,
            "slope": slope,
        }


@dataclass
class Evaluation:
    total: Stats = field(default_factory=Stats)
    side: dict[str, Stats] = field(default_factory=lambda: defaultdict(Stats))
    source: dict[str, Stats] = field(default_factory=lambda: defaultdict(Stats))
    bucket: dict[str, Stats] = field(default_factory=lambda: defaultdict(Stats))
    material_bucket: dict[int, Stats] = field(default_factory=lambda: defaultdict(Stats))
    skipped: int = 0


def score_bucket(target: float) -> str:
    value = abs(target)
    if value < 50:
        return "0-50"
    if value < 100:
        return "50-100"
    if value < 300:
        return "100-300"
    if value < 800:
        return "300-800"
    if value < 1600:
        return "800-1600"
    return "1600+"


def source_name(row: dict[str, object]) -> str:
    return str(row.get("source_type") or row.get("teacher") or "unknown")


RECKLESS_OUTPUT_BUCKETS = (
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4,
    5, 5, 5, 6, 6, 6, 7, 7, 7, 7,
)


def fen_piece_count(fen: str) -> int:
    return sum(char.isalpha() for char in fen.split()[0])


def read_rows(path: Path, *, limit: int, skip: int,
              score_field: str) -> list[tuple[int, str, float, str]]:
    rows: list[tuple[int, str, float, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line_no <= skip:
                continue
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "fen" not in row or score_field not in row:
                continue
            fen = str(row["fen"])
            target = float(row[score_field])
            rows.append((line_no, fen, target, source_name(row)))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def print_stats(label: str, stats: Stats) -> None:
    d = stats.derived()
    print(
        f"{label} rows={int(d['rows'])} mae={d['mae']:.3f} "
        f"mse={d['mse']:.3f} rmse={d['rmse']:.3f} "
        f"sign={d['sign']:.2f}% wrong_sign={int(d['wrong_sign'])}/"
        f"{int(d['sign_total'])} bias={d['bias']:.3f} "
        f"pred_mean={d['pred_mean']:.3f} target_mean={d['target_mean']:.3f} "
        f"corr={d['corr']:.6f} slope={d['slope']:.6f}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate scored JSONL rows with Enyo evalnet.")
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--net", required=True, type=Path)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--score-field", default="score")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--buckets", action="store_true")
    parser.add_argument("--sources", action="store_true")
    parser.add_argument("--material-buckets", choices=("reckless",))
    parser.add_argument("--progress", type=int, default=1000)
    args = parser.parse_args()

    rows = read_rows(
        args.jsonl.expanduser(),
        limit=args.rows,
        skip=args.skip,
        score_field=args.score_field,
    )
    if not rows:
        raise SystemExit("no scored JSONL rows found")

    result = Evaluation()
    engine = EnyoEvalNet(
        args.engine.expanduser(),
        args.net.expanduser(),
        threads=args.threads,
        hash_mb=args.hash,
        timeout=args.timeout,
    )
    try:
        for i, (line_no, fen, target, source) in enumerate(rows, start=1):
            try:
                pred = engine.eval_stm_cp(fen)
            except EngineError as exc:
                print(f"row {line_no}: {exc}", file=sys.stderr, flush=True)
                result.skipped += 1
                continue
            side = fen.split()[1]
            result.total.update(pred, target)
            result.side[side].update(pred, target)
            result.source[source].update(pred, target)
            result.bucket[score_bucket(target)].update(pred, target)
            if args.material_buckets == "reckless":
                # Some external validation rows contain illegal promoted-piece
                # counts above 32. Reckless maps every such position to its
                # highest material bucket.
                piece_count = min(fen_piece_count(fen), 32)
                result.material_bucket[RECKLESS_OUTPUT_BUCKETS[piece_count]].update(
                    pred, target)
            if args.progress > 0 and i % args.progress == 0:
                print(f"evaluated {i}/{len(rows)}", flush=True)
    finally:
        engine.close()

    print_stats("all", result.total)
    overall = result.total.derived()
    for key in ("mae", "sign", "corr", "bias", "slope"):
        print(f"{key}={overall[key]:.6f}")
    print(f"skipped={result.skipped}", flush=True)
    for side in ("w", "b"):
        if result.side[side].rows:
            print_stats(f"side:{side}", result.side[side])
    if args.sources:
        for source, stats in sorted(result.source.items()):
            print_stats(f"source:{source}", stats)
    if args.buckets:
        for bucket in ("0-50", "50-100", "100-300", "300-800",
                       "800-1600", "1600+"):
            if result.bucket[bucket].rows:
                print_stats(f"bucket:{bucket}", result.bucket[bucket])
    if args.material_buckets:
        for bucket in range(8):
            stats = result.material_bucket[bucket]
            derived = stats.derived()
            print(f"material_bucket_{bucket}_rows={stats.rows}")
            print(f"material_bucket_{bucket}_corr={derived['corr']:.6f}")
            print(f"material_bucket_{bucket}_slope={derived['slope']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
