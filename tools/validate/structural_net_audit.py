#!/usr/bin/env python3
"""Compare static net evals across UCI engine/net subjects.

This is intentionally engine-agnostic but optimized for Enyo's non-UCI
`eval` command.  It answers structural questions before launching another
training run: are two nets on comparable score scales, where do they disagree,
and is the disagreement concentrated in material/phase buckets?
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import re
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


EVAL_RE = re.compile(r"^eval\s+(-?\d+)\s*$")
EVALNET_RE = re.compile(r"^evalnet\s+(-?\d+)\s+cp\b")

DEFAULT_FENS = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 2 3",
    "r2q1rk1/ppp2ppp/2n2n2/2bp4/2B5/2NP1NP1/PPP2PBP/R2Q1RK1 w - - 2 9",
    "2r2rk1/1bqnbppp/p2ppn2/1p6/3NP3/1BN1BP2/PPP1Q1PP/2KR3R w - - 0 14",
    "8/2p5/3p4/1p1Pp3/1P2Pp2/2P2P2/8/6Kk w - - 0 45",
    "8/5pk1/6p1/6Pp/7P/5PK1/8/8 b - - 0 52",
)

PIECE_VALUES = {
    "p": 1,
    "n": 3,
    "b": 3,
    "r": 5,
    "q": 9,
}


@dataclass(frozen=True)
class Subject:
    name: str
    engine: str
    net: str | None
    command: str


class EngineTimeout(RuntimeError):
    pass


class UciEvalEngine:
    def __init__(
        self,
        subject: Subject,
        *,
        threads: int,
        hash_mb: int,
        timeout_s: float,
        uci_options: list[str],
    ) -> None:
        self.subject = subject
        self.timeout_s = timeout_s
        self.proc = subprocess.Popen(
            [subject.engine],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.reader = threading.Thread(target=self._read_stdout, daemon=True)
        self.reader.start()
        self.send("uci")
        self.wait_for("uciok")
        self.setoption("Threads", str(threads))
        self.setoption("Hash", str(hash_mb))
        self.setoption("use_syzygy", "false")
        for option in uci_options:
            name, value = option.split("=", 1)
            self.setoption(name, value)
        if subject.net:
            self.setoption("nnue_file", subject.net)
        self.send("isready")
        self.wait_for("readyok")

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.send("quit")
        except (BrokenPipeError, OSError):
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines.put(line.strip())
        self.lines.put(None)

    def send(self, command: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def readline(self, deadline: float) -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise EngineTimeout(
                f"{self.subject.name}: timed out after {self.timeout_s:.1f}s")
        try:
            line = self.lines.get(timeout=remaining)
        except queue.Empty:
            raise EngineTimeout(
                f"{self.subject.name}: timed out after {self.timeout_s:.1f}s")
        if line is None:
            raise RuntimeError(f"{self.subject.name}: engine exited")
        return line

    def wait_for(self, token: str) -> None:
        deadline = time.monotonic() + self.timeout_s
        while True:
            if self.readline(deadline) == token:
                return

    def setoption(self, name: str, value: str) -> None:
        self.send(f"setoption name {name} value {value}")

    def eval(self, fen: str) -> int:
        deadline = time.monotonic() + self.timeout_s
        self.send(f"position fen {fen}")
        self.send(self.subject.command)
        while True:
            line = self.readline(deadline)
            match = (
                EVALNET_RE.match(line)
                if self.subject.command == "evalnet"
                else EVAL_RE.match(line)
            )
            if match:
                return int(match.group(1))


def default_command(net: str | None) -> str:
    if net and net.endswith(".nn"):
        return "evalnet"
    return "eval"


def parse_subject(raw: str, default_engine: str) -> Subject:
    if "=" not in raw:
        raise ValueError(
            f"subject must be NAME=NET or NAME=ENGINE,NET[,eval|evalnet]: {raw}")
    name, spec = raw.split("=", 1)
    if not name:
        raise ValueError(f"empty subject name: {raw}")
    if "," in spec:
        parts = spec.split(",")
        if len(parts) not in (2, 3):
            raise ValueError(
                f"subject must be NAME=ENGINE,NET[,eval|evalnet]: {raw}")
        engine, net = parts[0], parts[1]
        command = parts[2] if len(parts) == 3 and parts[2] else ""
        net_value = net or None
        return Subject(
            name=name,
            engine=engine or default_engine,
            net=net_value,
            command=command or default_command(net_value),
        )
    return Subject(
        name=name,
        engine=default_engine,
        net=spec or None,
        command=default_command(spec or None),
    )


def fen_from_json_line(line: str) -> str | None:
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return None
    fen = row.get("fen")
    return fen if isinstance(fen, str) else None


def load_fens(path: Path | None, limit: int) -> list[str]:
    if path is None:
        return list(DEFAULT_FENS)[:limit]
    fens: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fen = fen_from_json_line(stripped) if stripped.startswith("{") else stripped
            if fen:
                fens.append(fen)
            if len(fens) >= limit:
                break
    return fens


def board_part(fen: str) -> str:
    return fen.split()[0]


def side_to_move(fen: str) -> str:
    parts = fen.split()
    return parts[1] if len(parts) > 1 else "?"


def piece_count(fen: str) -> int:
    return sum(1 for ch in board_part(fen) if ch.isalpha())


def non_pawn_material(fen: str) -> int:
    return sum(
        PIECE_VALUES.get(ch.lower(), 0)
        for ch in board_part(fen)
        if ch.isalpha() and ch.lower() != "p"
    )


def material_bucket(fen: str) -> str:
    count = piece_count(fen)
    if count >= 28:
        return "opening"
    if count >= 18:
        return "middlegame"
    if count >= 10:
        return "late"
    return "endgame"


def eval_bucket(value: int) -> str:
    abs_value = abs(value)
    if abs_value < 50:
        return "000-049"
    if abs_value < 100:
        return "050-099"
    if abs_value < 300:
        return "100-299"
    if abs_value < 800:
        return "300-799"
    return "800+"


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def rmse(values: list[float]) -> float:
    return math.sqrt(mean([v * v for v in values])) if values else 0.0


def corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    cov = mean([(x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)])
    vx = mean([(x - mx) ** 2 for x in xs])
    vy = mean([(y - my) ** 2 for y in ys])
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    return cov / math.sqrt(vx * vy)


def slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    var = mean([(x - mx) ** 2 for x in xs])
    if var <= 0.0:
        return 0.0
    cov = mean([(x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)])
    return cov / var


def print_subject_summary(name: str, values: list[int]) -> None:
    print(
        f"{name:18} rows={len(values):6d}"
        f" mean={mean(values):8.2f}"
        f" stdev={statistics.pstdev(values) if len(values) > 1 else 0.0:8.2f}"
        f" min={min(values):6d}"
        f" max={max(values):6d}"
        f" abs>=2000={sum(abs(v) >= 2000 for v in values):5d}"
    )


def print_pair_summary(
    base_name: str,
    other_name: str,
    base: list[int],
    other: list[int],
) -> None:
    deltas = [o - b for b, o in zip(base, other, strict=True)]
    print(
        f"{other_name:18} vs {base_name:18}"
        f" mean_delta={mean(deltas):8.2f}"
        f" mae={mean([abs(d) for d in deltas]):8.2f}"
        f" rmse={rmse(deltas):8.2f}"
        f" corr={corr(base, other):8.4f}"
        f" slope={slope(base, other):8.4f}"
        f" sign_disagree={sum((b > 0) != (o > 0) for b, o in zip(base, other, strict=True)):5d}"
    )


def grouped_pair_summary(
    fens: list[str],
    base: list[int],
    other: list[int],
    group_fn,
) -> list[tuple[str, int, float, float, float, float]]:
    groups: dict[str, tuple[list[int], list[int]]] = {}
    for fen, b, o in zip(fens, base, other, strict=True):
        key = group_fn(fen, b, o)
        xs, ys = groups.setdefault(key, ([], []))
        xs.append(b)
        ys.append(o)
    rows = []
    for key, (xs, ys) in sorted(groups.items()):
        deltas = [y - x for x, y in zip(xs, ys, strict=True)]
        rows.append((
            key,
            len(xs),
            mean(deltas),
            mean([abs(d) for d in deltas]),
            corr(xs, ys),
            slope(xs, ys),
        ))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="/home/petter/code/cpp/chess/enyo/build/enyo")
    ap.add_argument("--subject", action="append", required=True,
                    help="NAME=NET or NAME=ENGINE,NET. First subject is baseline.")
    ap.add_argument("--fen-file", type=Path)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--timeout-s", type=float, default=20.0)
    ap.add_argument("--uci-option", action="append", default=[],
                    help="Extra UCI option NAME=VALUE for every subject.")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    subjects = [parse_subject(raw, args.engine) for raw in args.subject]
    fens = load_fens(args.fen_file, args.limit)
    if not fens:
        raise SystemExit("no FENs loaded")

    print(f"rows={len(fens)}")
    print("subjects=" + ",".join(subject.name for subject in subjects))

    scores: dict[str, list[int]] = {}
    for subject in subjects:
        engine = UciEvalEngine(
            subject,
            threads=args.threads,
            hash_mb=args.hash,
            timeout_s=args.timeout_s,
            uci_options=args.uci_option,
        )
        try:
            values = []
            for fen in fens:
                values.append(engine.eval(fen))
            scores[subject.name] = values
        finally:
            engine.close()

    print("\nSubject summary")
    for subject in subjects:
        print_subject_summary(subject.name, scores[subject.name])

    base_name = subjects[0].name
    base = scores[base_name]
    print("\nPair summary")
    for subject in subjects[1:]:
        print_pair_summary(base_name, subject.name, base, scores[subject.name])

    print("\nGrouped by material phase")
    for subject in subjects[1:]:
        print(f"{subject.name} vs {base_name}")
        for key, rows, delta, mae, group_corr, group_slope in grouped_pair_summary(
            fens, base, scores[subject.name],
            lambda fen, _b, _o: material_bucket(fen),
        ):
            print(
                f"  {key:10} rows={rows:6d} mean_delta={delta:8.2f}"
                f" mae={mae:8.2f} corr={group_corr:8.4f}"
                f" slope={group_slope:8.4f}")

    print("\nGrouped by baseline eval")
    for subject in subjects[1:]:
        print(f"{subject.name} vs {base_name}")
        for key, rows, delta, mae, group_corr, group_slope in grouped_pair_summary(
            fens, base, scores[subject.name],
            lambda _fen, b, _o: eval_bucket(b),
        ):
            print(
                f"  {key:10} rows={rows:6d} mean_delta={delta:8.2f}"
                f" mae={mae:8.2f} corr={group_corr:8.4f}"
                f" slope={group_slope:8.4f}")

    if args.json_out:
        payload = {
            "fens": fens,
            "subjects": [subject.__dict__ for subject in subjects],
            "scores": scores,
            "features": [
                {
                    "side": side_to_move(fen),
                    "piece_count": piece_count(fen),
                    "non_pawn_material": non_pawn_material(fen),
                    "material_bucket": material_bucket(fen),
                }
                for fen in fens
            ],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
