#!/usr/bin/env python3
"""Compare Python NNUE eval with Enyo runtime evalnet for exported nets."""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lib import enyo_nnue as nn2
from lib.nnue_model import load_model_from_nn


EVALNET_RE = re.compile(r"^evalnet\s+(-?\d+)\s+cp\s+\(stm=(white|black)\)")

FENS = (
    "rn1qkbnr/ppp2ppp/3p4/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 5",
    "r1bq1rk1/ppp2ppp/2np1n2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 b - - 5 7",
    "4k3/8/3p4/2pPp3/2P1P3/8/8/4K3 w - - 0 1",
    "4k3/8/8/2p1p3/1pPpPp2/1P1P1P2/8/4K3 b - - 0 1",
    "r3k2r/pp1n1ppp/2pbpn2/q7/3P4/2N1PN2/PPQB1PPP/R3KB1R w KQkq - 3 10",
    "2r2rk1/1bqnbppp/p2ppn2/1p6/3NP3/1BN1B3/PPPQ1PPP/2KR3R b - - 4 14",
    "8/2k5/8/2P1p3/4Pp2/5P2/4K3/8 b - - 0 53",
    "8/8/2k5/2P1p3/4Pp2/5P2/4K3/8 w - - 0 54",
)


class EngineError(RuntimeError):
    pass


class EvalNetEngine:
    def __init__(self, engine: Path, net: Path, timeout: float) -> None:
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
        self.wait_for("uciok")
        self.send(f"setoption name nnue_file value {net}")
        self.send("isready")
        self.wait_for("readyok")

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

    def read_line(self) -> str:
        try:
            line = self.lines.get(timeout=self.timeout)
        except queue.Empty as exc:
            raise EngineError("engine read timed out") from exc
        if line is None:
            raise EngineError("engine exited")
        return line

    def wait_for(self, token: str) -> None:
        while True:
            if token in self.read_line():
                return

    def evalnet(self, fen: str) -> int:
        self.send(f"position fen {fen}")
        self.send("evalnet")
        while True:
            line = self.read_line()
            match = EVALNET_RE.match(line)
            if match:
                return int(match.group(1))
            if line.startswith("evalnet error:"):
                raise EngineError(line)


@dataclass(frozen=True)
class FeatureSummary:
    base: int
    threats: int
    total: int
    min_index: int | None
    max_index: int | None
    out_of_range: int


def expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser()


def feature_summary(features: list[int], base_features: int,
                    model_features: int) -> FeatureSummary:
    return FeatureSummary(
        base=sum(1 for value in features if value < base_features),
        threats=sum(1 for value in features if value >= base_features),
        total=len(features),
        min_index=min(features) if features else None,
        max_index=max(features) if features else None,
        out_of_range=sum(1 for value in features if value < 0 or value >= model_features),
    )


def python_eval(model, fen: str) -> tuple[int, FeatureSummary, FeatureSummary]:
    pieces, stm = nn2.parse_fen(fen)
    pieces.sort(key=lambda item: item[2])
    w_feats = nn2.features_from_pieces(
        pieces, nn2.WHITE, model.input_buckets, model.feature_channels,
        model.full_threats)
    b_feats = nn2.features_from_pieces(
        pieces, nn2.BLACK, model.input_buckets, model.feature_channels,
        model.full_threats)
    phase_scale = nn2.phase_scale_from_pieces(pieces)
    piece_count = len(pieces)
    output_bucket = nn2.output_bucket_from_pieces(pieces, model.output_buckets)
    with torch.no_grad():
        score = model(
            torch.tensor(w_feats, dtype=torch.long),
            torch.tensor(b_feats, dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([stm], dtype=torch.long),
            torch.tensor([phase_scale], dtype=torch.float32),
            torch.tensor([output_bucket], dtype=torch.long),
            piece_count=torch.tensor([piece_count], dtype=torch.long),
        )
    base_features = nn2.feature_count(model.input_buckets, model.feature_channels)
    model_features = nn2.input_feature_count(
        model.input_buckets, model.feature_channels, model.full_threats)
    return (
        int(round(float(score.item()))),
        feature_summary(w_feats, base_features, model_features),
        feature_summary(b_feats, base_features, model_features),
    )


def finite(value: int | None) -> int | None:
    if value is None:
        return None
    if not math.isfinite(float(value)):
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default="~/assets/engines/candidate")
    parser.add_argument("--net", required=True)
    parser.add_argument("--fen", action="append", default=[])
    parser.add_argument("--tolerance", type=int, default=15)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--jsonl", action="store_true")
    args = parser.parse_args()

    engine = expand_path(args.engine)
    net = expand_path(args.net)
    fens = args.fen or list(FENS)
    model = load_model_from_nn(net, device="cpu")
    model.eval()

    failures = 0
    worst = 0
    runtime = EvalNetEngine(engine, net, args.timeout)
    try:
        for index, fen in enumerate(fens, start=1):
            py_score, white, black = python_eval(model, fen)
            engine_score = runtime.evalnet(fen)
            diff = py_score - engine_score
            worst = max(worst, abs(diff))
            ok = abs(diff) <= args.tolerance and white.out_of_range == 0 and black.out_of_range == 0
            failures += 0 if ok else 1
            row = {
                "index": index,
                "fen": fen,
                "python": py_score,
                "engine": engine_score,
                "diff": diff,
                "ok": ok,
                "white": white.__dict__,
                "black": black.__dict__,
            }
            if args.jsonl:
                print(json.dumps(row, separators=(",", ":")), flush=True)
            else:
                print(
                    f"{index}: python={py_score:+d} engine={engine_score:+d} "
                    f"diff={diff:+d} w={white.base}+{white.threats} "
                    f"b={black.base}+{black.threats} fen={fen}",
                    flush=True,
                )
    finally:
        runtime.close()

    if failures:
        print(
            f"FAIL: {failures}/{len(fens)} positions exceeded tolerance "
            f"or had invalid features; worst diff={worst} cp",
            file=sys.stderr,
        )
        return 1
    print(f"PASS: {len(fens)} positions within {args.tolerance} cp; worst diff={worst} cp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
