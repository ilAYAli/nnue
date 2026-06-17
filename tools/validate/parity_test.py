#!/usr/bin/env python3
"""Compare Python .nn evaluation against Enyo's evalnet command."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import enyo_nnue as nn2
from lib.nnue_model import load_model_from_nn


DEFAULT_FENS = (
    "startpos",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r3k2r/pppq1ppp/2npbn2/4p3/2B1P3/2NP1N2/PPPQ1PPP/R3K2R b KQkq - 3 9",
    "8/2p5/3p4/3Pp3/2P1P3/8/8/4K2k w - e6 0 1",
)
EVALNET_RE = re.compile(r"^evalnet\s+(-?\d+)\s+cp\s+\(stm=(white|black)\)")


def normalize_fen(fen: str) -> str:
    if fen == "startpos":
        return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    return fen


def eval_python(net_path: Path, fens: list[str]) -> list[int]:
    model = load_model_from_nn(net_path, device="cpu")
    model.eval()
    scores: list[int] = []
    for raw_fen in fens:
        fen = normalize_fen(raw_fen)
        pieces, stm = nn2.parse_fen(fen)
        pieces.sort(key=lambda item: item[2])
        w_feats = nn2.features_from_pieces(
            pieces, nn2.WHITE, model.input_buckets, model.feature_channels)
        b_feats = nn2.features_from_pieces(
            pieces, nn2.BLACK, model.input_buckets, model.feature_channels)
        phase_scale = nn2.phase_scale_from_pieces(pieces)
        with torch.no_grad():
            score = model(
                torch.tensor(w_feats, dtype=torch.long),
                torch.tensor(b_feats, dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
                torch.tensor([stm], dtype=torch.long),
                torch.tensor([phase_scale], dtype=torch.float32),
                piece_count=torch.tensor([len(pieces)], dtype=torch.long),
            )
        scores.append(int(round(float(score.item()))))
    return scores


class EnyoEvalNet:
    def __init__(self, engine: Path, net: Path, *, timeout: float) -> None:
        self.timeout = timeout
        self.proc = subprocess.Popen(
            [str(engine)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.send("uci")
        self.wait_for("uciok")
        self.send(f"setoption name nnue_file value {net}")
        self.send("isready")
        self.wait_for("readyok")

    def close(self) -> None:
        if self.proc.poll() is None:
            self.send("quit")
            try:
                self.proc.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=self.timeout)

    def send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def readline(self) -> str:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if line == "":
            raise RuntimeError("engine exited before expected output")
        return line.rstrip("\n")

    def wait_for(self, marker: str) -> None:
        while True:
            line = self.readline()
            if marker in line:
                return

    def eval(self, fen: str) -> int:
        if fen == "startpos":
            self.send("position startpos")
        else:
            self.send(f"position fen {fen}")
        self.send("evalnet")
        while True:
            line = self.readline()
            match = EVALNET_RE.match(line)
            if match:
                return int(match.group(1))
            if line.startswith("evalnet error:"):
                raise RuntimeError(line)


def eval_engine(engine: Path, net_path: Path, fens: list[str], *, timeout: float) -> list[int]:
    engine_proc = EnyoEvalNet(engine, net_path, timeout=timeout)
    try:
        return [engine_proc.eval(fen) for fen in fens]
    finally:
        engine_proc.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--net", required=True, type=Path)
    parser.add_argument("--fen", action="append", default=[])
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    fens = args.fen or list(DEFAULT_FENS)
    net_path = args.net.expanduser().resolve()
    engine_path = args.engine.expanduser().resolve()
    if not net_path.is_file():
        raise SystemExit(f"missing net: {net_path}")
    if not engine_path.is_file():
        raise SystemExit(f"missing engine: {engine_path}")

    py_scores = eval_python(net_path, fens)
    engine_scores = eval_engine(engine_path, net_path, fens, timeout=args.timeout)

    failed = False
    for fen, py_score, engine_score in zip(fens, py_scores, engine_scores):
        diff = abs(py_score - engine_score)
        status = "ok" if diff <= args.tolerance else "FAIL"
        print(
            f"{status}: python={py_score:+d} engine={engine_score:+d} "
            f"diff={diff} fen={fen}",
            flush=True,
        )
        failed = failed or diff > args.tolerance

    if failed:
        print(f"FAIL: parity exceeded tolerance {args.tolerance} cp", file=sys.stderr)
        return 1
    print(f"PASS: parity within {args.tolerance} cp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
