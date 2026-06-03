#!/usr/bin/env python3
"""Audit sidecar move-policy root choices from an SPRT PGN."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import chess
    import chess.pgn
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("python-chess is required: python3 -m pip install chess") from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validate.eval_move_policy_export import load_model, score  # noqa: E402


SCORE_RE = re.compile(r"\bscore\s+(cp|mate)\s+(-?\d+)\b")
PV_RE = re.compile(r"\bpv\s+(.+)$")
EVALNET_RE = re.compile(r"^evalnet\s+(-?\d+)\s+cp\s+\(stm=(white|black)\)")


class UciEngine:
    def __init__(self, path: str, *, uci: list[str], timeout_s: float) -> None:
        self.timeout_s = timeout_s
        self.proc = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.send("uci")
        self.wait_for("uciok")
        for item in uci:
            name, value = parse_name_value(item)
            self.setoption(name, value)
        self.send("isready")
        self.wait_for("readyok")

    def close(self) -> None:
        try:
            self.send("quit")
        finally:
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

    def send(self, command: str) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("engine stdin closed")
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def readline(self) -> str:
        if self.proc.stdout is None:
            raise RuntimeError("engine stdout closed")
        line = self.proc.stdout.readline()
        if line == "":
            raise RuntimeError("engine exited")
        return line.strip()

    def wait_for(self, token: str) -> None:
        while True:
            if self.readline() == token:
                return

    def setoption(self, name: str, value: str) -> None:
        if value == "":
            self.send(f"setoption name {name} value")
        else:
            self.send(f"setoption name {name} value {value}")

    def search(self, fen: str, *, depth: int, movetime_ms: int) -> dict[str, Any]:
        self.send(f"position fen {fen}")
        if depth > 0:
            self.send(f"go depth {depth}")
        elif movetime_ms > 0:
            self.send(f"go movetime {movetime_ms}")
        else:
            raise ValueError("depth or movetime_ms must be positive")

        score_cp: int | None = None
        mate: int | None = None
        pv: list[str] = []
        while True:
            line = self.readline()
            score_match = SCORE_RE.search(line)
            if score_match:
                kind, raw_value = score_match.groups()
                if kind == "cp":
                    score_cp = int(raw_value)
                    mate = None
                else:
                    mate = int(raw_value)
            pv_match = PV_RE.search(line)
            if pv_match:
                pv = pv_match.group(1).split()
            if line.startswith("bestmove "):
                parts = line.split()
                return {
                    "bestmove": parts[1] if len(parts) > 1 else None,
                    "score_cp": score_cp,
                    "mate": mate,
                    "pv": pv,
                }

    def evalnet(self, fen: str) -> int | None:
        self.send(f"position fen {fen}")
        self.send("evalnet")
        while True:
            line = self.readline()
            match = EVALNET_RE.match(line)
            if match:
                return int(match.group(1))
            if line.startswith("evalnet error:"):
                return None


def parse_name_value(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"expected NAME=VALUE, got {raw!r}")
    name, value = raw.split("=", 1)
    if not name:
        raise ValueError(f"empty UCI option name in {raw!r}")
    return name, value


def candidate_result(result: str, candidate_color: chess.Color) -> str:
    if result == "1/2-1/2":
        return "draw"
    if result == "1-0":
        return "win" if candidate_color == chess.WHITE else "loss"
    if result == "0-1":
        return "win" if candidate_color == chess.BLACK else "loss"
    return "unknown"


def parent_cp_after_move(engine: UciEngine, board: chess.Board, move_uci: str) -> int | None:
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        return None
    child = board.copy(stack=False)
    child.push(move)
    child_score = engine.evalnet(child.fen(en_passant="fen"))
    if child_score is None:
        return None
    return -child_score


def policy_scores(model: dict[str, Any], fen: str) -> list[dict[str, Any]]:
    board = chess.Board(fen)
    out = []
    for move in board.legal_moves:
        move_uci = move.uci()
        out.append({"move": move_uci, "policy_score": score(model, fen, move_uci)})
    out.sort(key=lambda item: (-float(item["policy_score"]), str(item["move"])))
    for rank, item in enumerate(out, start=1):
        item["policy_rank"] = rank
    return out


def move_policy_rank(scores: list[dict[str, Any]], move_uci: str) -> int | None:
    for item in scores:
        if item["move"] == move_uci:
            return int(item["policy_rank"])
    return None


def move_policy_score(scores: list[dict[str, Any]], move_uci: str) -> float | None:
    for item in scores:
        if item["move"] == move_uci:
            return float(item["policy_score"])
    return None


def iter_candidate_rows(
    pgn_path: Path,
    *,
    model: dict[str, Any],
    candidate_name: str,
    engine: UciEngine | None,
    depth: int,
    movetime_ms: int,
    threshold: float,
    max_eval_drop: int,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stats = {
        "games": 0,
        "candidate_games": 0,
        "candidate_moves": 0,
        "policy_top_played": 0,
        "policy_top_differs_from_baseline": 0,
        "guard_like_triggers": 0,
        "missing_baseline": 0,
    }
    with pgn_path.open(encoding="utf-8", errors="replace") as handle:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            stats["games"] += 1
            headers = game.headers
            candidate_color: chess.Color | None = None
            if headers.get("White") == candidate_name:
                candidate_color = chess.WHITE
            elif headers.get("Black") == candidate_name:
                candidate_color = chess.BLACK
            if candidate_color is None:
                continue
            stats["candidate_games"] += 1

            board = game.board()
            result = headers.get("Result", "*")
            round_id = headers.get("Round", str(stats["games"]))
            game_result = candidate_result(result, candidate_color)
            for ply, node in enumerate(game.mainline()):
                move = node.move
                if move is None:
                    continue
                if board.turn == candidate_color:
                    fen = board.fen(en_passant="fen")
                    played = move.uci()
                    scores = policy_scores(model, fen)
                    top_policy = scores[0]
                    played_policy_score = move_policy_score(scores, played)
                    top_margin_vs_played = (
                        float(top_policy["policy_score"]) - float(played_policy_score)
                        if played_policy_score is not None else None
                    )

                    baseline: dict[str, Any] = {}
                    baseline_move: str | None = None
                    baseline_policy_score: float | None = None
                    top_margin_vs_baseline: float | None = None
                    baseline_eval_cp: int | None = None
                    policy_eval_cp: int | None = None
                    eval_drop_cp: int | None = None
                    if engine is not None:
                        baseline = engine.search(
                            fen, depth=depth, movetime_ms=movetime_ms)
                        baseline_move = baseline.get("bestmove")
                        if baseline_move:
                            baseline_policy_score = move_policy_score(scores, baseline_move)
                            if baseline_policy_score is not None:
                                top_margin_vs_baseline = (
                                    float(top_policy["policy_score"]) - baseline_policy_score
                                )
                            baseline_eval_cp = parent_cp_after_move(engine, board, baseline_move)
                        policy_eval_cp = parent_cp_after_move(
                            engine, board, str(top_policy["move"]))
                        if baseline_eval_cp is not None and policy_eval_cp is not None:
                            eval_drop_cp = baseline_eval_cp - policy_eval_cp
                    else:
                        stats["missing_baseline"] += 1

                    guard_like = (
                        top_margin_vs_baseline is not None
                        and eval_drop_cp is not None
                        and top_margin_vs_baseline >= threshold
                        and eval_drop_cp <= max_eval_drop
                    )
                    if played == top_policy["move"]:
                        stats["policy_top_played"] += 1
                    if baseline_move and top_policy["move"] != baseline_move:
                        stats["policy_top_differs_from_baseline"] += 1
                    if guard_like:
                        stats["guard_like_triggers"] += 1

                    row = {
                        "schema": "enyo.move_policy_root_audit.v1",
                        "id": f"{round_id}:{ply}:{played}",
                        "source": str(pgn_path),
                        "round": round_id,
                        "ply": ply,
                        "fen": fen,
                        "candidate_color": "white" if candidate_color else "black",
                        "result": result,
                        "candidate_result": game_result,
                        "played": played,
                        "played_san": board.san(move),
                        "played_comment": node.comment,
                        "played_policy_rank": move_policy_rank(scores, played),
                        "played_policy_score": played_policy_score,
                        "policy_best": top_policy["move"],
                        "policy_best_score": top_policy["policy_score"],
                        "policy_margin_vs_played": top_margin_vs_played,
                        "baseline_best": baseline_move,
                        "baseline_score_cp": baseline.get("score_cp"),
                        "baseline_mate": baseline.get("mate"),
                        "baseline_pv": baseline.get("pv", []),
                        "baseline_policy_score": baseline_policy_score,
                        "policy_margin_vs_baseline": top_margin_vs_baseline,
                        "baseline_child_eval_cp": baseline_eval_cp,
                        "policy_child_eval_cp": policy_eval_cp,
                        "eval_drop_cp": eval_drop_cp,
                        "threshold": threshold,
                        "max_eval_drop": max_eval_drop,
                        "guard_like_trigger": guard_like,
                        "policy_top_played": played == top_policy["move"],
                    }
                    rows.append(row)
                    stats["candidate_moves"] += 1
                    if limit > 0 and len(rows) >= limit:
                        return rows, stats
                board.push(move)
    return rows, stats


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Audit sidecar root choices from a candidate/reference SPRT PGN.")
    ap.add_argument("--pgn", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--candidate-name", default="candidate")
    ap.add_argument("--engine")
    ap.add_argument("--engine-uci", action="append", default=[],
                    metavar="NAME=VALUE")
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--movetime-ms", type=int, default=100)
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--max-eval-drop", type=int, default=80)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    model = load_model(args.model)
    threshold = args.threshold
    if threshold is None:
        threshold = float(model.get("threshold", 18.0))
    engine = None
    if args.engine:
        engine = UciEngine(args.engine, uci=args.engine_uci, timeout_s=30.0)
    try:
        rows, stats = iter_candidate_rows(
            args.pgn,
            model=model,
            candidate_name=args.candidate_name,
            engine=engine,
            depth=args.depth,
            movetime_ms=args.movetime_ms,
            threshold=threshold,
            max_eval_drop=args.max_eval_drop,
            limit=args.limit,
        )
    finally:
        if engine is not None:
            engine.close()

    write_jsonl(args.output, rows)
    summary = {
        "pgn": str(args.pgn),
        "model": str(args.model),
        "output": str(args.output),
        "threshold": threshold,
        "max_eval_drop": args.max_eval_drop,
        "engine": args.engine,
        "rows": len(rows),
        **stats,
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
