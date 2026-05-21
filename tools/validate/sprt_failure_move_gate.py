#!/usr/bin/env python3
"""Build a move-choice gate from failed SPRT games."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import chess.pgn


@dataclass
class Target:
    game: int
    round_name: str
    fullmove: int
    ply: int
    side: str
    fen: str
    candidate_move: str
    reference_move: str
    legal_moves: int


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


class UciEngine:
    def __init__(
        self, engine: Path, net: Path, *, threads: int, hash_mb: int
    ) -> None:
        self.proc = subprocess.Popen(
            [str(engine)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        send(self.proc, "uci")
        read_until(self.proc, "uciok")
        send(self.proc, f"setoption name Threads value {threads}")
        send(self.proc, f"setoption name Hash value {hash_mb}")
        send(self.proc, f"setoption name nnue_file value {net}")
        send(self.proc, "isready")
        read_until(self.proc, "readyok")

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

    def bestmove(self, fen: str, nodes: int) -> tuple[str, int]:
        send(self.proc, f"position fen {fen}")
        send(self.proc, f"go nodes {nodes}")
        best = ""
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
                best = parts[1] if len(parts) > 1 else ""
                return best, nodes_seen


def candidate_lost(headers: chess.pgn.Headers, candidate_name: str) -> bool:
    result = headers.get("Result", "*")
    white = headers.get("White", "")
    black = headers.get("Black", "")
    if white == candidate_name:
        return result == "0-1"
    if black == candidate_name:
        return result == "1-0"
    return False


def candidate_color(headers: chess.pgn.Headers, candidate_name: str) -> chess.Color | None:
    if headers.get("White", "") == candidate_name:
        return chess.WHITE
    if headers.get("Black", "") == candidate_name:
        return chess.BLACK
    return None


def iter_candidate_positions(
    pgn: Path,
    *,
    candidate_name: str,
    lost_only: bool,
    min_ply: int,
    max_games: int,
) -> list[tuple[int, chess.pgn.Headers, int, chess.Board]]:
    out: list[tuple[int, chess.pgn.Headers, int, chess.Board]] = []
    games_seen = 0
    with pgn.open(encoding="utf-8", errors="replace") as handle:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            color = candidate_color(game.headers, candidate_name)
            if color is None:
                continue
            if lost_only and not candidate_lost(game.headers, candidate_name):
                continue
            games_seen += 1
            if max_games > 0 and games_seen > max_games:
                break
            board = game.board()
            ply = 0
            for move in game.mainline_moves():
                if board.turn == color and ply >= min_ply:
                    out.append((games_seen, game.headers, ply, board.copy(stack=False)))
                board.push(move)
                ply += 1
    return out


def score_cp(score: chess.engine.PovScore, turn: chess.Color) -> int:
    return score.pov(turn).score(mate_score=32000) or 0


def analyze_child(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    move: chess.Move,
    nodes: int,
) -> int:
    child = board.copy(stack=False)
    root_turn = board.turn
    child.push(move)
    info = engine.analyse(child, chess.engine.Limit(nodes=nodes))
    return score_cp(info["score"], root_turn)


def write_targets(path: Path, targets: list[Target]) -> None:
    fieldnames = [
        "game",
        "round",
        "fullmove",
        "ply",
        "side",
        "fen",
        "candidate_move",
        "reference_move",
        "legal_moves",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for target in targets:
            writer.writerow({
                "game": target.game,
                "round": target.round_name,
                "fullmove": target.fullmove,
                "ply": target.ply,
                "side": target.side,
                "fen": target.fen,
                "candidate_move": target.candidate_move,
                "reference_move": target.reference_move,
                "legal_moves": target.legal_moves,
            })


def gate_rows(scores: list[dict[str, object]], choice_field: str) -> list[dict[str, object]]:
    by_target: dict[tuple[str, str], dict[str, object]] = {}
    for row in scores:
        key = (str(row["log"]), str(row["ply"]))
        item = by_target.setdefault(key, {
            "log": row["log"],
            "ply": row["ply"],
            "fullmove": row["fullmove"],
            "side": row["side"],
            "fen": row["fen"],
            "best_move": row["move"],
            "engine_move": row[choice_field],
            "rank": 999,
            "gap_cp": 32000,
            "score_cp": 0,
        })
        if int(row["rank"]) == 1:
            item["best_move"] = row["move"]
        if row["move"] == row[choice_field]:
            item["rank"] = int(row["rank"])
            item["gap_cp"] = int(row["gap_cp"])
            item["score_cp"] = int(row["score_cp"])
    return list(by_target.values())


def gate_summary(rows: list[dict[str, object]]) -> dict[str, int]:
    total = len(rows)
    top1 = sum(1 for row in rows if int(row["rank"]) == 1)
    top3 = sum(1 for row in rows if int(row["rank"]) <= 3)
    gaps = [int(row["gap_cp"]) for row in rows]
    return {
        "targets": total,
        "top1": top1,
        "top3": top3,
        "sum_gap_cp": sum(gaps),
        "worst_gap_cp": max(gaps) if gaps else 0,
    }


def write_gate(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "log",
        "ply",
        "fullmove",
        "side",
        "fen",
        "best_move",
        "engine_move",
        "rank",
        "gap_cp",
        "score_cp",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    *,
    pgn_positions: int,
    targets: list[Target],
    candidate_rows: list[dict[str, object]],
    reference_rows: list[dict[str, object]],
) -> None:
    cand = gate_summary(candidate_rows)
    ref = gate_summary(reference_rows)
    candidate_better = 0
    reference_better = 0
    sum_diff = 0
    worst_regression = 0
    best_gain = 0
    for c_row, r_row in zip(candidate_rows, reference_rows):
        diff = int(r_row["gap_cp"]) - int(c_row["gap_cp"])
        sum_diff += diff
        if diff > 0:
            candidate_better += 1
        elif diff < 0:
            reference_better += 1
        worst_regression = min(worst_regression, diff)
        best_gain = max(best_gain, diff)

    lines = [
        f"pgn_candidate_positions={pgn_positions}",
        f"targets={len(targets)}",
        (
            "candidate "
            f"top1={cand['top1']}/{cand['targets']} "
            f"top3={cand['top3']}/{cand['targets']} "
            f"sum_gap_cp={cand['sum_gap_cp']} "
            f"worst_gap_cp={cand['worst_gap_cp']}"
        ),
        (
            "reference "
            f"top1={ref['top1']}/{ref['targets']} "
            f"top3={ref['top3']}/{ref['targets']} "
            f"sum_gap_cp={ref['sum_gap_cp']} "
            f"worst_gap_cp={ref['worst_gap_cp']}"
        ),
        f"candidate_better={candidate_better}",
        f"reference_better={reference_better}",
        f"sum_diff_cp={sum_diff}",
        f"worst_regression_cp={worst_regression}",
        f"best_gain_cp={best_gain}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score failed-SPRT positions where candidate/reference differ.")
    parser.add_argument("--pgn", required=True, type=Path)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--candidate-net", required=True, type=Path)
    parser.add_argument("--reference-net", required=True, type=Path)
    parser.add_argument("--stockfish", default="stockfish")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--lost-only", action="store_true", default=True)
    parser.add_argument("--include-nonlosses", dest="lost_only", action="store_false")
    parser.add_argument("--max-games", type=int, default=40)
    parser.add_argument("--max-targets", type=int, default=60)
    parser.add_argument("--min-ply", type=int, default=0)
    parser.add_argument("--engine-nodes", type=int, default=100000)
    parser.add_argument("--oracle-nodes", type=int, default=200000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=128)
    parser.add_argument("--stockfish-threads", type=int, default=1)
    parser.add_argument("--stockfish-hash", type=int, default=128)
    parser.add_argument("--syzygy-path", default="")
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    positions = iter_candidate_positions(
        args.pgn.expanduser(),
        candidate_name=args.candidate_name,
        lost_only=args.lost_only,
        min_ply=args.min_ply,
        max_games=args.max_games,
    )

    targets: list[Target] = []
    seen: set[str] = set()
    candidate = UciEngine(
        args.engine.expanduser(), args.candidate_net.expanduser(),
        threads=args.threads, hash_mb=args.hash)
    reference = UciEngine(
        args.engine.expanduser(), args.reference_net.expanduser(),
        threads=args.threads, hash_mb=args.hash)
    try:
        for game_no, headers, ply, board in positions:
            fen = board.fen()
            if fen in seen:
                continue
            seen.add(fen)
            cand_move, _ = candidate.bestmove(fen, args.engine_nodes)
            ref_move, _ = reference.bestmove(fen, args.engine_nodes)
            if cand_move == ref_move:
                continue
            targets.append(Target(
                game=game_no,
                round_name=headers.get("Round", ""),
                fullmove=board.fullmove_number,
                ply=ply,
                side="White" if board.turn == chess.WHITE else "Black",
                fen=fen,
                candidate_move=cand_move,
                reference_move=ref_move,
                legal_moves=board.legal_moves.count(),
            ))
            print(
                f"target {len(targets)}/{args.max_targets}: "
                f"game={game_no} ply={ply} cand={cand_move} ref={ref_move}",
                flush=True,
            )
            if len(targets) >= args.max_targets:
                break
    finally:
        candidate.close()
        reference.close()

    write_targets(out_dir / "targets.csv", targets)

    scores: list[dict[str, object]] = []
    with chess.engine.SimpleEngine.popen_uci(args.stockfish) as stockfish:
        options: dict[str, object] = {
            "Hash": args.stockfish_hash,
            "Threads": args.stockfish_threads,
        }
        if args.syzygy_path:
            options["SyzygyPath"] = str(Path(args.syzygy_path).expanduser())
        stockfish.configure(options)
        for index, target in enumerate(targets, start=1):
            board = chess.Board(target.fen)
            scored = []
            for move in board.legal_moves:
                cp = analyze_child(stockfish, board, move, args.oracle_nodes)
                scored.append((move.uci(), cp))
            scored.sort(key=lambda item: item[1], reverse=True)
            best = scored[0][1] if scored else 0
            for rank, (move, cp) in enumerate(scored, start=1):
                scores.append({
                    "log": f"sprt_game_{target.game}",
                    "fullmove": target.fullmove,
                    "ply": target.ply,
                    "side": target.side,
                    "fen": target.fen,
                    "move": move,
                    "rank": rank,
                    "score_cp": cp,
                    "gap_cp": best - cp,
                    "candidate_move": target.candidate_move,
                    "reference_move": target.reference_move,
                    "round": target.round_name,
                    "legal_moves": target.legal_moves,
                })
            print(
                f"scored {index}/{len(targets)} targets, "
                f"rows={len(scores)}",
                flush=True,
            )

    score_fields = [
        "log",
        "fullmove",
        "ply",
        "side",
        "fen",
        "move",
        "rank",
        "score_cp",
        "gap_cp",
        "candidate_move",
        "reference_move",
        "round",
        "legal_moves",
    ]
    with (out_dir / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=score_fields)
        writer.writeheader()
        writer.writerows(scores)

    candidate_rows = gate_rows(scores, "candidate_move")
    reference_rows = gate_rows(scores, "reference_move")
    write_gate(out_dir / "candidate_gate.csv", candidate_rows)
    write_gate(out_dir / "reference_gate.csv", reference_rows)
    write_summary(
        out_dir / "summary.txt",
        pgn_positions=len(positions),
        targets=targets,
        candidate_rows=candidate_rows,
        reference_rows=reference_rows,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
