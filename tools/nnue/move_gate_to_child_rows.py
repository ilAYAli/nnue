from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess


def child_fen(fen: str, move_uci: str) -> str:
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        raise ValueError(f"illegal move {move_uci} in {fen}")
    board.push(move)
    return board.fen()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.cases.open() as src, args.output.open("w") as dst:
        for line in src:
            if not line.strip():
                continue
            case = json.loads(line)
            for kind, move_key in (("played", "played"), ("best", "best")):
                row = {
                    "fen": child_fen(case["fen"], case[move_key]),
                    "score": 0,
                    "wdl": 0.5,
                    "source_type": "hard_move_child",
                    "hard_parent_fen": case["fen"],
                    "hard_parent_source": case["source"],
                    "hard_parent_line": case["line"],
                    "hard_parent_ply": case["ply"],
                    "hard_parent_severity": case["severity"],
                    "hard_parent_loss_cp": case["loss_cp"],
                    "hard_child_kind": kind,
                    "hard_child_move": case[move_key],
                    "hard_parent_played": case["played"],
                    "hard_parent_best": case["best"],
                }
                dst.write(json.dumps(row, sort_keys=True) + "\n")
                written += 1
    print(f"wrote {written} child rows to {args.output}")


if __name__ == "__main__":
    main()
