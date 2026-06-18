#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import chess


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "validate"))

import audit_move_policy_root  # noqa: E402


def compact_linear_model(preferred_to_square: chess.Square) -> dict:
    input_dim = 172
    weights = [0.0] * input_dim
    # compact features: 14 scalar fields, 64 from-square slots, 64 to-square slots.
    weights[14 + 64 + preferred_to_square] = 1.0
    return {
        "schema": "enyo.move_policy.v1",
        "feature_set": "compact",
        "threshold": 0.5,
        "mean": [0.0] * input_dim,
        "std": [1.0] * input_dim,
        "layers": [
            {
                "weight": [weights],
                "bias": [0.0],
            },
        ],
    }


class AuditMovePolicyRootTests(unittest.TestCase):
    def test_extracts_candidate_moves_and_policy_top(self) -> None:
        pgn = textwrap.dedent("""\
            [Event "Fastchess Tournament"]
            [Site "?"]
            [Date "2026.06.03"]
            [Round "7"]
            [White "candidate"]
            [Black "reference"]
            [Result "1-0"]

            1. e4 {+0.20/8 0.01s} e5 {-0.10/8 0.01s} 2. Nf3 {+0.30/8 0.01s} *
            """)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pgn_path = root / "game.pgn"
            out_path = root / "audit.jsonl"
            pgn_path.write_text(pgn, encoding="utf-8")

            rows, stats = audit_move_policy_root.iter_candidate_rows(
                pgn_path,
                model=compact_linear_model(chess.E4),
                candidate_name="candidate",
                engine=None,
                depth=0,
                movetime_ms=1,
                threshold=0.5,
                max_eval_drop=80,
                limit=0,
            )
            audit_move_policy_root.write_jsonl(out_path, rows)

            written = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(stats["candidate_games"], 1)
        self.assertEqual(stats["candidate_moves"], 2)
        self.assertEqual([row["played"] for row in written], ["e2e4", "g1f3"])
        self.assertEqual(written[0]["policy_best"], "e2e4")
        self.assertTrue(written[0]["policy_top_played"])
        self.assertEqual(written[0]["candidate_result"], "win")
        self.assertIsNone(written[0]["baseline_best"])


if __name__ == "__main__":
    unittest.main()
