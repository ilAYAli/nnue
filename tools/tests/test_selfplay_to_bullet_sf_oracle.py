#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO / "tools" / "bullet" / "selfplay_to_bullet_sf_oracle.py"
SPEC = importlib.util.spec_from_file_location("selfplay_to_bullet_sf_oracle", TOOL_PATH)
assert SPEC is not None
selfplay_to_bullet_sf_oracle = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(selfplay_to_bullet_sf_oracle)


class SelfplayToBulletSfOracleTests(unittest.TestCase):
    def test_empty_pgn_produces_forge_stats_without_starting_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pgn_dir = root / "pgn"
            pgn_dir.mkdir()
            (pgn_dir / "empty.pgn").touch()
            output = root / "dataset.bullet"
            stats_path = root / "dataset.stats.json"
            argv = [
                str(TOOL_PATH),
                "--pgn-dir", str(pgn_dir),
                "--output", str(output),
                "--stats", str(stats_path),
                "--shard-slice", "0/1",
            ]

            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    selfplay_to_bullet_sf_oracle,
                    "convert_shard",
                    side_effect=AssertionError("empty shard must not start the evaluator"),
                ),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(0, selfplay_to_bullet_sf_oracle.main())

            self.assertEqual(b"", output.read_bytes())
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            self.assertEqual("enyo.label-stats.v1", stats["schema"])
            self.assertEqual(0, stats["shard_index"])
            self.assertEqual(1, stats["read"])
            self.assertEqual(0, stats["written"])


if __name__ == "__main__":
    unittest.main()
