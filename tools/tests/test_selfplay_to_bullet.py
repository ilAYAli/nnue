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
TOOL_PATH = REPO / "tools" / "bullet" / "selfplay_to_bullet.py"
SPEC = importlib.util.spec_from_file_location("selfplay_to_bullet", TOOL_PATH)
assert SPEC is not None
selfplay_to_bullet = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(selfplay_to_bullet)


class SelfplayToBulletTests(unittest.TestCase):
    def test_empty_pgn_is_recorded_and_does_not_run_converter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pgn_dir = root / "pgn"
            pgn_dir.mkdir()
            (pgn_dir / "empty.pgn").touch()
            output = root / "dataset.bullet"
            stale_tmp = root / ".tmp"
            stale_tmp.mkdir()
            (stale_tmp / "empty.chunk.bullet").touch()
            (stale_tmp / "empty.stats.json").write_text("{}", encoding="utf-8")
            argv = [
                str(TOOL_PATH),
                "--pgn-dir", str(pgn_dir),
                "--output", str(output),
                "--label-mode", "self-distillation",
            ]

            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    selfplay_to_bullet,
                    "convert_shard",
                    side_effect=AssertionError("empty shard must not be converted"),
                ),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(0, selfplay_to_bullet.main())

            state = json.loads(
                (root / "dataset.state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(["empty.pgn"], state["processed_shards"])
            self.assertEqual(0, state["total_rows"])
            self.assertEqual(b"", output.read_bytes())
            self.assertFalse(stale_tmp.exists())

            skip = json.loads(
                (root / "dataset.skips.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual("empty_pgn", skip["skip_reason"])
            self.assertEqual(0, skip["bullet_written"])


if __name__ == "__main__":
    unittest.main()
