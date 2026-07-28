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
    def test_shard_slice_partitions_sorted_pgns_without_overlap(self) -> None:
        paths = [Path(name) for name in ("d.pgn", "a.pgn", "c.pgn", "b.pgn")]
        sorted_paths = sorted(paths)

        slices = [
            selfplay_to_bullet.select_shards(sorted_paths, (index, 3))
            for index in range(3)
        ]

        self.assertEqual(sorted_paths, sorted(path for group in slices for path in group))
        self.assertEqual([], list(set(slices[0]) & set(slices[1])))
        self.assertEqual([Path("a.pgn"), Path("d.pgn")], slices[0])

    def test_shard_slice_rejects_invalid_values(self) -> None:
        for value in ("", "0", "x/2", "-1/2", "2/2", "0/0"):
            with self.subTest(value=value), self.assertRaises(
                selfplay_to_bullet.argparse.ArgumentTypeError
            ):
                selfplay_to_bullet.parse_shard_slice(value)

    def test_shard_ranges_partition_contiguous_pgns_without_overlap(self) -> None:
        paths = [Path(f"{index}.pgn") for index in range(10)]

        ranges = [
            selfplay_to_bullet.select_shard_range(paths, (0, 3)),
            selfplay_to_bullet.select_shard_range(paths, (3, 4)),
            selfplay_to_bullet.select_shard_range(paths, (7, 3)),
        ]

        self.assertEqual(paths, [path for group in ranges for path in group])

    def test_shard_range_rejects_invalid_values(self) -> None:
        for value in ("", "0", "x/2", "-1/2", "0/0"):
            with self.subTest(value=value), self.assertRaises(
                selfplay_to_bullet.argparse.ArgumentTypeError
            ):
                selfplay_to_bullet.parse_shard_range(value)

    def test_empty_pgn_is_recorded_and_does_not_run_converter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pgn_dir = root / "pgn"
            pgn_dir.mkdir()
            (pgn_dir / "empty.pgn").touch()
            output = root / "dataset.bullet"
            stats_path = root / "dataset.stats.json"
            stale_tmp = root / ".tmp"
            stale_tmp.mkdir()
            (stale_tmp / "empty.chunk.bullet").touch()
            (stale_tmp / "empty.stats.json").write_text("{}", encoding="utf-8")
            argv = [
                str(TOOL_PATH),
                "--pgn-dir", str(pgn_dir),
                "--output", str(output),
                "--stats", str(stats_path),
                "--label-mode", "self-distillation",
                "--shard-slice", "0/1",
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
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            self.assertEqual("enyo.label-stats.v1", stats["schema"])
            self.assertEqual(0, stats["shard_index"])
            self.assertEqual(1, stats["read"])
            self.assertEqual(0, stats["written"])


if __name__ == "__main__":
    unittest.main()
