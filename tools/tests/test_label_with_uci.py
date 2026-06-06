#!/usr/bin/env python3
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "score"))

import label_with_uci  # noqa: E402


class FakeEngine:
    def __init__(self, path: str, *, threads: int, hash_mb: int) -> None:
        self.path = path

    def label(self, fen: str, *, depth: int) -> tuple[int | None, str | None]:
        return 42, None

    def close(self) -> None:
        pass


class LabelWithUciTests(unittest.TestCase):
    def run_labeler(self, row: dict, *, output_format: str = "jsonl") -> tuple[Path, Path]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "in.jsonl"
            suffix = {
                "packed": ".npz",
                "bullet-data": ".data",
            }.get(output_format, ".txt")
            dst = root / f"out{suffix}"
            src.write_text(json.dumps(row) + "\n", encoding="utf-8")

            old_argv = sys.argv
            old_engine = label_with_uci.UciEngine
            try:
                label_with_uci.UciEngine = FakeEngine  # type: ignore[assignment]
                sys.argv = [
                    "label_with_uci.py",
                    "--input", str(src),
                    "--output", str(dst),
                    "--engine", "/tmp/stockfish",
                    "--depth", "12",
                    "--progress", "0",
                    "--output-format", output_format,
                ]
                label_with_uci.main()
            finally:
                sys.argv = old_argv
                label_with_uci.UciEngine = old_engine  # type: ignore[assignment]

            data = dst.read_bytes()
            stats_data = dst.with_suffix(dst.suffix + ".stats.json").read_bytes()

        with tempfile.NamedTemporaryFile(delete=False, suffix=dst.suffix) as handle:
            handle.write(data)
            kept = Path(handle.name)
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=dst.suffix + ".stats.json",
        ) as handle:
            handle.write(stats_data)
            kept_stats = Path(handle.name)
        return kept, kept_stats

    def run_json_labeler(self, row: dict) -> dict:
        path, stats_path = self.run_labeler(row)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        finally:
            path.unlink(missing_ok=True)
            stats_path.unlink(missing_ok=True)

    def test_labels_rows_without_existing_score(self) -> None:
        row = self.run_json_labeler({
            "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
            "source": "lc0_training_data",
        })

        self.assertEqual(row["score"], 42)
        self.assertNotIn("source_score", row)
        self.assertEqual(row["teacher"], "stockfish")
        self.assertEqual(row["teacher_depth"], 12)

    def test_preserves_existing_score_as_source_score(self) -> None:
        row = self.run_json_labeler({
            "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
            "score": -17,
            "source": "enyo",
        })

        self.assertEqual(row["score"], 42)
        self.assertEqual(row["source_score"], -17)

    def test_writes_packed_label_shard(self) -> None:
        path, stats_path = self.run_labeler({
            "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
            "source_type": "selfplay",
        }, output_format="packed")
        try:
            with np.load(path, allow_pickle=False) as shard:
                self.assertEqual(str(shard["format"].item()), "enyo-packed-label-shard-v1")
                self.assertEqual(shard["score"].tolist(), [42.0])
                self.assertEqual(shard["counts"].tolist(), [2])
                self.assertEqual(json.loads(str(shard["source_map_json"].item())), {"selfplay": 0})
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            self.assertEqual(stats["packed"]["rows"], 1)
        finally:
            path.unlink(missing_ok=True)
            stats_path.unlink(missing_ok=True)

    def test_writes_bullet_text_shard(self) -> None:
        path, stats_path = self.run_labeler({
            "fen": "8/8/8/8/8/8/8/K6k b - - 0 1",
        }, output_format="bullet-text")
        try:
            self.assertEqual(
                path.read_text(encoding="utf-8").strip(),
                "8/8/8/8/8/8/8/K6k b - - 0 1 | -42 | 0.5",
            )
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            self.assertEqual(stats["written"], 1)
        finally:
            path.unlink(missing_ok=True)
            stats_path.unlink(missing_ok=True)

    def test_writes_bullet_data_shard(self) -> None:
        path, stats_path = self.run_labeler({
            "fen": "8/8/8/8/8/8/8/K6k b - - 0 1",
        }, output_format="bullet-data")
        try:
            data = path.read_bytes()
            self.assertEqual(len(data), 32)
            _occ, _pcs, score, result, _ksq, _opp_ksq, _extra = struct.unpack(
                "<Q16shBBB3s",
                data,
            )
            self.assertEqual(score, 42)
            self.assertEqual(result, 1)

            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            self.assertEqual(stats["written"], 1)
            self.assertEqual(stats["bullet_format"]["record_bytes"], 32)
            self.assertEqual(stats["bullet_format"]["records"], 1)
        finally:
            path.unlink(missing_ok=True)
            stats_path.unlink(missing_ok=True)
