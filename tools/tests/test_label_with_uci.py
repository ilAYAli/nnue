#!/usr/bin/env python3
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "score"))

import label_with_uci  # noqa: E402


def make_engine_with_transcript(
    *,
    id_name: str = "Stockfish 17.1",
    net_response: list[str] | None = None,
    use_net: bool = True,
    net_path: Path | None = None,
) -> label_with_uci.UciEngine:
    """Run UciEngine's handshake verification steps against a scripted transcript.

    Exercises _verify_engine_identity()/_verify_net_loaded() exactly as
    start() calls them, without spawning a real subprocess. _verify_net_loaded
    hashes the net file for real, so a real (temporary) file is required.
    """
    engine = object.__new__(label_with_uci.UciEngine)
    engine.net = str(net_path) if use_net else None
    engine.net_option = "EvalFile"
    engine.output_history = list(
        ([f"id name {id_name}"] if id_name else []) + ["id author test", "uciok"]
    )
    engine.net_load_confirmed = False
    engine.net_load_confirmation = None

    remaining = list(net_response or [])

    def fake_send(command: str) -> None:
        pass

    def fake_readline(*, timeout_s: float | None = None) -> str:
        if not remaining:
            raise label_with_uci.EngineTimeout("no more scripted output")
        return remaining.pop(0)

    engine.send = fake_send  # type: ignore[method-assign]
    engine.readline = fake_readline  # type: ignore[method-assign]

    label_with_uci.UciEngine._verify_engine_identity(engine)
    if engine.net is not None:
        label_with_uci.UciEngine._verify_net_loaded(engine, timeout_s=5.0)
    return engine


class FakeEngine:
    def __init__(self, path: str, *, threads: int, hash_mb: int) -> None:
        self.path = path

    def label(self, fen: str, *, depth: int, timeout_s: float) -> tuple[int | None, str | None]:
        return 42, None

    def close(self) -> None:
        pass


class TimeoutEngine(FakeEngine):
    restarts = 0

    def label(self, fen: str, *, depth: int, timeout_s: float) -> tuple[int | None, str | None]:
        raise label_with_uci.EngineTimeout("test timeout")

    def restart(self) -> None:
        type(self).restarts += 1


class LabelWithUciTests(unittest.TestCase):
    def test_restart_retries_a_failed_engine_handshake(self) -> None:
        engine = object.__new__(label_with_uci.UciEngine)
        engine.close = mock.Mock()
        engine.start = mock.Mock(side_effect=[RuntimeError("dead"), None])

        engine.restart(delay_s=0)

        self.assertEqual(2, engine.start.call_count)
        self.assertGreaterEqual(engine.close.call_count, 2)

    def test_restart_reports_all_failed_handshakes(self) -> None:
        engine = object.__new__(label_with_uci.UciEngine)
        engine.close = mock.Mock()
        engine.start = mock.Mock(side_effect=RuntimeError("dead"))

        with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
            engine.restart(delay_s=0)

        self.assertEqual(3, engine.start.call_count)

    def test_start_confirms_a_stockfish_style_net_load(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".nnue") as handle:
            handle.write(b"net-bytes")
            handle.flush()
            net_path = Path(handle.name)
            engine = make_engine_with_transcript(
                net_path=net_path,
                # Stockfish answers readyok immediately; the confirmation
                # only appears lazily, in response to the eval probe.
                net_response=["readyok", f"info string NNUE evaluation using {net_path.name} (1MiB)"],
            )
        self.assertTrue(engine.net_load_confirmed)
        self.assertIn(net_path.name, engine.net_load_confirmation)

    def test_start_confirms_an_enyo_style_net_load_by_content_hash(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".nn") as handle:
            handle.write(b"enyo-net-bytes")
            handle.flush()
            net_path = Path(handle.name)
            import hashlib
            digest = hashlib.sha256(b"enyo-net-bytes").hexdigest()
            engine = make_engine_with_transcript(
                net_path=net_path,
                # Enyo reports net-load status eagerly, before readyok.
                net_response=[
                    f"info string evaluator=enyo-nnue path='{net_path}' sha256={digest} hidden=1024",
                    "readyok",
                ],
            )
        self.assertTrue(engine.net_load_confirmed)
        self.assertIn(digest, engine.net_load_confirmation)

    def test_start_rejects_a_net_that_failed_to_load(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".nnue") as handle:
            handle.write(b"net-bytes")
            handle.flush()
            with self.assertRaisesRegex(RuntimeError, "failed to load"):
                make_engine_with_transcript(
                    net_path=Path(handle.name),
                    # Stockfish's ERROR line also only appears lazily.
                    net_response=[
                        "readyok",
                        "info string ERROR: Network evaluation parameters compatible with the engine must be available.",
                    ],
                )

    def test_start_rejects_an_enyo_architecture_mismatch_error(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".nnue") as handle:
            handle.write(b"net-bytes")
            handle.flush()
            with self.assertRaisesRegex(RuntimeError, "failed to load"):
                make_engine_with_transcript(
                    net_path=Path(handle.name),
                    net_response=[
                        "info string ERROR: nnue_file unsupported Stockfish NNUE architecture hash; "
                        "falling back to embedded default.nn",
                    ],
                )

    def test_start_rejects_a_silent_fallback_even_without_the_word_error(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".nnue") as handle:
            handle.write(b"net-bytes")
            handle.flush()
            with self.assertRaisesRegex(RuntimeError, "failed to load"):
                make_engine_with_transcript(
                    net_path=Path(handle.name),
                    net_response=["info string evaluator=enyo-nnue source=embedded sha256=deadbeef"],
                )

    def test_start_rejects_an_unconfirmed_net(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".nnue") as handle:
            handle.write(b"net-bytes")
            handle.flush()
            filler = ["readyok", "info string NNUE evaluation using nn-different.nnue (1MiB)"] + ["info string filler"] * 199
            with self.assertRaisesRegex(RuntimeError, "did not confirm"):
                make_engine_with_transcript(net_path=Path(handle.name), net_response=filler)

    def test_start_requires_an_id_name(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "did not report an id name"):
            make_engine_with_transcript(id_name="", use_net=False)

    def run_labeler(
        self,
        row: dict,
        *,
        output_format: str = "jsonl",
        engine_type: type = FakeEngine,
    ) -> tuple[Path, Path]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "in.jsonl"
            suffix = {
                "packed": ".npz",
                "bullet-data": ".bullet",
            }.get(output_format, ".txt")
            dst = root / f"out{suffix}"
            src.write_text(json.dumps(row) + "\n", encoding="utf-8")

            old_argv = sys.argv
            old_engine = label_with_uci.UciEngine
            try:
                label_with_uci.UciEngine = engine_type  # type: ignore[assignment]
                sys.argv = [
                    "label_with_uci.py",
                    "--input", str(src),
                    "--output", str(dst),
                    "--engine", "/tmp/stockfish",
                    "--depth", "12",
                    "--engine-timeout-s", "0.5",
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

    def test_skips_and_restarts_on_engine_timeout(self) -> None:
        TimeoutEngine.restarts = 0
        path, stats_path = self.run_labeler({
            "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
            "source": "timeout",
        }, engine_type=TimeoutEngine)
        try:
            self.assertEqual("", path.read_text(encoding="utf-8"))
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            self.assertEqual(1, stats["selected"])
            self.assertEqual(0, stats["written"])
            self.assertEqual(1, stats["skipped_timeout"])
            self.assertEqual(1, TimeoutEngine.restarts)
        finally:
            path.unlink(missing_ok=True)
            stats_path.unlink(missing_ok=True)

    def test_writes_packed_label_shard(self) -> None:
        if np is None:
            self.skipTest("numpy is not installed")
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
            "result": "1/2-1/2",
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
            "result": "1/2-1/2",
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
