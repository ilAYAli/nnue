#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import importlib.util
import io
import json
import sys
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "posgen"))

import lc0_to_jsonl  # noqa: E402

LABEL_PATH = REPO / "tools" / "score" / "label.py"
LABEL_SPEC = importlib.util.spec_from_file_location("streaming_label", LABEL_PATH)
if LABEL_SPEC is None or LABEL_SPEC.loader is None:
    raise RuntimeError(f"cannot load {LABEL_PATH}")
labeler = importlib.util.module_from_spec(LABEL_SPEC)
LABEL_SPEC.loader.exec_module(labeler)


class FakeEngine:
    last_net_option: str | None = None

    def __init__(
        self,
        path: str,
        *,
        threads: int,
        hash_mb: int,
        net: str | None = None,
        net_option: str = "nnue_file",
    ) -> None:
        type(self).last_net_option = net_option

    def label(self, fen: str, *, depth: int, timeout_s: float) -> tuple[int, None]:
        return 37, None

    def static_eval(self, fen: str, *, timeout_s: float) -> int:
        return 37

    def restart(self) -> None:
        pass

    def close(self) -> None:
        pass


def row(move: str = "a2a3") -> dict:
    return {
        "fen": "7k/8/8/8/8/8/P7/K7 w - - 0 1",
        "result": "1/2-1/2",
        "wdl": 0.5,
        "moves": [{"move": move, "roles": ["played"], "legal": True}],
    }


class StreamingLabelingTests(unittest.TestCase):
    def test_script_entrypoint_loads(self) -> None:
        result = subprocess.run(
            [sys.executable, str(LABEL_PATH), "--help"],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)

    def test_inventory_partition_happens_before_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = [
                {"path": f"training.{index}.gz", "size": index, "sha256": str(index) * 64}
                for index in range(6)
            ]
            inventory = root / "inventory.json"
            inventory.write_text(json.dumps({
                "schema": lc0_to_jsonl.INVENTORY_SCHEMA,
                "files": files,
            }), encoding="utf-8")

            shards = [
                lc0_to_jsonl.selected_input_paths(
                    root,
                    inventory=inventory,
                    shard_count=3,
                    shard_index=index,
                )
                for index in range(3)
            ]
            self.assertEqual(
                [[path.name for path in shard] for shard in shards],
                [["training.0.gz", "training.3.gz"],
                 ["training.1.gz", "training.4.gz"],
                 ["training.2.gz", "training.5.gz"]],
            )
            self.assertEqual(6, len(set().union(*(set(shard) for shard in shards))))

    def test_directory_input_includes_test91_tar_archives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "training-run2-test91-20260427-2317.tar"
            archive.touch()
            ignored = root / "unrelated.tar"
            ignored.touch()
            paths = lc0_to_jsonl.selected_input_paths(root)
            self.assertEqual([archive], paths)

    def test_record_quotas_sum_exactly(self) -> None:
        quotas = [lc0_to_jsonl.split_record_limit(10, 4, index) for index in range(4)]
        self.assertEqual([3, 3, 2, 2], quotas)
        self.assertEqual(10, sum(quotas))

    def test_rejects_noncanonical_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inventory = root / "inventory.json"
            inventory.write_text(json.dumps({
                "schema": lc0_to_jsonl.INVENTORY_SCHEMA,
                "files": [{"path": "z"}, {"path": "a"}],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sorted"):
                lc0_to_jsonl.inventory_paths(inventory, root)

    def test_quiet_filter(self) -> None:
        self.assertTrue(labeler.is_quiet(row()))
        self.assertFalse(labeler.is_quiet(row("a2a3q")))

    def test_tar_input_is_decoded_as_compressed_training_members(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "test91.tar"
            raw = bytearray(lc0_to_jsonl.RECORD_SIZE)
            lc0_to_jsonl.HEADER.pack_into(raw, 0, 6, 1)
            with tarfile.open(archive, "w") as tar:
                payload = gzip.compress(bytes(raw))
                member = tarfile.TarInfo("test91/training-run2-test91-20260427-2317.gz")
                member.size = len(payload)
                tar.addfile(member, io.BytesIO(payload))

            stats = lc0_to_jsonl.Stats()
            self.assertEqual(
                [],
                list(lc0_to_jsonl.iter_rows(archive, top_policy=0, stats=stats)),
            )
            self.assertEqual(1, stats.files)
            self.assertEqual(1, stats.records)
            self.assertEqual(0, stats.unsupported_records)
            self.assertEqual(1, stats.invalid_boards)

    def test_streams_to_atomic_bullet_and_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "shard.bullet"
            stats_path = root / "shard.stats.json"
            args = argparse.Namespace(
                input=root,
                inventory=None,
                output=output,
                stats=stats_path,
                engine="stockfish",
                net=None,
                net_option="EvalFile",
                score_source="uci",
                eval_scale=400.0,
                value_epsilon=1e-6,
                static=False,
                depth=12,
                threads=1,
                hash=16,
                max_records=5,
                min_ply=1,
                quiet_only=True,
                shard_count=1,
                shard_index=0,
                engine_timeout_s=1.0,
                max_abs_cp=10000,
                progress=0,
                enyo_runtime_target=False,
            )

            def rows(*args: object, stats: lc0_to_jsonl.Stats, **kwargs: object):
                stats.files = 1
                stats.records = 3
                yield row(), 0
                yield row(), 1
                yield row(), 2

            with mock.patch.object(lc0_to_jsonl, "iter_rows", rows):
                stats = labeler.label(args, engine_type=FakeEngine)

            self.assertEqual("EvalFile", FakeEngine.last_net_option)

            self.assertEqual(64, output.stat().st_size)
            self.assertEqual(3, stats["read"])
            self.assertEqual(2, stats["selected"])
            self.assertEqual(2, stats["written"])
            self.assertIsNone(stats["inventory"])
            self.assertEqual("enyo.label-stats.v2", stats["schema"])
            self.assertEqual(stats, json.loads(stats_path.read_text(encoding="utf-8")))
            self.assertEqual([], list(root.glob("*.partial.*")))

    def test_missing_result_is_rejected_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = argparse.Namespace(
                input=root,
                inventory=None,
                output=root / "shard.bullet",
                stats=root / "shard.stats.json",
                engine="stockfish",
                net=None,
                net_option="EvalFile",
                score_source="uci",
                eval_scale=400.0,
                value_epsilon=1e-6,
                static=False,
                depth=12,
                threads=1,
                hash=16,
                max_records=5,
                min_ply=1,
                quiet_only=True,
                shard_count=1,
                shard_index=0,
                engine_timeout_s=1.0,
                max_abs_cp=10000,
                progress=0,
                enyo_runtime_target=False,
            )

            def rows(*args: object, stats: lc0_to_jsonl.Stats, **kwargs: object):
                stats.files = 1
                stats.records = 1
                missing = row()
                del missing["result"]
                yield missing, 1

            with mock.patch.object(lc0_to_jsonl, "iter_rows", rows):
                with self.assertRaisesRegex(ValueError, "ground-truth result"):
                    labeler.label(args, engine_type=FakeEngine)

    def test_lc0_root_score_is_stm_relative_and_needs_no_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "shard.bullet"
            stats_path = root / "shard.stats.json"
            args = argparse.Namespace(
                input=root,
                inventory=root / "inventory.json",
                output=output,
                stats=stats_path,
                engine="must-not-be-opened",
                net=None,
                net_option="EvalFile",
                score_source="lc0-root",
                eval_scale=400.0,
                value_epsilon=1e-6,
                static=False,
                depth=12,
                threads=1,
                hash=16,
                max_records=5,
                min_ply=1,
                quiet_only=True,
                shard_count=1,
                shard_index=0,
                engine_timeout_s=1.0,
                max_abs_cp=10000,
                progress=0,
                enyo_runtime_target=False,
            )
            black = row()
            black["fen"] = "k7/p7/8/8/8/8/8/7K b - - 0 1"
            black["moves"] = [{"move": "a7a6", "roles": ["played"], "legal": True}]
            black["lc0"] = {"root_q": 0.8, "root_d": 0.0}
            white = row()
            white["lc0"] = {"root_q": 0.8, "root_d": 0.0}
            invalid = row()
            invalid["lc0"] = {"root_q": 2.0, "root_d": 0.0}

            def rows(*args: object, stats: lc0_to_jsonl.Stats, **kwargs: object):
                stats.files = 1
                stats.records = 3
                yield white, 1
                yield black, 2
                yield invalid, 3

            with mock.patch.object(lc0_to_jsonl, "iter_rows", rows):
                stats = labeler.label(args, engine_type=FakeEngine)

            self.assertEqual(2, stats["written"])
            self.assertEqual(1, stats["skipped_invalid_lc0_value"])
            self.assertEqual(2, stats["lc0_root_value_count"])
            self.assertEqual(879, stats["lc0_root_score_cp_min"])
            self.assertEqual(879, stats["lc0_root_score_cp_max"])
            self.assertEqual(0.9, stats["lc0_root_probability_min"])
            self.assertEqual(0.9, stats["lc0_root_probability_max"])
            self.assertEqual(64, output.stat().st_size)
            scores = [
                labeler.bullet_format.STRUCT.unpack_from(output.read_bytes(), offset)[2]
                for offset in range(0, output.stat().st_size, 32)
            ]
            self.assertEqual([879, 879], scores)

    def test_lc0_root_score_uses_logit_and_clamps_extremes(self) -> None:
        score, probability, low, high, out_of_range = labeler.lc0_root_score(
            {"lc0": {"root_q": 0.8, "root_d": 0.6}},
            eval_scale=400.0,
            value_epsilon=1e-6,
        )
        self.assertAlmostEqual(0.9, probability)
        self.assertEqual(879, score)
        self.assertFalse(low)
        self.assertFalse(high)
        self.assertFalse(out_of_range)

        score, probability, low, high, out_of_range = labeler.lc0_root_score(
            {"lc0": {"root_q": 1.0, "root_d": 0.0}},
            eval_scale=400.0,
            value_epsilon=1e-6,
        )
        self.assertEqual(1.0, probability)
        self.assertEqual(5526, score)
        self.assertFalse(low)
        self.assertTrue(high)
        self.assertFalse(out_of_range)

        score, probability, low, high, out_of_range = labeler.lc0_root_score(
            {"lc0": {"root_q": -1.0, "root_d": 1e-8}},
            eval_scale=400.0,
            value_epsilon=1e-6,
        )
        self.assertEqual(-5526, score)
        self.assertEqual(0.0, probability)
        self.assertTrue(low)
        self.assertFalse(high)
        self.assertFalse(out_of_range)

    def test_empty_shard_retains_decoder_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "shard.bullet"
            stats_path = root / "shard.stats.json"
            args = argparse.Namespace(
                input=root,
                inventory=None,
                output=output,
                stats=stats_path,
                engine="stockfish",
                net=None,
                net_option="EvalFile",
                score_source="lc0-root",
                eval_scale=400.0,
                value_epsilon=1e-6,
                static=False,
                depth=12,
                threads=1,
                hash=16,
                max_records=5,
                min_ply=2,
                quiet_only=True,
                shard_count=1,
                shard_index=0,
                engine_timeout_s=1.0,
                max_abs_cp=10000,
                progress=0,
                enyo_runtime_target=False,
            )

            def rows(*args: object, stats: lc0_to_jsonl.Stats, **kwargs: object):
                stats.files = 1
                stats.records = 1
                yield row(), 1

            with mock.patch.object(lc0_to_jsonl, "iter_rows", rows):
                with self.assertRaisesRegex(ValueError, "Bullet shard is empty"):
                    labeler.label(args, engine_type=FakeEngine)

            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            self.assertEqual("Bullet shard is empty", stats["error"])
            self.assertEqual(1, stats["read"])
            self.assertEqual(1, stats["skipped_min_ply"])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
