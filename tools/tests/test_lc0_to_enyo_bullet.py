from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
import unittest


def sparse_file(path: Path, size: int) -> None:
    path.touch()
    os.truncate(path, size)

from tools.forge import lc0_to_enyo_bullet as pipeline


def make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        input=Path("/in"), output=Path("/out"), engine=Path("/eng"), net=Path("/net"),
        depth=12, threads=1, hash=128, min_ply=16, quiet_only=True,
        batch_bytes=20_000_000_000, target_task_bytes=100_000_000,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class Lc0ToEnyoBulletTests(unittest.TestCase):
    def test_small_source_is_rejected_without_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "training.1.tar").write_bytes(b"fixture")
            with self.assertRaisesRegex(SystemExit, "undersized LC0 input"):
                pipeline.preflight_source(root, allow_small=False)

    def test_source_preflight_counts_only_lc0_archives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "training.1.tar").write_bytes(b"one")
            (root / "other.tar").write_bytes(b"ignored")
            self.assertEqual((1, 3), pipeline.preflight_source(root, allow_small=True))

    def test_forge_command_has_the_three_labeling_contract_flags(self) -> None:
        args = type("Args", (), {
            "input": Path("/home/petter/assets/training/lc0/test91-forge-input"),
            "output": Path("/home/petter/assets/training/bullet/out.bullet"),
            "engine": Path("/home/petter/assets/engines/reference"),
            "net": Path("/home/petter/assets/nets/known.nnue"),
            "depth": 12,
            "threads": 1,
            "hash": 128,
            "engine_timeout_s": 120,
            "min_ply": 16,
            "quiet_only": True,
        })()
        command = pipeline.build_command(
            args, Path("template.json"),
            batch_input=Path("/tmp/batch-input"), batch_output_dir=Path("/tmp/batch-output"),
            shard_count=5, net_sha256="net-hash",
        )
        self.assertIn("--net", command)
        self.assertIn("--net-sha256", command)
        self.assertIn("net-hash", command)
        # Deliberately no --engine-sha256: each worker compiles its own
        # engine binary for its own CPU, so pinning it to the coordinator's
        # hash would make Forge auto-sync (overwrite) every worker's binary.
        self.assertNotIn("--engine-sha256", command)
        self.assertIn("nnue_file", Path("tools/forge/label-lc0-stockfish-enyo.template.json").read_text())

    def test_start_command_uses_the_validated_manifest(self) -> None:
        command = pipeline.build_start_command(Path("/tmp/validated.manifest.json"))
        self.assertEqual(command[:3], ["forge", "start", "/tmp/validated.manifest.json"])
        self.assertIn("--workers", command)
        self.assertNotIn("--wait", command)

    def test_resume_command_targets_the_existing_run(self) -> None:
        command = pipeline.build_resume_command("label-lc0-stockfish-enyo-input-0000-12-20260830-0000")
        self.assertEqual(
            command[:3],
            ["forge", "resume", "label-lc0-stockfish-enyo-input-0000-12-20260830-0000"],
        )
        self.assertIn("--workers", command)

    def test_materialized_partition_must_match_preflight(self) -> None:
        task = {
            "id": "label_0000",
            "inputs": [{
                "tree": "lc0-inventory",
                "digest": "digest",
                "files": 2,
                "bytes": 20,
                "path": "~/.cache/forge/task-inputs/digest",
                "inventory_source": "~/.cache/forge/task-inputs/digest",
            }],
        }
        with self.assertRaisesRegex(SystemExit, "partition changed"):
            pipeline.validate_materialized_partition(
                {"tasks": [task]},
                {"tasks": [{**task, "inputs": [{**task["inputs"][0], "files": 3}]}]},
            )

    def test_batch_shard_count_targets_task_bytes_per_batch(self) -> None:
        self.assertEqual(1, pipeline.batch_shard_count(50_000_000, 100_000_000, override=None))
        self.assertEqual(5, pipeline.batch_shard_count(462_000_000, 100_000_000, override=None))
        self.assertEqual(407, pipeline.batch_shard_count(40_661_790_720, 100_000_000, override=None))
        self.assertEqual(3, pipeline.batch_shard_count(1, 100_000_000, override=3))

    def test_build_batches_is_byte_weighted_per_batch_not_corpus_wide(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Sparse files: truncate to the target size instead of writing
            # hundreds of megabytes, to keep this test fast.
            sparse_file(root / "training.a.tar", 221_296_640)
            sparse_file(root / "training.b.tar", 241_762_880)
            sparse_file(root / "training.c.tar", 40_661_790_720)
            archives = pipeline.archive_paths(root)
            batches = pipeline.build_batches(
                archives, batch_bytes=20_000_000_000, target_task_bytes=100_000_000,
                shard_override=None,
            )
            shard_counts = sorted(batch.shard_count for batch in batches)
            # The huge archive alone must not starve the small batch's task
            # count the way a single corpus-wide --shards budget would.
            self.assertIn(407, shard_counts)
            self.assertGreater(min(shard_counts), 1)

    def test_work_state_detects_drift_and_refuses_to_replan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "training.a.tar"
            archive.write_bytes(b"data")
            archives = pipeline.archive_paths(root)
            batches = pipeline.build_batches(
                archives, batch_bytes=20_000_000_000, target_task_bytes=100_000_000,
                shard_override=None,
            )
            work_dir = root / "work"
            args = make_args()
            state = pipeline.init_or_verify_work_state(work_dir, args=args, archives=archives, batches=batches)
            self.assertEqual(len(batches), state["batch_count"])

            # Re-running with identical inputs must not raise.
            pipeline.init_or_verify_work_state(work_dir, args=args, archives=archives, batches=batches)

            # A changed source archive must be a hard failure, never a silent re-plan.
            archive.write_bytes(b"different data")
            drifted_archives = pipeline.archive_paths(root)
            with self.assertRaisesRegex(SystemExit, "drifted"):
                pipeline.init_or_verify_work_state(work_dir, args=args, archives=drifted_archives, batches=batches)

    def test_batch_plan_digest_changes_when_archives_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "training.a.tar"
            a.write_bytes(b"one")
            archives = pipeline.archive_paths(root)
            batches = pipeline.build_batches(
                archives, batch_bytes=20_000_000_000, target_task_bytes=100_000_000,
                shard_override=None,
            )
            digest_1 = pipeline.batch_plan_digest(batches[0])
            a.write_bytes(b"one-modified-content-longer")
            batches_after = pipeline.build_batches(
                pipeline.archive_paths(root), batch_bytes=20_000_000_000,
                target_task_bytes=100_000_000, shard_override=None,
            )
            digest_2 = pipeline.batch_plan_digest(batches_after[0])
            self.assertNotEqual(digest_1, digest_2)

    def test_validate_shards_rejects_wrong_net_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "state"
            log_dir = root / "logs"
            state_dir.mkdir()
            log_dir.mkdir()
            bullet = root / "task_0000.bullet"
            bullet.write_bytes(b"x" * 32)
            stats_path = root / "task_0000.stats.json"
            stats_path.write_text(json.dumps({
                "score_source": "uci", "error": None, "net_load_confirmed": True,
                "engine_sha256": "expected-engine", "net_sha256": "WRONG-net",
            }), encoding="utf-8")
            (state_dir / "task_0000.done.json").write_text(json.dumps({
                "rc": 0, "elapsed_s": 1.0, "task_execution_sha256": "abc",
            }), encoding="utf-8")
            manifest = {
                "state_dir": str(state_dir), "log_dir": str(log_dir),
                "tasks": [{"id": "task_0000", "outputs": [str(bullet), str(stats_path)]}],
            }
            batch_dir = root / "batch"
            with self.assertRaisesRegex(SystemExit, "net_sha256"):
                pipeline.validate_shards(
                    batch_dir, manifest,
                    expected_net_sha256="expected-net",
                )

    def test_validate_shards_rejects_a_missing_engine_hash(self) -> None:
        # engine_sha256 isn't gated on a specific value (each worker compiles
        # its own binary), but label.py must have hashed *something*.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "state"
            log_dir = root / "logs"
            state_dir.mkdir()
            log_dir.mkdir()
            bullet = root / "task_0000.bullet"
            bullet.write_bytes(b"x" * 32)
            stats_path = root / "task_0000.stats.json"
            stats_path.write_text(json.dumps({
                "score_source": "uci", "error": None, "net_load_confirmed": True,
                "engine_sha256": None, "net_sha256": "expected-net",
            }), encoding="utf-8")
            (state_dir / "task_0000.done.json").write_text(json.dumps({
                "rc": 0, "elapsed_s": 1.0, "task_execution_sha256": "abc",
            }), encoding="utf-8")
            manifest = {
                "state_dir": str(state_dir), "log_dir": str(log_dir),
                "tasks": [{"id": "task_0000", "outputs": [str(bullet), str(stats_path)]}],
            }
            batch_dir = root / "batch"
            with self.assertRaisesRegex(SystemExit, "engine_sha256"):
                pipeline.validate_shards(
                    batch_dir, manifest,
                    expected_net_sha256="expected-net",
                )

    def test_validate_shards_accepts_a_different_engine_hash_per_host(self) -> None:
        # Two shards from two different hosts, each with its own genuinely
        # distinct, correctly-compiled engine binary, must both be accepted.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "state"
            log_dir = root / "logs"
            state_dir.mkdir()
            log_dir.mkdir()
            tasks = []
            for index, engine_hash in enumerate(["host-a-engine-hash", "host-b-engine-hash"]):
                task_id = f"task_{index:04d}"
                bullet = root / f"{task_id}.bullet"
                bullet.write_bytes(b"x" * 32)
                stats_path = root / f"{task_id}.stats.json"
                stats_path.write_text(json.dumps({
                    "score_source": "uci", "error": None, "net_load_confirmed": True,
                    "engine_sha256": engine_hash, "net_sha256": "expected-net",
                }), encoding="utf-8")
                (state_dir / f"{task_id}.done.json").write_text(json.dumps({
                    "rc": 0, "elapsed_s": 1.0, "task_execution_sha256": "abc",
                }), encoding="utf-8")
                tasks.append({"id": task_id, "outputs": [str(bullet), str(stats_path)]})
            manifest = {"state_dir": str(state_dir), "log_dir": str(log_dir), "tasks": tasks}
            batch_dir = root / "batch"
            shard_paths = pipeline.validate_shards(
                batch_dir, manifest,
                expected_net_sha256="expected-net",
            )
            self.assertEqual(2, len(shard_paths))

    def test_validate_shards_accepts_a_clean_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "state"
            log_dir = root / "logs"
            state_dir.mkdir()
            log_dir.mkdir()
            bullet = root / "task_0000.bullet"
            bullet.write_bytes(b"x" * 32)
            stats_path = root / "task_0000.stats.json"
            stats_path.write_text(json.dumps({
                "score_source": "uci", "error": None, "net_load_confirmed": True,
                "engine_sha256": "expected-engine", "net_sha256": "expected-net",
            }), encoding="utf-8")
            (state_dir / "task_0000.done.json").write_text(json.dumps({
                "rc": 0, "elapsed_s": 1.0, "task_execution_sha256": "abc",
            }), encoding="utf-8")
            manifest = {
                "state_dir": str(state_dir), "log_dir": str(log_dir),
                "tasks": [{"id": "task_0000", "outputs": [str(bullet), str(stats_path)]}],
            }
            batch_dir = root / "batch"
            shard_paths = pipeline.validate_shards(
                batch_dir, manifest,
                expected_net_sha256="expected-net",
            )
            self.assertEqual([bullet], shard_paths)
            recorded = json.loads((batch_dir / "shards" / "task_0000.json").read_text())
            self.assertEqual("valid", recorded["status"])

    def test_cleanup_does_not_remove_static_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "lc0-root-old.bullet"
            old.write_bytes(b"old")
            sidecar = root / "lc0-root-old.bullet.calibration.json"
            sidecar.write_text("old", encoding="utf-8")
            static = root / "lc0-static-bulk.bullet"
            static.write_bytes(b"keep")
            removed = pipeline.cleanup_old_outputs(root)
            self.assertEqual({old, sidecar}, set(removed))
            self.assertFalse(old.exists())
            self.assertTrue(static.exists())

    def test_cleanup_unpacked_lc0_sources_removes_only_cache_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp) / "unpacked-lc0"
            batch_source = cache_root / "abc123"
            batch_source.mkdir(parents=True)
            (batch_source / "training.0.gz").write_bytes(b"data")

            outside_source = Path(tmp) / "not-cache" / "def456"
            outside_source.mkdir(parents=True)

            manifest = {
                "tasks": [
                    {
                        "inputs": [
                            {"tree": "lc0-inventory", "source": str(batch_source)},
                            {"tree": "other", "source": str(outside_source)},
                        ],
                    },
                    {
                        "inputs": [
                            {"tree": "lc0-inventory", "source": str(batch_source)},
                        ],
                    },
                ],
            }

            original_root = pipeline.UNPACKED_LC0_CACHE_ROOT
            pipeline.UNPACKED_LC0_CACHE_ROOT = cache_root.resolve()
            try:
                removed = pipeline.cleanup_unpacked_lc0_sources(manifest)
            finally:
                pipeline.UNPACKED_LC0_CACHE_ROOT = original_root

            self.assertEqual([str(batch_source)], removed)
            self.assertFalse(batch_source.exists())
            self.assertTrue(outside_source.exists())


if __name__ == "__main__":
    unittest.main()
