from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from tools.forge import lc0_to_enyo_bullet as pipeline


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
            "shards": 1600,
            "min_ply": 16,
            "quiet_only": True,
        })()
        command = pipeline.build_command(args, Path("template.json"))
        self.assertIn("--net", command)
        self.assertIn("EvalFile", Path("tools/forge/label-lc0-stockfish-enyo.template.json").read_text())
        self.assertIn("--wait", command)

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


if __name__ == "__main__":
    unittest.main()
