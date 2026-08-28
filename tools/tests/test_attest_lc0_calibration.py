from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PATH = REPO / "tools" / "validate" / "attest_lc0_calibration.py"
SPEC = importlib.util.spec_from_file_location("attest_lc0_calibration", PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load attestor")
attestor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(attestor)


class AttestationTests(unittest.TestCase):
    def write_calibration(self, root: Path) -> Path:
        path = root / "calibration.json"
        path.write_text(json.dumps({
            "schema": "enyo.lc0-calibration.v1", "valid": True,
            "anchors": [[0, 0], [100, 50]], "holdout": {"passed": True},
            "reference_target": {"net_sha256": "net", "engine_sha256": ["engine"], "mode": "search", "depth": 1},
        }), encoding="utf-8")
        return path

    def test_attestation_rejects_stats_without_matching_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.bullet"
            corpus.write_bytes(bytes(32))
            stats = root / "stats"
            stats.mkdir()
            (stats / "one.stats.json").write_text(json.dumps({"written": 1, "calibration": None}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatched calibration"):
                attestor.attest(corpus, self.write_calibration(root), stats)

    def test_attestation_binds_exact_corpus_and_all_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.bullet"
            corpus.write_bytes(bytes(64))
            calibration = self.write_calibration(root)
            _, digest = attestor.lc0_calibration.load(calibration)
            stats = root / "stats"
            stats.mkdir()
            for index in range(2):
                (stats / f"{index}.stats.json").write_text(json.dumps({
                    "written": 1,
                    "calibration": {"schema": "enyo.lc0-calibration.v1", "sha256": digest},
                }), encoding="utf-8")
            manifest = attestor.attest(corpus, calibration, stats)
            self.assertTrue(manifest["valid"])
            self.assertEqual(2, manifest["records"])


if __name__ == "__main__":
    unittest.main()
