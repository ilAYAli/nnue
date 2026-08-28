from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PATH = REPO / "tools" / "score" / "lc0_calibration.py"
SPEC = importlib.util.spec_from_file_location("lc0_calibration", PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load calibration module")
calibration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibration)


def pairs() -> list[dict]:
    # Fixed synthetic relationship: the fit must recover a 0.5 scale while
    # the holdout remains independent of the fitting rows.
    result = []
    for index in range(200):
        raw = (-1 if index % 2 else 1) * (20 + index * 4)
        result.append({"raw_score": raw, "target_score": raw // 2,
                       "split": "holdout" if index % 5 == 0 else "fit",
                       "reference_engine_sha256": "engine-a",
                       "reference_net_sha256": "net-a"})
    return result


class CalibrationTests(unittest.TestCase):
    def test_split_is_stable_and_uses_both_sets(self) -> None:
        values = [calibration.deterministic_split("a.gz", index) for index in range(100)]
        self.assertEqual(values, [calibration.deterministic_split("a.gz", index) for index in range(100)])
        self.assertIn("fit", values)
        self.assertIn("holdout", values)

    def test_sample_and_split_hashes_are_independent(self) -> None:
        splits = set()
        for index in range(100_000):
            source = "a.gz"
            selected = int.from_bytes(
                calibration.hashlib.sha256(f"sample\0{source}\0{index}".encode()).digest()[:8], "big"
            ) % 100 == 0
            if selected:
                splits.add(calibration.deterministic_split(source, index))
        self.assertEqual({"fit", "holdout"}, splits)

    def test_fit_requires_independent_improvement(self) -> None:
        artifact = calibration.fit_artifact(
            pairs(), bins=16, min_fit_pairs=100, min_holdout_pairs=20,
            min_improvement=0.20, max_slope_error=0.10,
        )
        self.assertTrue(artifact["valid"])
        self.assertTrue(artifact["holdout"]["passed"])
        self.assertEqual(0, calibration.apply_anchors(0, artifact["anchors"]))
        self.assertLess(calibration.apply_anchors(-400, artifact["anchors"]), 0)
        calibration.validate_artifact(artifact)
        self.assertEqual("net-a", artifact["reference_target"]["net_sha256"])

    def test_fit_rejects_mixed_reference_nets(self) -> None:
        rows = pairs()
        rows[-1]["reference_net_sha256"] = "net-b"
        with self.assertRaisesRegex(ValueError, "exactly one reference net"):
            calibration.fit_artifact(rows, bins=16, min_fit_pairs=100,
                                     min_holdout_pairs=20, min_improvement=0.2,
                                     max_slope_error=0.1)

    def test_fit_rejects_zero_target_fallback(self) -> None:
        rows = pairs()
        for row in rows:
            row["target_score"] = 0
        with self.assertRaisesRegex(ValueError, "all static target scores are zero"):
            calibration.fit_artifact(rows, bins=16, min_fit_pairs=100,
                                     min_holdout_pairs=20, min_improvement=0.2,
                                     max_slope_error=0.1)

    def test_invalid_holdout_cannot_be_loaded(self) -> None:
        artifact = {
            "schema": calibration.SCHEMA, "valid": False,
            "anchors": [[0, 0], [10, 10]], "holdout": {"passed": False},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not valid"):
                calibration.load(path)

    def test_nonmonotone_artifact_is_rejected(self) -> None:
        artifact = {
            "schema": calibration.SCHEMA, "valid": True,
            "anchors": [[0, 0], [100, 50], [200, 40]], "holdout": {"passed": True},
        }
        with self.assertRaisesRegex(ValueError, "not monotone"):
            calibration.validate_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
