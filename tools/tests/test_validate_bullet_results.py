#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import warnings


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from lib import bullet_format  # noqa: E402

VALIDATOR_PATH = REPO / "tools" / "validate" / "validate_bullet_results.py"
SPEC = importlib.util.spec_from_file_location("validate_bullet_results", VALIDATOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {VALIDATOR_PATH}")
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def row(result: str, score: int = 0) -> dict:
    return {
        "fen": "7k/8/8/8/8/8/P7/K7 w - - 0 1",
        "score": score,
        "result": result,
    }


class ValidateBulletResultsTests(unittest.TestCase):
    def write_shard(self, path: Path, results: list[str], *, scores: list[int] | None = None) -> None:
        scores = scores if scores is not None else [0] * len(results)
        with path.open("wb") as handle:
            for result, score in zip(results, scores):
                bullet_format.write_row(handle, row(result, score), enyo_runtime_target=False)

    def test_validates_and_merges_all_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_shard(root / "chunk-0000.bullet", ["0-1", "1/2-1/2"])
            self.write_shard(root / "chunk-0001.bullet", ["1-0"])
            output = root / "merged.bullet"

            result = validator.validate_and_merge(
                validator.bullet_paths(root),
                merge_output=output,
                require_win_loss=True,
            )

            self.assertEqual(96, result["bytes"])
            self.assertEqual(3, result["records"])
            self.assertEqual({"loss": 1, "draw": 1, "win": 1}, result["result_counts"])
            self.assertEqual(96, output.stat().st_size)

    def test_rejects_synthetic_all_draw_corpus_without_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "draws.bullet"
            output = root / "merged.bullet"
            self.write_shard(source, ["1/2-1/2", "1/2-1/2"])

            with self.assertRaisesRegex(ValueError, "missing decisive outcome"):
                validator.validate_and_merge(
                    [source], merge_output=output, require_win_loss=True,
                )

            self.assertFalse(output.exists())
            self.assertEqual([], list(root.glob("*.partial.*")))

    def test_reports_score_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_shard(
                root / "chunk-0000.bullet",
                ["0-1", "1/2-1/2", "1-0"],
                scores=[-300, 0, 250],
            )

            result = validator.validate_and_merge(validator.bullet_paths(root))

            self.assertEqual(-300, result["score_min"])
            self.assertEqual(250, result["score_max"])
            self.assertAlmostEqual(-50 / 3, result["score_mean"])
            self.assertEqual(
                sum(result["score_histogram"]["counts"]),
                result["records"],
            )

    def test_score_sum_does_not_overflow_int16_accumulation(self) -> None:
        # Regression: a real run against ~90k records tripped a numpy
        # RuntimeWarning (silent int16 wraparound) because Python's builtin
        # sum() over an int16 ndarray keeps numpy's narrow accumulator type
        # instead of promoting to a wide one.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = ["1-0" if i % 2 == 0 else "0-1" for i in range(10000)]
            scores = [30000 if i % 2 == 0 else -30000 for i in range(10000)]
            self.write_shard(root / "chunk-0000.bullet", results, scores=scores)

            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                result = validator.validate_and_merge(validator.bullet_paths(root))

            self.assertEqual(0.0, result["score_mean"])

    def test_rejects_degenerate_score_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_shard(
                root / "chunk-0000.bullet",
                ["0-1", "1-0"],
                scores=[5, 5],
            )

            with self.assertRaisesRegex(ValueError, "degenerate score distribution"):
                validator.validate_and_merge(
                    validator.bullet_paths(root), min_score_spread=10,
                )

    def test_accepts_non_degenerate_score_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_shard(
                root / "chunk-0000.bullet",
                ["0-1", "1-0"],
                scores=[-50, 50],
            )

            result = validator.validate_and_merge(
                validator.bullet_paths(root), min_score_spread=10,
            )
            self.assertEqual(100, result["score_max"] - result["score_min"])
