#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from lib import bullet_format  # noqa: E402

VALIDATOR_PATH = REPO / "tools" / "validate" / "validate_bullet_results.py"
SPEC = importlib.util.spec_from_file_location("validate_bullet_results", VALIDATOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {VALIDATOR_PATH}")
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def row(result: str) -> dict:
    return {
        "fen": "7k/8/8/8/8/8/P7/K7 w - - 0 1",
        "score": 0,
        "result": result,
    }


class ValidateBulletResultsTests(unittest.TestCase):
    def write_shard(self, path: Path, results: list[str]) -> None:
        with path.open("wb") as handle:
            for result in results:
                bullet_format.write_row(handle, row(result), enyo_runtime_target=False)

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
