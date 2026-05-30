#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import build  # noqa: E402


class BuildConfigTests(unittest.TestCase):
    def test_engine_static_validation_includes_source_breakdown(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "create": {
                    "backend": "bullet",
                    "engine_static_jsonl": "score/labeled.jsonl",
                    "engine_static_rows": 10,
                }
            }, handle)
            path = handle.name
        try:
            defaults = build.load_create_arg_defaults(path)
            parser = build.build_parser(defaults)
            args = parser.parse_args(["create", "-c", path, "--dry-run"])
            config = build.create_config(args)
            step = next(
                step for step in config["steps"]
                if step["name"] == "validate_engine_static"
            )
            self.assertIn("--sources", step["command"])
        finally:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
