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

    def test_bullet_can_generate_selfplay_source_and_gate_provenance(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "create": {
                    "backend": "bullet",
                    "bullet_generate_source": True,
                    "bullet_source_jsonl": "",
                    "bullet_data": "",
                    "bullet_limit": 1000,
                    "bullet_static_data": "",
                    "engine_static_rows": 10,
                    "require_clean_enyo_owned": True,
                }
            }, handle)
            path = handle.name
        try:
            defaults = build.load_create_arg_defaults(path)
            parser = build.build_parser(defaults)
            args = parser.parse_args(["create", "-c", path, "--dry-run"])
            config = build.create_config(args)
            names = [step["name"] for step in config["steps"]]

            self.assertLess(names.index("score_merge"), names.index("bullet_text"))
            self.assertLess(names.index("bullet_train"), names.index("validate_provenance"))

            bullet_text = next(
                step for step in config["steps"] if step["name"] == "bullet_text"
            )
            self.assertIn("{score}/labeled.jsonl", bullet_text["command"])

            engine_static = next(
                step for step in config["steps"]
                if step["name"] == "validate_engine_static"
            )
            self.assertIn("{score}/labeled.jsonl", engine_static["command"])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_empty_nnue_file_is_not_passed_to_selfplay(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "create": {
                    "backend": "bullet",
                    "bullet_generate_source": True,
                    "bullet_source_jsonl": "",
                    "bullet_data": "",
                    "nnue_file": "",
                    "engine_static_rows": 10,
                }
            }, handle)
            path = handle.name
        try:
            defaults = build.load_create_arg_defaults(path)
            parser = build.build_parser(defaults)
            args = parser.parse_args(["create", "-c", path, "--dry-run"])
            config = build.create_config(args)
            selfplay = next(
                step for step in config["steps"]
                if step["name"] == "posgen_selfplay"
            )
            self.assertNotIn("--nnue-file", selfplay["command"])
        finally:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
