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
                    "nnue_file": "/repo/enyo/net/clean-owned-source.nn",
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

    def test_clean_owned_generation_rejects_empty_nnue_file(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "create": {
                    "backend": "bullet",
                    "bullet_generate_source": True,
                    "nnue_file": "",
                    "require_clean_enyo_owned": True,
                }
            }, handle)
            path = handle.name
        try:
            defaults = build.load_create_arg_defaults(path)
            parser = build.build_parser(defaults)
            args = parser.parse_args(["create", "-c", path, "--dry-run"])
            with self.assertRaises(SystemExit) as ctx:
                build.create_config(args)
            self.assertIn("embedded default evaluator", str(ctx.exception))
        finally:
            Path(path).unlink(missing_ok=True)

    def test_clean_owned_generation_rejects_default_net(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "create": {
                    "backend": "bullet",
                    "bullet_generate_source": True,
                    "nnue_file": "/repo/enyo/net/default.net",
                    "require_clean_enyo_owned": True,
                }
            }, handle)
            path = handle.name
        try:
            defaults = build.load_create_arg_defaults(path)
            parser = build.build_parser(defaults)
            args = parser.parse_args(["create", "-c", path, "--dry-run"])
            with self.assertRaises(SystemExit) as ctx:
                build.create_config(args)
            self.assertIn("default.net", str(ctx.exception))
        finally:
            Path(path).unlink(missing_ok=True)

    def test_clean_owned_generation_can_use_hce_selfplay(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "create": {
                    "backend": "bullet",
                    "bullet_generate_source": True,
                    "bullet_source_jsonl": "",
                    "bullet_data": "",
                    "nnue_file": "",
                    "selfplay_use_nnue": False,
                    "require_clean_enyo_owned": True,
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
            self.assertIn("use_nnue=false", selfplay["command"])
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

    def test_score_limit_is_forwarded_to_all_score_shards(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "create": {
                    "backend": "bullet",
                    "bullet_generate_source": True,
                    "bullet_source_jsonl": "",
                    "bullet_data": "",
                    "nnue_file": "/repo/enyo/net/clean-owned-source.nn",
                    "score_limit": 12345,
                    "score_shards": 3,
                }
            }, handle)
            path = handle.name
        try:
            defaults = build.load_create_arg_defaults(path)
            parser = build.build_parser(defaults)
            args = parser.parse_args(["create", "-c", path, "--dry-run"])
            config = build.create_config(args)
            score_steps = [
                step for step in config["steps"]
                if str(step["name"]).startswith("score_")
                and step["name"] != "score_merge"
            ]
            self.assertEqual(3, len(score_steps))
            for step in score_steps:
                self.assertIn("--limit", step["command"])
                limit_index = step["command"].index("--limit")
                self.assertEqual("12345", step["command"][limit_index + 1])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_score_existing_source_jsonl_feeds_bullet_training(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "create": {
                    "backend": "bullet",
                    "bullet_generate_source": False,
                    "score_source_jsonl": "/runs/source/source.jsonl",
                    "bullet_source_jsonl": "",
                    "bullet_data": "",
                    "score_shards": 2,
                    "score_limit": 1000,
                    "engine_static_rows": 10,
                }
            }, handle)
            path = handle.name
        try:
            defaults = build.load_create_arg_defaults(path)
            parser = build.build_parser(defaults)
            args = parser.parse_args(["create", "-c", path, "--dry-run"])
            config = build.create_config(args)
            score_steps = [
                step for step in config["steps"]
                if str(step["name"]).startswith("score_")
                and step["name"] != "score_merge"
            ]
            self.assertEqual(2, len(score_steps))
            for step in score_steps:
                input_index = step["command"].index("--input")
                self.assertEqual(
                    "/runs/source/source.jsonl",
                    step["command"][input_index + 1],
                )
                limit_index = step["command"].index("--limit")
                self.assertEqual("1000", step["command"][limit_index + 1])

            bullet_text = next(
                step for step in config["steps"]
                if step["name"] == "bullet_text"
            )
            input_index = bullet_text["command"].index("--input")
            self.assertEqual("{score}/labeled.jsonl", bullet_text["command"][input_index + 1])

            engine_static = next(
                step for step in config["steps"]
                if step["name"] == "validate_engine_static"
            )
            jsonl_index = engine_static["command"].index("--jsonl")
            self.assertEqual("{score}/labeled.jsonl", engine_static["command"][jsonl_index + 1])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_distributed_score_plan_feeds_bullet_training(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "create": {
                    "backend": "bullet",
                    "bullet_generate_source": True,
                    "bullet_source_jsonl": "",
                    "bullet_data": "",
                    "score_distrib": True,
                    "python": "/score/python",
                    "score_distrib_python": "/coord/python",
                    "score_distrib_local_slots": 2,
                    "score_distrib_require_notify": False,
                    "score_distrib_path_map": [
                        "localhost:/home/petter/code/cpp/chess=/Users/pwahlman/code/cpp/chess",
                    ],
                    "score_shards": 4,
                    "score_limit": 123,
                    "engine_static_rows": 10,
                }
            }, handle)
            path = handle.name
        try:
            defaults = build.load_create_arg_defaults(path)
            parser = build.build_parser(defaults)
            args = parser.parse_args(["create", "-c", path, "--dry-run"])
            config = build.create_config(args)
            names = [step["name"] for step in config["steps"]]

            self.assertIn("score_distrib_plan", names)
            self.assertIn("score_distrib_doctor", names)
            self.assertIn("score_distrib_work_00", names)
            self.assertIn("score_distrib_work_01", names)
            self.assertIn("score_distrib_merge", names)
            self.assertLess(names.index("score_distrib_merge"), names.index("bullet_text"))

            plan = next(
                step for step in config["steps"]
                if step["name"] == "score_distrib_plan"
            )
            self.assertEqual("/coord/python", plan["command"][0])
            self.assertIn("--path-map", plan["command"])
            self.assertIn(
                "localhost:/home/petter/code/cpp/chess=/Users/pwahlman/code/cpp/chess",
                plan["command"],
            )
            template = plan["command"][plan["command"].index("--command-template") + 1]
            self.assertTrue(template.startswith("/score/python "))
            self.assertIn("--input '{{source}}'", template)
            self.assertIn("--output '{{output}}'", template)
            self.assertIn("--shard-count '{{shards}}'", template)
            self.assertIn("--shard-index '{{index}}'", template)
            self.assertIn("--limit 123", template)

            workers = [
                step for step in config["steps"]
                if str(step["name"]).startswith("score_distrib_work_")
            ]
            self.assertEqual(2, len(workers))
            for step in workers:
                self.assertEqual("score_distrib_work", step["parallel_group"])
                self.assertEqual("/coord/python", step["command"][0])

            bullet_text = next(
                step for step in config["steps"]
                if step["name"] == "bullet_text"
            )
            input_index = bullet_text["command"].index("--input")
            self.assertEqual("{score}/labeled.jsonl", bullet_text["command"][input_index + 1])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_distributed_score_requires_local_worker_slot(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "create": {
                    "backend": "bullet",
                    "bullet_generate_source": True,
                    "score_distrib": True,
                    "score_distrib_local_slots": 0,
                }
            }, handle)
            path = handle.name
        try:
            defaults = build.load_create_arg_defaults(path)
            parser = build.build_parser(defaults)
            args = parser.parse_args(["create", "-c", path, "--dry-run"])
            with self.assertRaises(SystemExit) as ctx:
                build.create_config(args)
            self.assertIn("score_distrib_local_slots", str(ctx.exception))
        finally:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
