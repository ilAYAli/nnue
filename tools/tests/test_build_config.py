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
    def test_disabled_create_config_fails_before_default_build(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "create": {
                    "disabled": True,
                    "disabled_reason": "sidecar preflight only",
                }
            }, handle)
            path = handle.name
        try:
            defaults = build.load_create_arg_defaults(path)
            parser = build.build_parser(defaults)
            args = parser.parse_args(["create", "-c", path, "--dry-run"])
            with self.assertRaises(SystemExit) as ctx:
                build.create_config(args)
            self.assertIn("sidecar preflight only", str(ctx.exception))
        finally:
            Path(path).unlink(missing_ok=True)

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

    def test_bullet_can_train_from_existing_source_jsonl(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "create": {
                    "backend": "bullet",
                    "bullet_generate_source": False,
                    "bullet_source_jsonl": "runs/rescued/labeled.jsonl",
                    "bullet_data": "",
                    "bullet_limit": 1000,
                    "bullet_static_data": "",
                    "engine_static_jsonl": "runs/rescued/labeled.jsonl",
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

            self.assertNotIn("selfplay_crucible_plan", names)
            self.assertNotIn("score_crucible_plan", names)
            self.assertNotIn("score_merge", names)
            self.assertIn("bullet_text", names)
            self.assertIn("validate_engine_static", names)

            bullet_text = next(
                step for step in config["steps"] if step["name"] == "bullet_text"
            )
            input_index = bullet_text["command"].index("--input")
            self.assertTrue(
                bullet_text["command"][input_index + 1].endswith(
                    "/runs/rescued/labeled.jsonl"
                )
            )
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

    def test_bullet_score_can_emit_bullet_text_directly(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "create": {
                    "backend": "bullet",
                    "bullet_generate_source": False,
                    "score_source_jsonl": "/runs/source/source.jsonl",
                    "score_output_format": "bullet-text",
                    "bullet_source_jsonl": "",
                    "bullet_data": "",
                    "bullet_static_data": "",
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
            names = [step["name"] for step in config["steps"]]

            self.assertIn("score_merge", names)
            self.assertNotIn("bullet_text", names)
            self.assertNotIn("validate_engine_static", names)
            score_steps = [
                step for step in config["steps"]
                if str(step["name"]).startswith("score_")
                and step["name"] != "score_merge"
            ]
            self.assertEqual(2, len(score_steps))
            for step in score_steps:
                output_index = step["command"].index("--output")
                self.assertTrue(step["command"][output_index + 1].endswith(".txt"))
                self.assertIn("--output-format", step["command"])
                self.assertEqual(
                    "bullet-text",
                    step["command"][step["command"].index("--output-format") + 1],
                )
                self.assertIn("--enyo-runtime-target", step["command"])

            merge = next(step for step in config["steps"] if step["name"] == "score_merge")
            self.assertEqual("{score}/labeled.txt", merge["command"][6])

            bullet_format = next(
                step for step in config["steps"]
                if step["name"] == "bullet_format"
            )
            input_index = bullet_format["command"].index("--input")
            self.assertEqual("{score}/labeled.txt", bullet_format["command"][input_index + 1])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_bullet_score_can_emit_bullet_data_directly(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "create": {
                    "backend": "bullet",
                    "bullet_generate_source": False,
                    "score_source_jsonl": "/runs/source/source.jsonl",
                    "score_output_format": "bullet-data",
                    "bullet_source_jsonl": "",
                    "bullet_data": "",
                    "bullet_static_data": "",
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
            names = [step["name"] for step in config["steps"]]

            self.assertIn("score_merge", names)
            self.assertNotIn("bullet_text", names)
            self.assertNotIn("bullet_format", names)
            self.assertNotIn("validate_engine_static", names)
            score_steps = [
                step for step in config["steps"]
                if str(step["name"]).startswith("score_")
                and step["name"] != "score_merge"
            ]
            self.assertEqual(2, len(score_steps))
            for step in score_steps:
                output_index = step["command"].index("--output")
                self.assertTrue(step["command"][output_index + 1].endswith(".bullet"))
                self.assertIn("--output-format", step["command"])
                self.assertEqual(
                    "bullet-data",
                    step["command"][step["command"].index("--output-format") + 1],
                )
                self.assertIn("--enyo-runtime-target", step["command"])

            merge = next(step for step in config["steps"] if step["name"] == "score_merge")
            self.assertEqual("{score}/labeled.bullet", merge["command"][6])

            train = next(step for step in config["steps"] if step["name"] == "bullet_train")
            data_index = train["command"].index("--data")
            self.assertEqual("{score}/labeled.bullet", train["command"][data_index + 1])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_stockfish_static_score_backend_emits_bullet_data(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "create": {
                    "backend": "bullet",
                    "bullet_generate_source": False,
                    "score_source_jsonl": "/runs/source/source.jsonl",
                    "score_backend": "stockfish-static",
                    "score_static_datagen_tool": "/enyo/datagen",
                    "score_stockfish_net": "/sf/nn.nnue",
                    "score_enyo_net": "/enyo/model.nn",
                    "score_min_delta_cp": 75,
                    "score_output_format": "bullet-data",
                    "bullet_source_jsonl": "",
                    "bullet_data": "",
                    "bullet_static_data": "",
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
                command = step["command"]
                self.assertEqual("/enyo/datagen", command[0])
                self.assertIn("--stockfish-net", command)
                self.assertEqual("/sf/nn.nnue", command[command.index("--stockfish-net") + 1])
                self.assertIn("--enyo-net", command)
                self.assertEqual("/enyo/model.nn", command[command.index("--enyo-net") + 1])
                self.assertIn("--min-delta-cp", command)
                self.assertEqual("75", command[command.index("--min-delta-cp") + 1])
                self.assertIn("--shard-count", command)
                self.assertIn("--shard-index", command)
                self.assertNotIn("--output-format", command)

            merge = next(step for step in config["steps"] if step["name"] == "score_merge")
            self.assertEqual("{score}/labeled.bullet", merge["command"][6])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_pytorch_can_train_from_existing_labeled_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labeled = root / "labeled.jsonl"
            labeled.write_text("", encoding="utf-8")
            labeled.with_suffix(".wc").write_text("0 labeled.jsonl\n", encoding="utf-8")
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "create": {
                    "backend": "pytorch",
                    "labeled_jsonl": str(labeled),
                    "epochs": 1,
                    "engine_static_rows": 10,
                }
            }), encoding="utf-8")

            defaults = build.load_create_arg_defaults(str(config_path))
            parser = build.build_parser(defaults)
            args = parser.parse_args(["create", "-c", str(config_path), "--dry-run"])
            config = build.create_config(args)
            names = [step["name"] for step in config["steps"]]

            self.assertEqual("pack", names[0])
            self.assertNotIn("posgen_selfplay", names)
            self.assertNotIn("score_merge", names)

            pack = next(step for step in config["steps"] if step["name"] == "pack")
            self.assertEqual(
                str(labeled.resolve()),
                pack["command"][pack["command"].index("--input") + 1],
            )
            self.assertEqual(
                str(labeled.with_suffix(".wc").resolve()),
                pack["command"][pack["command"].index("--rows-file") + 1],
            )

            engine_static = next(
                step for step in config["steps"]
                if step["name"] == "validate_engine_static"
            )
            self.assertEqual(
                str(labeled.resolve()),
                engine_static["command"][engine_static["command"].index("--jsonl") + 1],
            )

    def test_pytorch_generated_source_can_merge_packed_score_shards(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "create": {
                    "backend": "pytorch",
                    "score_output_format": "packed",
                    "score_shards": 2,
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

            self.assertNotIn("score_merge", names)
            self.assertIn("pack", names)
            score_steps = [
                step for step in config["steps"]
                if str(step["name"]).startswith("score_")
            ]
            self.assertEqual(2, len(score_steps))
            for step in score_steps:
                output_index = step["command"].index("--output")
                self.assertTrue(step["command"][output_index + 1].endswith(".npz"))
                self.assertIn("--output-format", step["command"])
                self.assertEqual(
                    "packed",
                    step["command"][step["command"].index("--output-format") + 1],
                )

            pack = next(step for step in config["steps"] if step["name"] == "pack")
            self.assertIn("merge-shards", pack["command"])
            self.assertIn("--input-dir", pack["command"])
            self.assertIn("{score}/shards", pack["command"])
            self.assertLess(names.index("score_01"), names.index("pack"))
            self.assertLess(names.index("pack"), names.index("train"))
        finally:
            Path(path).unlink(missing_ok=True)

    def test_score_output_formats_are_backend_specific(self) -> None:
        parser = build.build_parser()
        args = parser.parse_args([
            "create",
            "--dry-run",
            "--backend", "pytorch",
            "--score-output-format", "bullet-text",
        ])
        with self.assertRaises(SystemExit):
            build.create_config(args)

        args = parser.parse_args([
            "create",
            "--dry-run",
            "--backend", "pytorch",
            "--score-output-format", "bullet-data",
        ])
        with self.assertRaises(SystemExit):
            build.create_config(args)

        args = parser.parse_args([
            "create",
            "--dry-run",
            "--backend", "bullet",
            "--score-output-format", "packed",
        ])
        with self.assertRaises(SystemExit):
            build.create_config(args)

    def test_pytorch_source_mix_feeds_pack_and_static_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broad = root / "broad.jsonl"
            rescue = root / "rescue.jsonl"
            broad.write_text("", encoding="utf-8")
            rescue.write_text("", encoding="utf-8")
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "create": {
                    "backend": "pytorch",
                    "source_mix_jsonl": [
                        f"{broad}:1000:800",
                        f"{rescue}:500:800",
                    ],
                    "source_mix_seed": 2026060601,
                    "source_mix_progress": 77,
                    "engine_static_rows": 10,
                    "require_clean_enyo_owned": True,
                }
            }), encoding="utf-8")

            defaults = build.load_create_arg_defaults(str(config_path))
            parser = build.build_parser(defaults)
            args = parser.parse_args(["create", "-c", str(config_path), "--dry-run"])
            config = build.create_config(args)
            names = [step["name"] for step in config["steps"]]

            self.assertEqual("source_mix", names[0])
            self.assertLess(names.index("source_mix"), names.index("pack"))
            self.assertLess(names.index("train"), names.index("validate_provenance"))
            self.assertLess(
                names.index("validate_provenance"),
                names.index("validate_engine_static"),
            )

            train = next(step for step in config["steps"] if step["name"] == "train")
            self.assertTrue(train["command"][0].endswith("nnue-train"))
            self.assertEqual("supervised", train["command"][1])

            source_mix = next(
                step for step in config["steps"]
                if step["name"] == "source_mix"
            )
            self.assertIn(f"{broad.resolve()}:1000:800", source_mix["command"])
            self.assertIn(f"{rescue.resolve()}:500:800", source_mix["command"])
            self.assertEqual(
                "2026060601",
                source_mix["command"][source_mix["command"].index("--seed") + 1],
            )
            self.assertEqual(
                "77",
                source_mix["command"][source_mix["command"].index("--progress") + 1],
            )

            pack = next(step for step in config["steps"] if step["name"] == "pack")
            self.assertEqual(
                "{score}/mixed.jsonl",
                pack["command"][pack["command"].index("--input") + 1],
            )

            engine_static = next(
                step for step in config["steps"]
                if step["name"] == "validate_engine_static"
            )
            self.assertEqual(
                "{score}/mixed.jsonl",
                engine_static["command"][engine_static["command"].index("--jsonl") + 1],
            )

    def test_source_mix_feeds_bullet_training(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "create": {
                    "backend": "bullet",
                    "source_mix_jsonl": [
                        "/runs/broad/labeled.jsonl:200000:800",
                        "/runs/replay/pairs.jsonl:4000",
                    ],
                    "source_mix_seed": 2026060302,
                    "source_mix_progress": 123,
                    "bullet_source_jsonl": "",
                    "bullet_data": "",
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

            self.assertLess(names.index("source_mix"), names.index("bullet_text"))

            source_mix = next(
                step for step in config["steps"]
                if step["name"] == "source_mix"
            )
            self.assertIn("--source", source_mix["command"])
            self.assertIn("/runs/broad/labeled.jsonl:200000:800", source_mix["command"])
            self.assertIn("/runs/replay/pairs.jsonl:4000", source_mix["command"])
            self.assertEqual(
                "2026060302",
                source_mix["command"][source_mix["command"].index("--seed") + 1],
            )
            self.assertEqual(
                "123",
                source_mix["command"][source_mix["command"].index("--progress") + 1],
            )

            bullet_text = next(
                step for step in config["steps"]
                if step["name"] == "bullet_text"
            )
            input_index = bullet_text["command"].index("--input")
            self.assertEqual("{score}/mixed.jsonl", bullet_text["command"][input_index + 1])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_pairwise_backend_packs_trains_and_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labeled = root / "broad.jsonl"
            pairs = root / "pairs.jsonl"
            gate = root / "gate.jsonl"
            baseline = root / "baseline.nn"
            labeled.write_text("", encoding="utf-8")
            labeled.with_suffix(".wc").write_text("0 broad.jsonl\n", encoding="utf-8")
            pairs.write_text("", encoding="utf-8")
            gate.write_text("", encoding="utf-8")
            baseline.write_text("", encoding="utf-8")
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "create": {
                    "backend": "pairwise",
                    "labeled_jsonl": str(labeled),
                    "pairwise_pairs_jsonl": str(pairs),
                    "pairwise_init_from_nn": str(baseline),
                    "pairwise_pair_weight": 3.0,
                    "pairwise_checkpoint_every": 2,
                    "pairwise_trainable": "l2-output",
                    "pairwise_project_export_weights_each_step": True,
                    "pairwise_move_gate_cases": str(gate),
                    "engine_static_jsonl": str(labeled),
                    "engine_static_rows": 10,
                    "require_clean_enyo_owned": True,
                }
            }), encoding="utf-8")

            defaults = build.load_create_arg_defaults(str(config_path))
            parser = build.build_parser(defaults)
            args = parser.parse_args(["create", "-c", str(config_path), "--dry-run"])
            config = build.create_config(args)
            names = [step["name"] for step in config["steps"]]

            self.assertEqual("pack", names[0])
            self.assertLess(names.index("train_pairwise"), names.index("validate_provenance"))
            self.assertLess(
                names.index("validate_provenance"),
                names.index("validate_engine_static_baseline"),
            )
            self.assertLess(
                names.index("validate_engine_static_baseline"),
                names.index("validate_engine_static"),
            )
            self.assertLess(names.index("validate_engine_static"), names.index("validate_move_gate"))

            train = next(step for step in config["steps"] if step["name"] == "train_pairwise")
            self.assertTrue(train["command"][0].endswith("nnue-train"))
            self.assertEqual("pairwise", train["command"][1])
            self.assertIn("--checkpoint-dir", train["command"])
            self.assertEqual(
                "2",
                train["command"][train["command"].index("--checkpoint-every") + 1],
            )
            self.assertEqual(
                "3.0",
                train["command"][train["command"].index("--pair-weight") + 1],
            )
            self.assertEqual(
                "l2-output",
                train["command"][train["command"].index("--trainable") + 1],
            )
            self.assertIn("--project-export-weights-each-step", train["command"])

            baseline_static = next(
                step for step in config["steps"]
                if step["name"] == "validate_engine_static_baseline"
            )
            self.assertEqual(
                str(baseline.resolve()),
                baseline_static["command"][baseline_static["command"].index("--net") + 1],
            )

            move_gate = next(
                step for step in config["steps"]
                if step["name"] == "validate_move_gate"
            )
            self.assertIn("--fail-if-candidate-below-baseline", move_gate["command"])
            self.assertEqual(
                str(gate.resolve()),
                move_gate["command"][move_gate["command"].index("--cases") + 1],
            )

    def test_public_provenance_gate_does_not_require_clean_owned(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "create": {
                    "backend": "bullet",
                    "bullet_source_jsonl": "/runs/source/labeled.jsonl",
                    "validate_provenance": True,
                    "require_clean_enyo_owned": False,
                    "engine_static_rows": 10,
                }
            }, handle)
            path = handle.name
        try:
            defaults = build.load_create_arg_defaults(path)
            parser = build.build_parser(defaults)
            args = parser.parse_args(["create", "-c", path, "--dry-run"])
            config = build.create_config(args)
            provenance = next(
                step for step in config["steps"]
                if step["name"] == "validate_provenance"
            )
            self.assertNotIn("--require-clean-enyo-owned", provenance["command"])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_bullet_init_net_converts_to_layout_weights(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "create": {
                    "backend": "bullet",
                    "bullet_source_jsonl": "/runs/source/labeled.jsonl",
                    "bullet_init_net": "/runs/native15/train/model.nn",
                    "bullet_enyo_input_buckets": 32,
                    "bullet_enyo_feature_channels": 11,
                    "bullet_enyo_runtime_input_buckets": 32,
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

            self.assertLess(names.index("bullet_init_weights"), names.index("bullet_train"))
            init_step = next(
                step for step in config["steps"]
                if step["name"] == "bullet_init_weights"
            )
            self.assertIn("--feature-channels", init_step["command"])
            self.assertEqual(
                "11",
                init_step["command"][init_step["command"].index("--feature-channels") + 1],
            )
            train = next(
                step for step in config["steps"] if step["name"] == "bullet_train"
            )
            self.assertIn("--enyo-feature-channels", train["command"])
            self.assertEqual(
                "11",
                train["command"][train["command"].index("--enyo-feature-channels") + 1],
            )
            self.assertIn("--init-weights", train["command"])
            self.assertEqual(
                "{assets}/init/optimiser_state/weights.bin",
                train["command"][train["command"].index("--init-weights") + 1],
            )
        finally:
            Path(path).unlink(missing_ok=True)

    def test_crucible_score_plan_feeds_bullet_training(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "create": {
                    "backend": "bullet",
                    "bullet_generate_source": True,
                    "bullet_source_jsonl": "",
                    "bullet_data": "",
                    "score_crucible": True,
                    "python": "/score/python",
                    "score_crucible_python": "/coord/python",
                    "score_crucible_local_slots": 2,
                    "score_crucible_require_notify": False,
                    "score_crucible_path_map": [
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

            self.assertIn("score_crucible_plan", names)
            self.assertIn("score_crucible_add_input", names)
            self.assertIn("score_crucible_doctor", names)
            self.assertIn("score_crucible_work_00", names)
            self.assertIn("score_crucible_work_01", names)
            self.assertIn("score_crucible_wait", names)
            self.assertIn("score_crucible_merge", names)
            self.assertLess(names.index("score_crucible_plan"), names.index("score_crucible_add_input"))
            self.assertLess(names.index("score_crucible_add_input"), names.index("score_crucible_doctor"))
            self.assertLess(names.index("score_crucible_work_01"), names.index("score_crucible_wait"))
            self.assertLess(names.index("score_crucible_wait"), names.index("score_crucible_merge"))
            self.assertLess(names.index("score_crucible_merge"), names.index("bullet_text"))

            plan = next(
                step for step in config["steps"]
                if step["name"] == "score_crucible_plan"
            )
            self.assertEqual("/coord/python", plan["command"][0])
            self.assertEqual("label", plan["command"][plan["command"].index("--kind") + 1])
            self.assertEqual(
                "score:uci",
                plan["command"][plan["command"].index("--task-label") + 1],
            )
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
            self.assertIn("--progress-unit", plan["command"])
            self.assertEqual(
                "rows",
                plan["command"][plan["command"].index("--progress-unit") + 1],
            )
            self.assertEqual(
                "{posgen}/source.jsonl",
                plan["command"][plan["command"].index("--progress-total-lines") + 1],
            )
            progress_regex = plan["command"][plan["command"].index("--progress-log-regex") + 1]
            self.assertIn("?P<done>", progress_regex)
            self.assertIn("?P<output>", progress_regex)
            self.assertIn("?P<rate>", progress_regex)
            self.assertEqual(
                "output",
                plan["command"][plan["command"].index("--progress-output-unit") + 1],
            )

            add_input = next(
                step for step in config["steps"]
                if step["name"] == "score_crucible_add_input"
            )
            self.assertEqual("/coord/python", add_input["command"][0])
            self.assertIn("add-input", add_input["command"])
            self.assertIn("--path", add_input["command"])

            workers = [
                step for step in config["steps"]
                if str(step["name"]).startswith("score_crucible_work_")
            ]
            self.assertEqual(2, len(workers))
            for step in workers:
                self.assertEqual("score_crucible_work", step["parallel_group"])
                self.assertEqual("/coord/python", step["command"][0])

            wait = next(
                step for step in config["steps"]
                if step["name"] == "score_crucible_wait"
            )
            self.assertEqual("/coord/python", wait["command"][0])
            self.assertIn("wait", wait["command"])

            merge = next(
                step for step in config["steps"]
                if step["name"] == "score_crucible_merge"
            )
            self.assertIn('verify "$run"', merge["command"][2])
            self.assertNotIn("--manifest", merge["command"][2])
            self.assertEqual("{score}/crucible", merge["command"][6])

            bullet_text = next(
                step for step in config["steps"]
                if step["name"] == "bullet_text"
            )
            input_index = bullet_text["command"].index("--input")
            self.assertEqual("{score}/labeled.jsonl", bullet_text["command"][input_index + 1])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_crucible_score_plan_can_use_stockfish_static_backend(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "create": {
                    "backend": "bullet",
                    "bullet_generate_source": True,
                    "bullet_source_jsonl": "",
                    "bullet_data": "",
                    "score_crucible": True,
                    "score_crucible_python": "/coord/python",
                    "score_crucible_require_notify": False,
                    "score_backend": "stockfish-static",
                    "score_static_datagen_tool": "/enyo/datagen",
                    "score_stockfish_net": "/sf/nn.nnue",
                    "score_output_format": "bullet-data",
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
            plan = next(
                step for step in config["steps"]
                if step["name"] == "score_crucible_plan"
            )

            self.assertEqual(
                "score:stockfish-static",
                plan["command"][plan["command"].index("--task-label") + 1],
            )
            template = plan["command"][plan["command"].index("--command-template") + 1]
            self.assertTrue(template.startswith("/enyo/datagen "))
            self.assertIn("--stockfish-net /sf/nn.nnue", template)
            self.assertIn("--input '{{source}}'", template)
            self.assertIn("--output '{{output}}'", template)
            self.assertIn("--shard-count '{{shards}}'", template)
            self.assertIn("--shard-index '{{index}}'", template)

            merge = next(
                step for step in config["steps"]
                if step["name"] == "score_crucible_merge"
            )
            self.assertEqual("{score}/labeled.bullet", merge["command"][7])
            train = next(step for step in config["steps"] if step["name"] == "bullet_train")
            data_index = train["command"].index("--data")
            self.assertEqual("{score}/labeled.bullet", train["command"][data_index + 1])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_crucible_selfplay_plan_feeds_source_extraction(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "create": {
                    "backend": "bullet",
                    "bullet_generate_source": True,
                    "bullet_source_jsonl": "",
                    "bullet_data": "",
                    "selfplay_games": 1001,
                    "selfplay_shard_games": 250,
                    "selfplay_crucible": True,
                    "selfplay_crucible_python": "/coord/python",
                    "selfplay_crucible_local_slots": 2,
                    "selfplay_crucible_require_notify": False,
                    "selfplay_crucible_path_map": [
                        "localhost:/home/petter/code/cpp/chess=/Users/pwahlman/code/cpp/chess",
                    ],
                    "python": "/work/python",
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

            self.assertIn("selfplay_crucible_plan", names)
            self.assertIn("selfplay_crucible_add_input", names)
            self.assertIn("selfplay_crucible_doctor", names)
            self.assertIn("selfplay_crucible_work_00", names)
            self.assertIn("selfplay_crucible_work_01", names)
            self.assertIn("selfplay_crucible_wait", names)
            self.assertIn("selfplay_crucible_merge", names)
            self.assertLess(names.index("selfplay_crucible_merge"), names.index("posgen_extract"))

            plan = next(
                step for step in config["steps"]
                if step["name"] == "selfplay_crucible_plan"
            )
            self.assertEqual("/coord/python", plan["command"][0])
            self.assertEqual("selfplay", plan["command"][plan["command"].index("--kind") + 1])
            self.assertEqual(
                "selfplay",
                plan["command"][plan["command"].index("--task-label") + 1],
            )
            self.assertIn("--shards", plan["command"])
            self.assertEqual("5", plan["command"][plan["command"].index("--shards") + 1])
            self.assertIn("--progress-unit", plan["command"])
            self.assertEqual(
                "games",
                plan["command"][plan["command"].index("--progress-unit") + 1],
            )
            self.assertIn("--progress-total", plan["command"])
            self.assertEqual(
                "250",
                plan["command"][plan["command"].index("--progress-total") + 1],
            )
            self.assertIn("?P<done>", plan["command"][plan["command"].index("--progress-log-regex") + 1])
            self.assertIn("?P<total>", plan["command"][plan["command"].index("--progress-log-regex") + 1])
            self.assertIn("--path-map", plan["command"])
            self.assertIn(
                "localhost:/home/petter/code/cpp/chess=/Users/pwahlman/code/cpp/chess",
                plan["command"],
            )
            template = plan["command"][plan["command"].index("--command-template") + 1]
            self.assertTrue(template.startswith("/work/python "))
            self.assertIn("posgen/selfplay_shards.py generate", template)
            self.assertIn("--total-games '{{total_games}}'", template)
            self.assertIn("--shards '{{shards}}'", template)
            self.assertIn("--shard-index '{{index}}'", template)
            self.assertIn("--output-pgn '{{output}}'", template)
            self.assertNotIn("{{pgn}}", template)
            self.assertNotIn("--metadata", template)
            outputs = [
                plan["command"][index + 1]
                for index, item in enumerate(plan["command"])
                if item == "--output-template"
            ]
            self.assertIn("{posgen}/selfplay_shards/shard.{{index}}.pgn", outputs)

            merge = next(
                step for step in config["steps"]
                if step["name"] == "selfplay_crucible_merge"
            )
            self.assertIn('verify "$run"', merge["command"][2])
            self.assertNotIn("verify --manifest", merge["command"][2])
            self.assertEqual("{posgen}/selfplay_crucible", merge["command"][6])
            self.assertTrue(any("selfplay_shards.py" in item for item in merge["command"]))
            formatted_merge = [
                part.format(posgen="/tmp/posgen")
                for part in merge["command"]
            ]
            self.assertIn("merge-pgns", formatted_merge[2])
            self.assertIn('"$posgen"/selfplay_shards/shard.*.pgn', formatted_merge[2])
            self.assertIn("${#pgns[@]}", formatted_merge[2])
            self.assertIn("${pgns[@]}", formatted_merge[2])
            self.assertIn("--expected-games", formatted_merge[2])
            self.assertIn("1001", merge["command"])
            self.assertNotIn("*.meta.json", formatted_merge[2])

            extract = next(
                step for step in config["steps"]
                if step["name"] == "posgen_extract"
            )
            self.assertIn("{posgen}/selfplay.pgn", extract["command"])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_crucible_score_accepts_cli_path_maps(self) -> None:
        parser = build.build_parser()
        args = parser.parse_args([
            "create",
            "--dry-run",
            "--backend", "bullet",
            "--bullet-generate-source",
            "--score-crucible",
            "--score-crucible-local-slots", "1",
            "--no-score-crucible-require-notify",
            "--score-crucible-path-map", "localhost:/coord=/local",
            "--score-crucible-path-map", "pwa-wsl:/coord=/wsl",
            "--engine-static-rows", "10",
        ])

        config = build.create_config(args)
        plan = next(
            step for step in config["steps"]
            if step["name"] == "score_crucible_plan"
        )

        self.assertIn("localhost:/coord=/local", plan["command"])
        self.assertIn("pwa-wsl:/coord=/wsl", plan["command"])

    def test_crucible_score_can_wait_for_external_workers_only(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "create": {
                    "backend": "bullet",
                    "bullet_generate_source": True,
                    "score_crucible": True,
                    "score_crucible_local_slots": 0,
                }
            }, handle)
            path = handle.name
        try:
            defaults = build.load_create_arg_defaults(path)
            parser = build.build_parser(defaults)
            args = parser.parse_args(["create", "-c", path, "--dry-run"])
            config = build.create_config(args)
            names = [step["name"] for step in config["steps"]]

            self.assertNotIn("score_crucible_work_00", names)
            self.assertIn("score_crucible_wait", names)
            self.assertLess(names.index("score_crucible_wait"), names.index("score_crucible_merge"))
        finally:
            Path(path).unlink(missing_ok=True)

    def test_crucible_score_deploys_workers_file(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "create": {
                    "backend": "bullet",
                    "bullet_generate_source": True,
                    "score_crucible": True,
                    "score_crucible_workers": "~/workers.json",
                    "score_crucible_jobs": 3,
                    "score_crucible_coordinator_host": "coordinator.example.net",
                    "score_crucible_remote_timeout_seconds": 42,
                    "score_crucible_verbose": True,
                    "score_crucible_local_slots": 2,
                    "score_crucible_require_notify": False,
                }
            }, handle)
            path = handle.name
        try:
            defaults = build.load_create_arg_defaults(path)
            parser = build.build_parser(defaults)
            args = parser.parse_args(["create", "-c", path, "--dry-run"])
            config = build.create_config(args)
            names = [step["name"] for step in config["steps"]]

            self.assertIn("score_crucible_deploy", names)
            self.assertNotIn("score_crucible_work_00", names)
            self.assertNotIn("score_crucible_wait", names)
            self.assertLess(names.index("score_crucible_deploy"), names.index("score_crucible_merge"))

            deploy = next(
                step for step in config["steps"]
                if step["name"] == "score_crucible_deploy"
            )
            self.assertEqual(["bash", "-lc"], deploy["command"][:2])
            self.assertIn("deploy", deploy["command"][2])
            self.assertIn("pass --resume or --replace", deploy["command"][2])
            self.assertIn("--resume", deploy["command"][2])
            self.assertIn("--jobs", deploy["command"][2])
            self.assertIn("--coordinator-host", deploy["command"][2])
            self.assertIn("--remote-timeout-seconds", deploy["command"][2])
            self.assertIn("--verbose", deploy["command"][2])
            self.assertIn("${cmd[@]}", deploy["command"][2].format())
            self.assertIn("${PIPESTATUS[0]}", deploy["command"][2].format())
            self.assertIn('if [ "$rc" -eq 0 ]', deploy["command"][2].format())
            self.assertNotIn('if "${cmd[@]}" 2>&1 | tee "$tmp"; then', deploy["command"][2].format())
            self.assertEqual(str(Path("~/workers.json").expanduser()), deploy["command"][6])
            self.assertEqual("3", deploy["command"][8])
            self.assertEqual("coordinator.example.net", deploy["command"][9])
            self.assertEqual("42", deploy["command"][10])
            self.assertEqual("1", deploy["command"][11])

            merge = next(
                step for step in config["steps"]
                if step["name"] == "score_crucible_merge"
            )
            self.assertEqual("score-{candidate}", merge["command"][6])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_crucible_selfplay_deploys_workers_file(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "create": {
                    "backend": "bullet",
                    "bullet_generate_source": True,
                    "bullet_source_jsonl": "",
                    "bullet_data": "",
                    "selfplay_games": 1000,
                    "selfplay_crucible": True,
                    "selfplay_crucible_workers": "~/workers.json",
                    "selfplay_crucible_jobs": 3,
                    "selfplay_crucible_coordinator_host": "coordinator.example.net",
                    "selfplay_crucible_remote_timeout_seconds": 42,
                    "selfplay_crucible_verbose": True,
                    "selfplay_crucible_local_slots": 2,
                    "selfplay_crucible_require_notify": False,
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

            self.assertIn("selfplay_crucible_deploy", names)
            self.assertNotIn("selfplay_crucible_work_00", names)
            self.assertNotIn("selfplay_crucible_wait", names)
            self.assertLess(names.index("selfplay_crucible_deploy"), names.index("selfplay_crucible_merge"))

            deploy = next(
                step for step in config["steps"]
                if step["name"] == "selfplay_crucible_deploy"
            )
            self.assertEqual(["bash", "-lc"], deploy["command"][:2])
            self.assertIn("deploy", deploy["command"][2])
            self.assertIn("pass --resume or --replace", deploy["command"][2])
            self.assertIn("--resume", deploy["command"][2])
            self.assertIn("--jobs", deploy["command"][2])
            self.assertIn("--coordinator-host", deploy["command"][2])
            self.assertIn("--remote-timeout-seconds", deploy["command"][2])
            self.assertIn("--verbose", deploy["command"][2])
            self.assertIn("${cmd[@]}", deploy["command"][2].format())
            self.assertIn("${PIPESTATUS[0]}", deploy["command"][2].format())
            self.assertEqual(str(Path("~/workers.json").expanduser()), deploy["command"][6])
            self.assertEqual("3", deploy["command"][8])
            self.assertEqual("coordinator.example.net", deploy["command"][9])
            self.assertEqual("42", deploy["command"][10])
            self.assertEqual("1", deploy["command"][11])

            merge = next(
                step for step in config["steps"]
                if step["name"] == "selfplay_crucible_merge"
            )
            self.assertEqual("selfplay-{candidate}", merge["command"][6])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_crucible_score_rejects_negative_local_worker_slots(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "create": {
                    "backend": "bullet",
                    "bullet_generate_source": True,
                    "score_crucible": True,
                    "score_crucible_local_slots": -1,
                }
            }, handle)
            path = handle.name
        try:
            defaults = build.load_create_arg_defaults(path)
            parser = build.build_parser(defaults)
            args = parser.parse_args(["create", "-c", path, "--dry-run"])
            with self.assertRaises(SystemExit) as ctx:
                build.create_config(args)
            self.assertIn("score_crucible_local_slots", str(ctx.exception))
        finally:
            Path(path).unlink(missing_ok=True)

    def test_status_run_argument_is_optional(self) -> None:
        parser = build.build_parser({})

        args = parser.parse_args(["status"])

        self.assertIsNone(args.run)
        self.assertFalse(args.verbose)

    def test_recipe_start_builds_direct_bullet_static_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            recipe_path = tmp_path / "recipe.json"
            recipe = {
                "name": "recipe-smoke",
                "desc": "direct static scoring smoke",
                "source": "runs/source/source.jsonl",
                "stockfish_net": "nets/stockfish.nnue",
                "init_net": "runs/base/model.nn",
                "workers": "runs/workers.json",
                "score": {
                    "shards": 8,
                    "jobs": 3,
                    "limit": 1000,
                },
                "train": {
                    "superbatches": 4,
                    "lr": 3e-7,
                    "batch_size": 64,
                },
            }
            recipe_path.write_text(json.dumps(recipe) + "\n", encoding="utf-8")

            defaults = build.recipe_to_create_defaults(
                build.load_recipe(recipe_path),
                recipe_path=recipe_path,
            )
            config = build.create_config(build.namespace_from_create_defaults(defaults))
            names = [step["name"] for step in config["steps"]]

            self.assertEqual("recipe-smoke", config["name"])
            self.assertIn("score_crucible_deploy", names)
            self.assertIn("score_crucible_merge", names)
            self.assertIn("bullet_train", names)
            self.assertNotIn("bullet_text", names)
            self.assertNotIn("bullet_format", names)
            train = next(step for step in config["steps"] if step["name"] == "bullet_train")
            self.assertIn("{score}/labeled.bullet", train["command"])
            static = next(
                step for step in config["steps"]
                if step["name"] == "validate_bullet_static"
            )
            self.assertIn("{score}/labeled.bullet", static["command"])

    def test_recipe_rejects_unknown_fields(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "name": "bad",
                "source": "source.jsonl",
                "stockfish_net": "sf.nnue",
                "init_net": "base.nn",
                "nonsense": True,
            }, handle)
            path = Path(handle.name)
        try:
            with self.assertRaises(SystemExit) as ctx:
                build.recipe_to_create_defaults(build.load_recipe(path), recipe_path=path)
            self.assertIn("unknown recipe field", str(ctx.exception))
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
