#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO / "tools" / "posgen" / "selfplay_shards.py"
SPEC = importlib.util.spec_from_file_location("selfplay_shards", TOOL_PATH)
assert SPEC is not None
selfplay_shards = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(selfplay_shards)


def write_pgn(path: Path, games: int) -> None:
    text = []
    for idx in range(games):
        text.append(f'[Event "Shard {idx + 1}"]\n')
        text.append('[Result "1/2-1/2"]\n\n')
        text.append("1. e4 e5 1/2-1/2\n\n")
    path.write_text("".join(text), encoding="utf-8")


def write_file(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return selfplay_shards.sha256_file(path)


class SelfplayShardTests(unittest.TestCase):
    def test_shard_game_count_spreads_remainder_to_early_shards(self) -> None:
        counts = [
            selfplay_shards.shard_game_count(1001, 5, index)
            for index in range(5)
        ]
        self.assertEqual([201, 200, 200, 200, 200], counts)
        self.assertEqual(1001, sum(counts))

    def test_shard_seed_offsets_numeric_seed(self) -> None:
        self.assertEqual("2026060107", selfplay_shards.shard_seed("2026060104", 3))

    def test_resolve_generate_args_from_total_games(self) -> None:
        args = type("Args", (), {
            "total_games": 1001,
            "shards": 5,
            "shard_index": 4,
            "games": None,
            "base_seed": "2026060104",
            "seed": None,
        })()
        selfplay_shards.resolve_generate_args(args)
        self.assertEqual(200, args.games)
        self.assertEqual("2026060108", args.seed)

    def make_metadata(
        self,
        root: Path,
        *,
        name: str,
        seed: str,
        games: int = 2,
        nnue_sha: str = "nnue",
        depth: int = 8,
    ) -> Path:
        pgn = root / f"{name}.pgn"
        write_pgn(pgn, games)
        payload = {
            "schema": selfplay_shards.SCHEMA,
            "created_at": "2026-06-01T00:00:00+0200",
            "host": "worker",
            "seed": seed,
            "pgn": str(pgn),
            "games_requested": games,
            "games_written": games,
            "engine_id": "Enyo test",
            "files": {
                "runner": {"path": "/runner", "sha256": "runner", "bytes": 1},
                "engine": {"path": "/engine", "sha256": "engine", "bytes": 1},
                "book": {"path": "/book", "sha256": "book", "bytes": 1},
                "nnue": {"path": "/net", "sha256": nnue_sha, "bytes": 1},
            },
            "settings": {
                "concurrency": 4,
                "threads": 1,
                "depth": depth,
                "tc": None,
                "book_format": "epd",
                "order": "random",
                "restart": "off",
                "engine_options": ["Hash=128"],
            },
        }
        meta = root / f"{name}.meta.json"
        meta.write_text(json.dumps(payload), encoding="utf-8")
        return meta

    def test_merge_writes_combined_pgn_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self.make_metadata(root, name="a", seed="100")
            second = self.make_metadata(root, name="b", seed="101")
            out = root / "merged.pgn"
            manifest = root / "manifest.json"

            with redirect_stdout(StringIO()):
                rc = selfplay_shards.cmd_merge(
                    type("Args", (), {
                        "metadata": [str(first), str(second)],
                        "output_pgn": str(out),
                        "manifest": str(manifest),
                        "force": False,
                    })()
                )

            self.assertEqual(0, rc)
            self.assertEqual(4, selfplay_shards.count_pgn_games(out))
            data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(selfplay_shards.MERGE_SCHEMA, data["schema"])
            self.assertEqual(4, data["games"])
            self.assertEqual(2, len(data["shards"]))

    def test_merge_allows_different_shard_game_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self.make_metadata(root, name="a", seed="100", games=2)
            second = self.make_metadata(root, name="b", seed="101", games=1)
            out = root / "merged.pgn"

            with redirect_stdout(StringIO()):
                rc = selfplay_shards.cmd_merge(
                    type("Args", (), {
                        "metadata": [str(first), str(second)],
                        "output_pgn": str(out),
                        "manifest": None,
                        "force": False,
                    })()
                )

            self.assertEqual(0, rc)
            self.assertEqual(3, selfplay_shards.count_pgn_games(out))

    def test_merge_rejects_incompatible_net_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self.make_metadata(root, name="a", seed="100", nnue_sha="one")
            second = self.make_metadata(root, name="b", seed="101", nnue_sha="two")

            with self.assertRaisesRegex(ValueError, "incompatible shard metadata"):
                selfplay_shards.validate_merge_inputs([first, second])

    def test_merge_rejects_incompatible_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self.make_metadata(root, name="a", seed="100", depth=8)
            second = self.make_metadata(root, name="b", seed="101", depth=9)

            with self.assertRaisesRegex(ValueError, "incompatible shard metadata"):
                selfplay_shards.validate_merge_inputs([first, second])

    def test_merge_rejects_duplicate_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self.make_metadata(root, name="a", seed="100")
            second = self.make_metadata(root, name="b", seed="100")

            with self.assertRaisesRegex(ValueError, "duplicate seed"):
                selfplay_shards.validate_merge_inputs([first, second])

    def test_merge_rejects_pgn_game_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = self.make_metadata(root, name="a", seed="100", games=2)
            payload = json.loads(meta.read_text(encoding="utf-8"))
            payload["games_written"] = 3
            meta.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "metadata says 3 games"):
                selfplay_shards.validate_merge_inputs([meta])


if __name__ == "__main__":
    unittest.main()
