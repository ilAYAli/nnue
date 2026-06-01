#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validate import net_provenance


class NetProvenanceTests(unittest.TestCase):
    def test_flags_recursive_berserk_init_and_stockfish_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_run = root / "runs" / "imported" / "fresh"
            init_train = init_run / "train" / "fresh_huber"
            init_train.mkdir(parents=True)
            init_net = init_train / "model.nn"
            init_net.write_bytes(b"")
            init_train.joinpath("train.log").write_text(
                "loading train rows from /tmp/enyo_teacher/fresh_labels/packed\n"
                "initializing from /repo/enyo/net/berserk-d43206fe90e4.nn\n",
                encoding="utf-8",
            )
            init_run.joinpath("assets").mkdir()

            run = root / "runs" / "candidate"
            train = run / "train"
            train.mkdir(parents=True)
            net = train / "model.nn"
            net.write_bytes(b"")
            run.joinpath("runs.log").write_text(
                f"init={init_net}\n"
                "data=/tmp/enyo_teacher/sf_d12_20m_20260510_115338/packed\n",
                encoding="utf-8",
            )
            pack = run / "pack" / "train"
            pack.mkdir(parents=True)
            pack.joinpath("meta.json").write_text(
                '{'
                '"source": "/tmp/enyo_teacher/sf_d12_20m_20260510_115338/labeled.jsonl",'
                '"source_map": {"stockfish": 0}'
                '}\n',
                encoding="utf-8",
            )

            result = net_provenance.analyze(net)

            self.assertEqual(result.init, "berserk-derived")
            self.assertEqual(result.label_source, "stockfish-oracle")
            self.assertFalse(result.clean_enyo_owned)
            self.assertIn("stockfish", result.label_sources)
            self.assertTrue(any("Berserk" in reason for reason in result.reasons))

    def test_allows_random_init_enyo_replay_with_stockfish_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "runs" / "clean"
            train = run / "train"
            train.mkdir(parents=True)
            net = train / "model.nn"
            net.write_bytes(b"")
            run.joinpath("runs.log").write_text(
                "start bullet_train\n"
                "enyo_l0_stdev=8 enyo_l1_stdev=1\n"
                "Training Preamble\n"
                "data=/data/replay-loss-20260531/loss_replay_child_targets.jsonl\n",
                encoding="utf-8",
            )
            pack = run / "pack" / "train"
            pack.mkdir(parents=True)
            pack.joinpath("meta.json").write_text(
                '{"source_map": {"stockfish": 0}}\n',
                encoding="utf-8",
            )

            result = net_provenance.analyze(net)

            self.assertEqual(result.init, "random")
            self.assertEqual(result.position_source, "enyo-replay")
            self.assertEqual(result.label_source, "stockfish-oracle")
            self.assertTrue(result.clean_enyo_owned)

    def test_unknown_init_is_not_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "runs" / "unknown"
            train = run / "train"
            train.mkdir(parents=True)
            net = train / "model.nn"
            net.write_bytes(b"")
            run.joinpath("runs.log").write_text(
                "data=/data/replay-loss-20260531/loss_replay_child_targets.jsonl\n",
                encoding="utf-8",
            )

            result = net_provenance.analyze(net)

            self.assertEqual(result.init, "unknown")
            self.assertFalse(result.clean_enyo_owned)

    def test_traces_clean_bullet_init_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "runs" / "base-clean"
            base_train = base / "train" / "base-clean" / "checkpoints" / "base-clean-64"
            weights = base_train / "optimiser_state" / "weights.bin"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"")
            base_train.joinpath("quantised.bin").write_bytes(b"")
            base.joinpath("runs.log").write_text(
                "start bullet_train\n"
                "enyo_l0_stdev=8 enyo_l1_stdev=1\n"
                "Training Preamble\n"
                "data=/data/replay-loss-20260531/loss_replay_child_targets.jsonl\n",
                encoding="utf-8",
            )
            pack = base / "pack" / "train"
            pack.mkdir(parents=True)
            pack.joinpath("meta.json").write_text(
                '{"source_map": {"stockfish": 0}}\n',
                encoding="utf-8",
            )

            run = root / "runs" / "continued-clean"
            train = run / "train"
            train.mkdir(parents=True)
            net = train / "model.nn"
            net.write_bytes(b"")
            run.joinpath("runs.log").write_text(
                "start bullet_train\n"
                "enyo_l0_stdev=8 enyo_l1_stdev=1\n"
                "Training Preamble\n"
                f"--init-weights {weights}\n"
                "data=/data/replay-loss-20260531/loss_replay_child_targets.jsonl\n",
                encoding="utf-8",
            )
            pack = run / "pack" / "train"
            pack.mkdir(parents=True)
            pack.joinpath("meta.json").write_text(
                '{"source_map": {"stockfish": 0}}\n',
                encoding="utf-8",
            )

            result = net_provenance.analyze(net)

            self.assertEqual(result.init, "random")
            self.assertIn(str(weights), result.init_chain)
            self.assertTrue(result.clean_enyo_owned)

    def test_rejects_borrowed_bullet_init_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "runs" / "base-borrowed"
            base_train = base / "train" / "base-borrowed" / "checkpoints" / "base-borrowed-64"
            weights = base_train / "optimiser_state" / "weights.bin"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"")
            base_train.joinpath("quantised.bin").write_bytes(b"")
            base.joinpath("runs.log").write_text(
                "start bullet_train\n"
                "enyo_l0_stdev=8 enyo_l1_stdev=1\n"
                "Training Preamble\n"
                "initializing from /repo/enyo/net/berserk-d43206fe90e4.nn\n"
                "data=/data/replay-loss-20260531/loss_replay_child_targets.jsonl\n",
                encoding="utf-8",
            )

            run = root / "runs" / "continued-borrowed"
            train = run / "train"
            train.mkdir(parents=True)
            net = train / "model.nn"
            net.write_bytes(b"")
            run.joinpath("runs.log").write_text(
                "start bullet_train\n"
                "enyo_l0_stdev=8 enyo_l1_stdev=1\n"
                "Training Preamble\n"
                f"--init-weights {weights}\n"
                "data=/data/replay-loss-20260531/loss_replay_child_targets.jsonl\n",
                encoding="utf-8",
            )
            pack = run / "pack" / "train"
            pack.mkdir(parents=True)
            pack.joinpath("meta.json").write_text(
                '{"source_map": {"stockfish": 0}}\n',
                encoding="utf-8",
            )

            result = net_provenance.analyze(net)

            self.assertEqual(result.init, "berserk-derived")
            self.assertIn(str(weights), result.init_chain)
            self.assertFalse(result.clean_enyo_owned)

    def test_ignores_previous_provenance_report_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "runs" / "clean-with-report"
            train = run / "train"
            train.mkdir(parents=True)
            net = train / "model.nn"
            net.write_bytes(b"")
            run.joinpath("runs.log").write_text(
                "start bullet_train\n"
                "enyo_l0_stdev=8 enyo_l1_stdev=1\n"
                "Training Preamble\n"
                "posgen.py selfplay --output /runs/posgen/selfplay.pgn "
                "--engine-option use_nnue=false\n"
                "data=/runs/posgen/source.jsonl\n"
                "net=/runs/clean-with-report/train/model.nn\n"
                "init=random\n"
                "data=mixed+mixed\n"
                "position_source=mixed\n"
                "label_source=mixed\n",
                encoding="utf-8",
            )
            pack = run / "pack" / "train"
            pack.mkdir(parents=True)
            pack.joinpath("meta.json").write_text(
                '{"source_map": {"stockfish": 0}}\n',
                encoding="utf-8",
            )

            result = net_provenance.analyze(net)

            self.assertEqual(result.init, "random")
            self.assertNotIn("random", result.init_chain)
            self.assertNotIn("mixed+mixed", result.position_refs)
            self.assertTrue(result.clean_enyo_owned)

    def test_rejects_external_stockfish_position_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "runs" / "external"
            train = run / "train"
            train.mkdir(parents=True)
            net = train / "model.nn"
            net.write_bytes(b"")
            run.joinpath("runs.log").write_text(
                "start bullet_train\n"
                "enyo_l0_stdev=8 enyo_l1_stdev=1\n"
                "Training Preamble\n"
                "data=/data/test80-2024-01-jan-2tb7p.min-v2.v6.binpack\n",
                encoding="utf-8",
            )

            result = net_provenance.analyze(net)

            self.assertEqual(result.init, "random")
            self.assertIn("external-stockfish", result.position_sources)
            self.assertFalse(result.clean_enyo_owned)

    def test_rejects_lc0_position_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "runs" / "lc0"
            train = run / "train"
            train.mkdir(parents=True)
            net = train / "model.nn"
            net.write_bytes(b"")
            run.joinpath("runs.log").write_text(
                "start bullet_train\n"
                "enyo_l0_stdev=8 enyo_l1_stdev=1\n"
                "Training Preamble\n"
                "data=/data/lc0/run1/lc0_positions.jsonl\n",
                encoding="utf-8",
            )

            result = net_provenance.analyze(net)

            self.assertEqual(result.init, "random")
            self.assertIn("lc0", result.position_sources)
            self.assertFalse(result.clean_enyo_owned)

    def test_rejects_berserk_selfplay_net_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "runs" / "berserk-selfplay"
            train = run / "train"
            train.mkdir(parents=True)
            net = train / "model.nn"
            net.write_bytes(b"")
            run.joinpath("runs.log").write_text(
                "start bullet_train\n"
                "enyo_l0_stdev=8 enyo_l1_stdev=1\n"
                "Training Preamble\n"
                "posgen.py selfplay --output /runs/posgen/selfplay.pgn "
                "--nnue-file /repo/enyo/net/berserk-d43206fe90e4.nn\n"
                "data=/runs/posgen/source.jsonl\n",
                encoding="utf-8",
            )
            pack = run / "pack" / "train"
            pack.mkdir(parents=True)
            pack.joinpath("meta.json").write_text(
                '{"source_map": {"stockfish": 0}}\n',
                encoding="utf-8",
            )

            result = net_provenance.analyze(net)

            self.assertIn("berserk", result.position_sources)
            self.assertFalse(result.clean_enyo_owned)

    def test_allows_hce_selfplay_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "runs" / "hce-selfplay"
            train = run / "train"
            train.mkdir(parents=True)
            net = train / "model.nn"
            net.write_bytes(b"")
            run.joinpath("runs.log").write_text(
                "start bullet_train\n"
                "enyo_l0_stdev=8 enyo_l1_stdev=1\n"
                "Training Preamble\n"
                "posgen.py selfplay --output /runs/posgen/selfplay.pgn "
                "--engine-option use_nnue=false\n"
                "data=/runs/posgen/source.jsonl\n",
                encoding="utf-8",
            )
            pack = run / "pack" / "train"
            pack.mkdir(parents=True)
            pack.joinpath("meta.json").write_text(
                '{"source_map": {"stockfish": 0}}\n',
                encoding="utf-8",
            )

            result = net_provenance.analyze(net)

            self.assertIn("enyo-hce", result.position_sources)
            self.assertTrue(result.clean_enyo_owned)

    def test_ignores_selfplay_runner_diagnostic_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "runs" / "hce-selfplay-runner"
            train = run / "train"
            train.mkdir(parents=True)
            net = train / "model.nn"
            net.write_bytes(b"")
            run.joinpath("runs.log").write_text(
                "start bullet_train\n"
                "enyo_l0_stdev=8 enyo_l1_stdev=1\n"
                "Training Preamble\n"
                "posgen.py selfplay --output /runs/posgen/selfplay.pgn "
                "--engine-option Hash=128 --engine-option use_nnue=false\n"
                "/repo/tools/posgen/run_selfplay.sh --output /runs/posgen/selfplay.pgn "
                "--engine-option Hash=128 --engine-option use_nnue=false\n"
                "selfplay: runner=/home/petter/source/fastchess/fastchess\n"
                "data=/runs/posgen/source.jsonl\n",
                encoding="utf-8",
            )
            pack = run / "pack" / "train"
            pack.mkdir(parents=True)
            pack.joinpath("meta.json").write_text(
                '{"source_map": {"stockfish": 0}}\n',
                encoding="utf-8",
            )

            result = net_provenance.analyze(net)

            self.assertIn("enyo-hce", result.position_sources)
            self.assertNotIn("borrowed-default", result.position_sources)
            self.assertTrue(result.clean_enyo_owned)

    def test_rejects_implicit_default_selfplay_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "runs" / "implicit-default"
            train = run / "train"
            train.mkdir(parents=True)
            net = train / "model.nn"
            net.write_bytes(b"")
            run.joinpath("runs.log").write_text(
                "start bullet_train\n"
                "enyo_l0_stdev=8 enyo_l1_stdev=1\n"
                "Training Preamble\n"
                "posgen.py selfplay --output /runs/posgen/selfplay.pgn\n"
                "data=/runs/posgen/source.jsonl\n",
                encoding="utf-8",
            )
            pack = run / "pack" / "train"
            pack.mkdir(parents=True)
            pack.joinpath("meta.json").write_text(
                '{"source_map": {"stockfish": 0}}\n',
                encoding="utf-8",
            )

            result = net_provenance.analyze(net)

            self.assertIn("borrowed-default", result.position_sources)
            self.assertFalse(result.clean_enyo_owned)

    def test_rejects_default_net_selfplay_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "runs" / "default-selfplay"
            train = run / "train"
            train.mkdir(parents=True)
            net = train / "model.nn"
            net.write_bytes(b"")
            run.joinpath("runs.log").write_text(
                "start bullet_train\n"
                "enyo_l0_stdev=8 enyo_l1_stdev=1\n"
                "Training Preamble\n"
                "posgen.py selfplay --output /runs/posgen/selfplay.pgn "
                "--nnue-file /repo/enyo/net/default.net\n"
                "data=/runs/posgen/source.jsonl\n",
                encoding="utf-8",
            )
            pack = run / "pack" / "train"
            pack.mkdir(parents=True)
            pack.joinpath("meta.json").write_text(
                '{"source_map": {"stockfish": 0}}\n',
                encoding="utf-8",
            )

            result = net_provenance.analyze(net)

            self.assertIn("borrowed-default", result.position_sources)
            self.assertFalse(result.clean_enyo_owned)
            self.assertTrue(any("default.net" in reason for reason in result.reasons))


if __name__ == "__main__":
    unittest.main()
