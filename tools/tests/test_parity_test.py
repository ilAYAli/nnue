from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.validate import parity_test


class _RecordingModel:
    def __init__(self) -> None:
        self.input_buckets = 32
        self.feature_channels = 11
        self.full_threats = False
        self.args: tuple[torch.Tensor, ...] | None = None
        self.kwargs: dict[str, torch.Tensor] | None = None
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True

    def __call__(
        self,
        *args: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        self.args = args
        self.kwargs = kwargs
        return torch.tensor([12.6])


class ParityTestRegressionTests(unittest.TestCase):
    def test_eval_python_uses_flat_features_zero_offsets_and_scalar_score(
        self,
    ) -> None:
        model = _RecordingModel()
        with mock.patch.object(
            parity_test,
            "load_model_from_nn",
            return_value=model,
        ):
            score = parity_test.eval_python(
                Path("unused.nn"),
                parity_test.TEST_FEN,
            )

        self.assertTrue(model.eval_called)
        self.assertEqual(score, 13)
        self.assertIsNotNone(model.args)
        args = model.args
        assert args is not None
        self.assertEqual(args[0].ndim, 1)
        self.assertEqual(args[1].ndim, 1)
        pieces, _ = parity_test.nn2.parse_fen(parity_test.TEST_FEN)
        pieces.sort(key=lambda item: item[2])
        torch.testing.assert_close(
            args[0],
            torch.tensor(parity_test.nn2.features_from_pieces(
                pieces, parity_test.nn2.WHITE, 32, 11)),
        )
        torch.testing.assert_close(
            args[1],
            torch.tensor(parity_test.nn2.features_from_pieces(
                pieces, parity_test.nn2.BLACK, 32, 11)),
        )
        torch.testing.assert_close(args[2], torch.tensor([0]))
        torch.testing.assert_close(args[3], torch.tensor([0]))
        self.assertEqual(args[4].shape, (1,))
        self.assertEqual(args[5].shape, (1,))
        self.assertIsNotNone(model.kwargs)
        assert model.kwargs is not None
        self.assertEqual(model.kwargs["piece_count"].shape, (1,))

    def test_run_spike_trainer_uses_current_env_and_batches(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "real.bullet"
            data.write_bytes(b"data")
            output = root / "output"
            captured: dict[str, object] = {}

            def fake_run(
                command: list[str],
                *,
                env: dict[str, str],
                capture_output: bool,
                text: bool,
            ) -> SimpleNamespace:
                captured["command"] = command
                captured["env"] = env
                self.assertTrue(capture_output)
                self.assertTrue(text)
                nn_path = output / "parity-3" / "quantised.bin"
                nn_path.parent.mkdir(parents=True)
                nn_path.write_bytes(b"checkpoint")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                mock.patch.object(parity_test.subprocess, "run", fake_run),
                mock.patch.object(
                    parity_test.shutil,
                    "which",
                    return_value="/fake/enyo-bullet-spike",
                ),
                mock.patch.object(
                    parity_test,
                    "expand_enyo_input_buckets",
                    return_value=b"runtime",
                ) as expand,
                mock.patch.object(
                    parity_test,
                    "enyo_network_size",
                    return_value=7,
                ) as network_size,
                mock.patch.object(
                    parity_test,
                    "pad_enyo_hidden",
                    return_value=b"runtime",
                ) as pad_hidden,
                mock.patch.object(
                    parity_test,
                    "enyo_v2_container",
                    return_value=b"runtime",
                ) as container,
            ):
                result = parity_test.run_spike_trainer(
                    data,
                    output,
                    rows=4097,
                    superbatches=3,
                )

            self.assertEqual(
                result,
                output / "parity-runtime.nn",
            )
            self.assertEqual(result.read_bytes(), b"runtime")
            checkpoint = output / "parity-3" / "quantised.bin"
            self.assertEqual(checkpoint.read_bytes(), b"checkpoint")
            expand.assert_called_once_with(
                b"checkpoint",
                16,
                feature_channels=12,
                runtime_input_buckets=16,
                output_buckets=1,
                hidden=1024,
                l2=16,
            )
            network_size.assert_called_once_with(
                16,
                feature_channels=12,
                output_buckets=1,
                hidden=1024,
                l2=16,
            )
            pad_hidden.assert_called_once_with(
                b"runtime",
                16,
                feature_channels=12,
                output_buckets=1,
                hidden=1024,
            )
            container.assert_called_once_with(
                b"runtime",
                input_buckets=16,
                feature_channels=12,
                hidden=1024,
                output_buckets=1,
            )
            expected_env = {
                "ENYO_BULLET_DATA": str(data),
                "ENYO_BULLET_OUT": str(output),
                "ENYO_BULLET_NET_ID": "parity",
                "ENYO_BULLET_MODE": "enyo",
                "ENYO_BULLET_HIDDEN": "1024",
                "ENYO_BULLET_L2": "16",
                "ENYO_BULLET_BATCH_SIZE": "2048",
                "ENYO_BULLET_BATCHES": "3",
                "ENYO_BULLET_SUPERBATCHES": "3",
                "ENYO_BULLET_THREADS": "4",
                "ENYO_BULLET_LR": "1e-3",
                "ENYO_BULLET_FINAL_LR": "1e-4",
                "ENYO_BULLET_WDL": "0.3",
                "ENYO_BULLET_ENYO_INPUT_BUCKETS": "16",
                "ENYO_BULLET_ENYO_FEATURE_CHANNELS": "12",
                "ENYO_BULLET_ENYO_OUTPUT_BUCKETS": "1",
                "ENYO_BULLET_ENYO_INPUT_FACTORISER": "0",
                "ENYO_BULLET_EVAL_SCALE": "400",
                "ENYO_BULLET_SAVE_RATE": "3",
            }
            captured_env = captured["env"]
            assert isinstance(captured_env, dict)
            for key, value in expected_env.items():
                self.assertEqual(captured_env.get(key), value)
            for obsolete in (
                "ENYO_BULLET_DATASET",
                "ENYO_BULLET_OUTPUT",
                "ENYO_BULLET_LIMIT",
                "ENYO_BULLET_L2_SIZE",
                "ENYO_BULLET_L3_SIZE",
                "ENYO_BULLET_INPUT_BUCKETS",
                "ENYO_BULLET_FEATURE_CHANNELS",
                "ENYO_BULLET_OUTPUT_BUCKETS",
                "ENYO_BULLET_WDL_LAMBDA",
                "ENYO_BULLET_EXPORT_INIT_ONLY",
            ):
                self.assertNotIn(obsolete, expected_env)

    def test_runtime_export_trims_bullet_trailer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "real.bullet"
            data.write_bytes(b"data")
            output = root / "output"
            expected_size = parity_test.enyo_network_size(
                16,
                feature_channels=12,
                output_buckets=1,
                hidden=1024,
                l2=16,
            )
            self.assertEqual(expected_size, 25_203_012)
            payload = b"\x5a" * expected_size
            trailer = b"bullet" * 10

            def fake_run(
                command: list[str],
                *,
                env: dict[str, str],
                capture_output: bool,
                text: bool,
            ) -> SimpleNamespace:
                del command, env, capture_output, text
                checkpoint = output / "parity-2" / "quantised.bin"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(payload + trailer)
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                mock.patch.object(parity_test.subprocess, "run", fake_run),
                mock.patch.object(
                    parity_test.shutil,
                    "which",
                    return_value="/fake/enyo-bullet-spike",
                ),
            ):
                runtime = parity_test.run_spike_trainer(
                    data,
                    output,
                    rows=1,
                    superbatches=2,
                )

            self.assertEqual(runtime, output / "parity-runtime.nn")
            self.assertEqual(
                runtime.stat().st_size,
                parity_test.ENYO_NETWORK_HEADER_SIZE + 25_203_012,
            )
            self.assertEqual(
                runtime.read_bytes()[parity_test.ENYO_NETWORK_HEADER_SIZE:],
                payload,
            )
            checkpoint = output / "parity-2" / "quantised.bin"
            self.assertEqual(checkpoint.read_bytes(), payload + trailer)

    def test_run_spike_trainer_rejects_nonpositive_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rows in (0, -1):
                with self.subTest(rows=rows):
                    with self.assertRaisesRegex(
                        ValueError,
                        "rows must be positive",
                    ):
                        parity_test.run_spike_trainer(
                            root / "data.bullet",
                            root / "output",
                            rows,
                            1,
                        )

    def test_main_passes_real_data_path_to_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "input.bullet"
            engine = root / "enyo"
            output = root / "output"
            data.write_bytes(b"data")
            engine.write_bytes(b"engine")
            captured: dict[str, Path] = {}

            def fake_trainer(
                data_path: Path,
                output_dir: Path,
                rows: int,
                superbatches: int,
                input_buckets: int,
                feature_channels: int,
                hidden: int,
                output_buckets: int,
                full_threats: bool,
            ) -> Path:
                captured["data"] = data_path
                captured["output"] = output_dir
                self.assertEqual(rows, 12)
                self.assertEqual(superbatches, 2)
                self.assertEqual(
                    (input_buckets, feature_channels, hidden, output_buckets),
                    (16, 12, 1024, 8),
                )
                self.assertFalse(full_threats)
                return root / "parity.nn"

            argv = [
                "parity_test.py",
                "--data",
                str(data),
                "--engine",
                str(engine),
                "--output",
                str(output),
                "--rows",
                "12",
                "--superbatches",
                "2",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    parity_test,
                    "run_spike_trainer",
                    side_effect=fake_trainer,
                ),
                mock.patch.object(parity_test, "eval_python", return_value=7),
                mock.patch.object(parity_test, "eval_engine", return_value=7),
            ):
                result = parity_test.main()

            self.assertEqual(result, 0)
            self.assertEqual(captured["data"], data)
            self.assertEqual(captured["output"], output)


if __name__ == "__main__":
    unittest.main()
