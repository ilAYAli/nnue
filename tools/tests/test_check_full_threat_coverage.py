from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from tools.lib import enyo_nnue as nn2

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools/validate/check_full_threat_coverage.py"


def _threat_net(threat_weights: np.ndarray) -> nn2.Net:
    input_buckets = 1
    feature_channels = 12
    base = nn2.feature_count(input_buckets, feature_channels)
    input_weights = np.zeros((base + nn2.N_THREAT_FEATURES, nn2.N_HIDDEN), dtype=np.int16)
    input_weights[base:base + nn2.N_THREAT_FEATURES] = threat_weights
    return nn2.Net(
        input_weights=input_weights,
        input_biases=np.zeros(nn2.N_HIDDEN, dtype=np.int16),
        l1_weights=np.zeros((nn2.N_L2, nn2.N_L1), dtype=np.int8),
        l1_biases=np.zeros(nn2.N_L2, dtype=np.int32),
        l2_weights=np.zeros((nn2.N_L3, nn2.N_L2), dtype=np.float32),
        l2_biases=np.zeros(nn2.N_L3, dtype=np.float32),
        output_weights=np.zeros((1, nn2.N_L3), dtype=np.float32),
        output_biases=np.zeros(1, dtype=np.float32),
        input_buckets=input_buckets,
        feature_channels=feature_channels,
        output_buckets=1,
        full_threats=True,
        format_version=nn2.NETWORK_FORMAT_VERSION,
    )


def _write(path: Path, threat_weights: np.ndarray) -> None:
    nn2.write_net(_threat_net(threat_weights), path)


class FullThreatCoverageTests(unittest.TestCase):
    def run_check(self, trained: Path, initial: Path | None = None,
                  min_fraction: float | None = None) -> subprocess.CompletedProcess[str]:
        args = [sys.executable, str(SCRIPT), "--trained", str(trained)]
        if initial is not None:
            args += ["--initial", str(initial)]
        if min_fraction is not None:
            args += ["--min-nonzero-fraction", str(min_fraction)]
        return subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_accepts_broadly_trained_threat_rows(self) -> None:
        rng = np.random.default_rng(0)
        shape = (nn2.N_THREAT_FEATURES, nn2.N_HIDDEN)
        weights = np.zeros(shape, dtype=np.int16)
        mask = rng.random(shape) < 0.05
        weights[mask] = rng.integers(1, 100, size=int(mask.sum()), dtype=np.int16)

        with_tmp = self._tmp_path("trained.nn")
        _write(with_tmp, weights)

        proc = self.run_check(with_tmp)
        self.assertEqual(0, proc.returncode, proc.stderr)
        fraction = float(next(
            line.split("=")[1] for line in proc.stdout.splitlines()
            if line.startswith("trained_nonzero_fraction=")))
        self.assertAlmostEqual(0.05, fraction, delta=0.01)

    def test_rejects_collapsed_threat_rows(self) -> None:
        shape = (nn2.N_THREAT_FEATURES, nn2.N_HIDDEN)
        weights = np.zeros(shape, dtype=np.int16)
        weights.reshape(-1)[:16] = 1

        with_tmp = self._tmp_path("collapsed.nn")
        _write(with_tmp, weights)

        proc = self.run_check(with_tmp)
        self.assertNotEqual(0, proc.returncode)
        self.assertIn("collapsed under quantization", proc.stderr)

    def test_reports_initial_alongside_trained(self) -> None:
        shape = (nn2.N_THREAT_FEATURES, nn2.N_HIDDEN)
        healthy = np.zeros(shape, dtype=np.int16)
        healthy.reshape(-1)[:int(healthy.size * 0.02)] = 5
        collapsed = np.zeros(shape, dtype=np.int16)
        collapsed.reshape(-1)[:16] = 1

        trained_path = self._tmp_path("trained.nn")
        initial_path = self._tmp_path("initial.nn")
        _write(trained_path, healthy)
        _write(initial_path, collapsed)

        proc = self.run_check(trained_path, initial=initial_path)
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("initial_nonzero_fraction=", proc.stdout)
        self.assertIn("trained_nonzero_fraction=", proc.stdout)

    def _tmp_path(self, name: str) -> Path:
        tmp_dir = Path(self._tmp_dir.name)
        return tmp_dir / name

    def setUp(self) -> None:
        import tempfile
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)


if __name__ == "__main__":
    unittest.main()
