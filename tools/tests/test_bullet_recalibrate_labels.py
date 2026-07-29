from __future__ import annotations

import io
import struct
import tempfile
import unittest
from pathlib import Path

from tools.bullet import bullet_recalibrate_labels as recal


RECORD = struct.Struct("<Q16shBBB3s")


def make_record(score: int) -> bytes:
    return RECORD.pack(0, bytes(16), score, 1, 4, 60, b"\0\0\0")


def score_of(record: bytes) -> int:
    return struct.unpack_from("<h", record, 24)[0]


class RecalibrateTests(unittest.TestCase):
    def test_zero_stays_zero(self) -> None:
        self.assertEqual(0, recal.recalibrate(0))

    def test_sign_is_preserved(self) -> None:
        for value in (37, 250, 900, 1800):
            self.assertGreater(recal.recalibrate(value), 0)
            self.assertEqual(-recal.recalibrate(value), recal.recalibrate(-value))

    def test_mapping_is_monotone(self) -> None:
        previous = -1
        for value in range(0, 3000, 7):
            mapped = recal.recalibrate(value)
            self.assertGreaterEqual(mapped, previous)
            previous = mapped

    def test_labels_are_compressed_toward_static_eval(self) -> None:
        # Search scores exceed static eval everywhere, so every magnitude must
        # shrink; the gap is proportionally largest in quiet positions.
        for value in (100, 300, 700, 1300):
            self.assertLess(recal.recalibrate(value), value)
        quiet_ratio = recal.recalibrate(155) / 155
        decisive_ratio = recal.recalibrate(1343) / 1343
        self.assertLess(quiet_ratio, decisive_ratio)

    def test_never_exceeds_the_runtime_clamp(self) -> None:
        for value in (2045, 5000, 10000, 32000):
            self.assertLessEqual(abs(recal.recalibrate(value)), 2045)
            self.assertLessEqual(abs(recal.recalibrate(-value)), 2045)

    def test_anchors_reproduce_the_measured_calibration(self) -> None:
        # Each anchor must map its measured mean |label| onto the measured
        # mean |static eval| it was paired with.
        for label, static in recal.CALIBRATION:
            self.assertAlmostEqual(recal.recalibrate(int(round(label))), round(static), delta=2)

    def test_rewrite_changes_only_the_score_field(self) -> None:
        original = make_record(700)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "in.bullet"
            source.write_bytes(original)
            out = io.BytesIO()
            stats = recal.rewrite(
                source, Path(tmp) / "out.bullet", output_stream=out, progress_every=0
            )
        transformed = out.getvalue()
        self.assertEqual(recal.RECORD_BYTES, len(transformed))
        self.assertEqual(original[:24], transformed[:24])
        self.assertEqual(original[26:], transformed[26:])
        self.assertEqual(recal.recalibrate(700), score_of(transformed))
        self.assertEqual(1, stats["records"])
        self.assertEqual(1, stats["changed"])

    def test_rewrite_rejects_misaligned_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "bad.bullet"
            source.write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "not divisible by 32"):
                recal.rewrite(source, Path(tmp) / "out.bullet", output_stream=io.BytesIO())


if __name__ == "__main__":
    unittest.main()
