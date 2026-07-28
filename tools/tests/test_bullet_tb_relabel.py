from __future__ import annotations

import io
import struct
import tempfile
import unittest
from pathlib import Path

from tools.bullet import bullet_tb_relabel


RECORD = struct.Struct("<Q16shBBB3s")


def make_record(pieces: list[tuple[int, int]], score: int, result: int = 1) -> bytes:
    occupancy = 0
    packed = bytearray(16)
    for index, (square, piece) in enumerate(sorted(pieces)):
        occupancy |= 1 << square
        packed[index // 2] |= piece << (4 * (index & 1))
    return RECORD.pack(occupancy, bytes(packed), score, result, 4, 60, b"\0\0\0")


class FakeTablebase:
    def __init__(self, wdl: int) -> None:
        self.wdl = wdl
        self.boards = []

    def probe_wdl(self, board) -> int:
        self.boards.append(board)
        return self.wdl


SIX_PIECE = [(4, 5), (60, 13), (8, 0), (48, 8), (3, 4), (59, 12)]
SEVEN_PIECE = SIX_PIECE + [(1, 1)]


def relabel_one(record: bytes, wdl: int) -> tuple[bytes, dict]:
    """Run one record through the relabeler and return the transformed bytes."""
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "source.bullet"
        source.write_bytes(record)
        output = io.BytesIO()
        stats = bullet_tb_relabel.relabel(
            source,
            Path(tmp) / "output.bullet",
            tablebase=FakeTablebase(wdl),
            output_stream=output,
            progress_every=0,
        )
    return output.getvalue(), stats


def score_of(record: bytes) -> int:
    return struct.unpack_from("<h", record, 24)[0]


def result_of(record: bytes) -> int:
    return record[26]


class BulletTbRelabelTests(unittest.TestCase):
    def test_touches_only_score_and_result_for_six_piece_positions(self) -> None:
        six_piece = make_record(SIX_PIECE, 123)
        seven_piece = make_record(SEVEN_PIECE, 456)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.bullet"
            destination = Path(tmp) / "output.bullet"
            source.write_bytes(six_piece + seven_piece)
            output = io.BytesIO()
            tablebase = FakeTablebase(-2)

            stats = bullet_tb_relabel.relabel(
                source,
                destination,
                tablebase=tablebase,
                output_stream=output,
                progress_every=0,
            )

        transformed = output.getvalue()
        self.assertEqual(2 * bullet_tb_relabel.RECORD_BYTES, len(transformed))
        # A proven loss keeps the eval's magnitude, floored, with TB's sign.
        self.assertEqual(-bullet_tb_relabel.TB_WIN_FLOOR, score_of(transformed))
        self.assertEqual(bullet_tb_relabel.RESULT_LOSS, result_of(transformed))
        self.assertEqual(six_piece[:24], transformed[:24])
        self.assertEqual(six_piece[27:32], transformed[27:32])
        # Positions outside tablebase range are copied verbatim.
        self.assertEqual(seven_piece, transformed[32:])
        self.assertEqual(1, stats["tb_candidates"])
        self.assertEqual(1, stats["tb_hits"])
        self.assertEqual(1, stats["changed"])
        self.assertEqual(1, len(tablebase.boards))
        self.assertTrue(tablebase.boards[0].turn)

    def test_never_labels_at_the_runtime_clamp(self) -> None:
        """2045 is the engine's output rail; no label may reach it."""
        for wdl in (-2, -1, 0, 1, 2):
            for score in (-2045, -1800, -50, 0, 50, 1800, 2045):
                transformed, _ = relabel_one(make_record(SIX_PIECE, score), wdl)
                self.assertLess(abs(score_of(transformed)), 2045)
                self.assertLessEqual(
                    abs(score_of(transformed)), bullet_tb_relabel.TB_WIN_CEIL
                )

    def test_win_preserves_ranking_between_floor_and_ceiling(self) -> None:
        near = score_of(relabel_one(make_record(SIX_PIECE, 900), 2)[0])
        far = score_of(relabel_one(make_record(SIX_PIECE, 400), 2)[0])
        self.assertEqual(900, near)
        self.assertEqual(400, far)
        # A clearly won position still outranks a barely won one.
        self.assertGreater(near, far)

    def test_win_clamps_into_band(self) -> None:
        self.assertEqual(
            bullet_tb_relabel.TB_WIN_FLOOR,
            score_of(relabel_one(make_record(SIX_PIECE, 10), 2)[0]),
        )
        self.assertEqual(
            bullet_tb_relabel.TB_WIN_CEIL,
            score_of(relabel_one(make_record(SIX_PIECE, 2045), 2)[0]),
        )

    def test_win_corrects_a_wrong_sign(self) -> None:
        transformed, _ = relabel_one(make_record(SIX_PIECE, -600), 2)
        self.assertEqual(600, score_of(transformed))
        self.assertEqual(bullet_tb_relabel.RESULT_WIN, result_of(transformed))

    def test_draw_damps_toward_zero_without_hard_zeroing(self) -> None:
        transformed, _ = relabel_one(make_record(SIX_PIECE, 800), 0)
        self.assertEqual(80, score_of(transformed))
        self.assertEqual(bullet_tb_relabel.RESULT_DRAW, result_of(transformed))

    def test_cursed_win_and_blessed_loss_are_draws(self) -> None:
        for wdl in (1, -1):
            transformed, _ = relabel_one(make_record(SIX_PIECE, 800), wdl)
            self.assertEqual(80, score_of(transformed))
            self.assertEqual(bullet_tb_relabel.RESULT_DRAW, result_of(transformed))

    def test_result_is_rewritten_even_when_score_is_unchanged(self) -> None:
        # Score already sits at the floor for a proven win, but the record
        # carries a "draw" outcome from self-play; the result must still move.
        record = make_record(SIX_PIECE, bullet_tb_relabel.TB_WIN_FLOOR, result=1)
        transformed, stats = relabel_one(record, 2)
        self.assertEqual(bullet_tb_relabel.TB_WIN_FLOOR, score_of(transformed))
        self.assertEqual(bullet_tb_relabel.RESULT_WIN, result_of(transformed))
        self.assertEqual(0, stats["score_changed"])
        self.assertEqual(1, stats["result_changed"])
        self.assertEqual(1, stats["changed"])

    def test_rejects_misaligned_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "bad.bullet"
            source.write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "not divisible by 32"):
                bullet_tb_relabel.relabel(
                    source,
                    Path(tmp) / "output.bullet",
                    tablebase=FakeTablebase(0),
                    output_stream=io.BytesIO(),
                )


if __name__ == "__main__":
    unittest.main()
