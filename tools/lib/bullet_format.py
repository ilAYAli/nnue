"""BulletFormat ChessBoard binary writer."""
from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO

from . import bullet_text


RECORD_BYTES = 32
FORMAT = "bulletformat-chessboard-v1"
STRUCT = struct.Struct("<Q16shBBB3s")

PIECE_TYPE = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
}


def swap_bytes(bb: int) -> int:
    return int.from_bytes(bb.to_bytes(8, "little")[::-1], "little")


def trailing_zeros(bb: int) -> int:
    if bb == 0:
        raise ValueError("missing bit")
    return (bb & -bb).bit_length() - 1


def parse_fen_bitboards(fen: str) -> tuple[list[int], int]:
    parts = fen.split()
    if len(parts) < 2:
        raise ValueError(f"malformed FEN: {fen!r}")
    board, stm_text = parts[0], parts[1]
    if stm_text not in {"w", "b"}:
        raise ValueError(f"malformed side-to-move in FEN: {fen!r}")

    bbs = [0] * 8
    ranks = board.split("/")
    if len(ranks) != 8:
        raise ValueError(f"malformed board in FEN: {fen!r}")

    for rank_index, rank in enumerate(ranks):
        file_index = 0
        for ch in rank:
            if "1" <= ch <= "8":
                file_index += int(ch)
                continue
            piece_type = PIECE_TYPE.get(ch.upper())
            if piece_type is None:
                raise ValueError(f"malformed piece {ch!r} in FEN: {fen!r}")
            if file_index >= 8:
                raise ValueError(f"too many files in FEN rank: {fen!r}")
            square = (7 - rank_index) * 8 + file_index
            bit = 1 << square
            bbs[0 if ch.isupper() else 1] |= bit
            bbs[2 + piece_type] |= bit
            file_index += 1
        if file_index != 8:
            raise ValueError(f"short FEN rank: {fen!r}")

    return bbs, 0 if stm_text == "w" else 1


def result_to_u8(result: float) -> int:
    if result > 0.5:
        return 2
    if result < 0.5:
        return 0
    return 1


def chessboard_from_raw(bbs: list[int], stm: int, score: int, result: float) -> bytes:
    if len(bbs) != 8:
        raise ValueError("expected 8 bitboards")
    if score < -32768 or score > 32767:
        raise ValueError(f"score out of i16 range: {score}")

    if stm == 1:
        bbs = [swap_bytes(bb) for bb in bbs]
        bbs[0], bbs[1] = bbs[1], bbs[0]
        score = -score
        result = 1.0 - result
    elif stm != 0:
        raise ValueError(f"invalid side-to-move: {stm}")

    occ = bbs[0] | bbs[1]
    pcs = bytearray(16)
    occ2 = occ
    idx = 0
    while occ2:
        square = trailing_zeros(occ2)
        bit = 1 << square
        occ2 &= occ2 - 1
        if idx >= 32:
            raise ValueError("too many pieces for BulletFormat ChessBoard")
        color = 8 if (bit & bbs[1]) else 0
        piece = next((i for i, bb in enumerate(bbs[2:]) if bit & bb), None)
        if piece is None:
            raise ValueError("occupied square has no piece type")
        pcs[idx // 2] |= (color | piece) << (4 * (idx & 1))
        idx += 1

    ksq = trailing_zeros(bbs[0] & bbs[7])
    opp_ksq = trailing_zeros(bbs[1] & bbs[7]) ^ 56
    return STRUCT.pack(occ, bytes(pcs), score, result_to_u8(result), ksq, opp_ksq, b"\0\0\0")


def row_to_bytes(row: dict, *, enyo_runtime_target: bool) -> bytes:
    fen = str(row["fen"])
    bbs, stm = parse_fen_bitboards(fen)
    score = bullet_text.white_score_from_row(
        row,
        enyo_runtime_target=enyo_runtime_target,
    )
    result = bullet_text.white_result_from_row(row)
    return chessboard_from_raw(bbs, stm, score, result)


def write_row(handle: BinaryIO, row: dict, *, enyo_runtime_target: bool) -> None:
    handle.write(row_to_bytes(row, enyo_runtime_target=enyo_runtime_target))


def record_count(path: str | Path) -> int:
    size = Path(path).stat().st_size
    if size % RECORD_BYTES:
        raise ValueError(f"{path}: size {size} is not a multiple of {RECORD_BYTES}")
    return size // RECORD_BYTES
