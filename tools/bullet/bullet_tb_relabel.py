#!/usr/bin/env python3
"""Copy BulletFormat data while correcting <=N-piece labels with Syzygy WDL.

BulletFormat ChessBoard records are normalized to side-to-move perspective:
piece colours 0/1 mean us/them, and both the stored score and the stored result
are relative to us.  We can therefore reconstruct the represented position with
us as White and apply Syzygy's side-to-move WDL directly.

Syzygy is used to *correct* the existing evaluation, not to overwrite it.  An
earlier revision mapped WDL straight onto -2045/0/+2045 and that destroyed the
lineage's endgame play, for three reasons:

  * 2045 is exactly the engine's runtime output clamp
    (tools/lib/bullet_text.py ENYO_RUNTIME_SCORE_CLAMP), so every won endgame
    trained at the saturation rail.
  * Every TB win collapsed to one value, so a mate-in-3 and a 60-move grind
    became indistinguishable and search lost the gradient it needs to convert.
  * Only the score was rewritten; the result byte kept the noisy game outcome,
    so a TB-drawn-but-won position trained score 0 against result "win".

Instead the oracle's own ranking is preserved *inside* each WDL class:

  * TB win/loss keeps the magnitude of the existing eval, clamped into
    [TB_WIN_FLOOR, TB_WIN_CEIL] and given the sign Syzygy proves correct.
  * TB draw damps the existing eval toward zero rather than hard-zeroing it,
    retaining "up material but theoretically held" as a small residual.
  * Cursed wins and blessed losses (|wdl| == 1) are draws under the 50-move
    rule and are treated as such.
  * The result byte is set from Syzygy, replacing the game outcome with ground
    truth for exactly the positions where ground truth is known.

All bytes other than the score and result fields are copied verbatim.
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import struct
import sys
import time
from pathlib import Path
from typing import BinaryIO

import chess
import chess.syzygy


RECORD_BYTES = 32
OCCUPANCY = struct.Struct("<Q")
SCORE = struct.Struct("<h")
RESULT = struct.Struct("<B")
SCORE_OFFSET = 24
RESULT_OFFSET = 26

# Kept well below ENYO_RUNTIME_SCORE_CLAMP (2045) so no label ever trains at
# the runtime output rail.
TB_WIN_FLOOR = 350
TB_WIN_CEIL = 1200
TB_DRAW_DAMP = 0.10

# BulletFormat result byte, side-to-move relative.
RESULT_LOSS = 0
RESULT_DRAW = 1
RESULT_WIN = 2

DEFAULT_TB_DIRS = (
    Path("~/assets/tablebases/6-wdl"),
    Path("~/assets/tablebases/3-4-5-wdl"),
)


def wdl_to_cp(wdl: int, score: int) -> int:
    """Correct `score` using Syzygy `wdl`, preserving rank within the class.

    `wdl` follows python-chess: 2 win, 1 cursed win, 0 draw, -1 blessed loss,
    -2 loss.  Cursed wins and blessed losses are draws under the 50-move rule.
    """
    if abs(wdl) <= 1:
        return int(round(score * TB_DRAW_DAMP))
    magnitude = min(max(abs(score), TB_WIN_FLOOR), TB_WIN_CEIL)
    return magnitude if wdl > 0 else -magnitude


def wdl_to_result(wdl: int) -> int:
    if wdl > 1:
        return RESULT_WIN
    if wdl < -1:
        return RESULT_LOSS
    return RESULT_DRAW


def record_board(record: bytes) -> chess.Board:
    if len(record) != RECORD_BYTES:
        raise ValueError(f"expected {RECORD_BYTES}-byte BulletFormat record")
    occupancy = OCCUPANCY.unpack_from(record)[0]
    packed_pieces = record[8:24]
    board = chess.Board.empty()
    board.turn = chess.WHITE
    board.castling_rights = chess.BB_EMPTY
    board.ep_square = None
    board.halfmove_clock = 0
    board.fullmove_number = 1
    for index, square in enumerate(chess.scan_forward(occupancy)):
        nibble = (packed_pieces[index // 2] >> (4 * (index & 1))) & 0xF
        piece_index = nibble & 0x7
        if piece_index > 5:
            raise ValueError(f"invalid BulletFormat piece code {nibble}")
        colour = chess.BLACK if nibble & 0x8 else chess.WHITE
        board.set_piece_at(square, chess.Piece(piece_index + 1, colour))
    if not board.is_valid():
        raise ValueError("invalid board in BulletFormat record")
    return board


def write_stats(path: Path | None, stats: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def relabel(
    source: Path,
    destination: Path,
    *,
    tablebase: chess.syzygy.Tablebase,
    tb_pieces: int = 6,
    progress_every: int = 1_000_000,
    output_stream: BinaryIO | None = None,
) -> dict[str, object]:
    size = source.stat().st_size
    if size % RECORD_BYTES:
        raise ValueError(
            f"{source}: size {size} is not divisible by {RECORD_BYTES}"
        )
    records = size // RECORD_BYTES
    read = tb_candidates = tb_hits = tb_misses = changed = 0
    score_changed = result_changed = 0
    started = time.monotonic()

    with source.open("rb") as input_file:
        output_file = output_stream or destination.open("wb")
        close_output = output_stream is None
        try:
            while True:
                record = input_file.read(RECORD_BYTES)
                if not record:
                    break
                if len(record) != RECORD_BYTES:
                    raise ValueError(f"{source}: truncated BulletFormat record")
                read += 1
                mutable: bytearray | None = None
                occupancy = OCCUPANCY.unpack_from(record)[0]
                if occupancy.bit_count() <= tb_pieces:
                    tb_candidates += 1
                    board = record_board(record)
                    try:
                        wdl = tablebase.probe_wdl(board)
                    except (chess.syzygy.MissingTableError, KeyError):
                        tb_misses += 1
                    else:
                        tb_hits += 1
                        original = SCORE.unpack_from(record, SCORE_OFFSET)[0]
                        original_result = RESULT.unpack_from(record, RESULT_OFFSET)[0]
                        replacement = wdl_to_cp(wdl, original)
                        result = wdl_to_result(wdl)
                        if replacement != original or result != original_result:
                            mutable = bytearray(record)
                            SCORE.pack_into(mutable, SCORE_OFFSET, replacement)
                            RESULT.pack_into(mutable, RESULT_OFFSET, result)
                            changed += 1
                            score_changed += replacement != original
                            result_changed += result != original_result
                output_file.write(mutable if mutable is not None else record)
                if progress_every > 0 and read % progress_every == 0:
                    elapsed = max(time.monotonic() - started, 0.001)
                    print(
                        f"read={read}/{records} tb_candidates={tb_candidates} "
                        f"tb_hits={tb_hits} tb_misses={tb_misses} "
                        f"changed={changed} rate={read / elapsed:.0f}/s",
                        flush=True,
                    )
            output_file.flush()
            if close_output:
                os.fsync(output_file.fileno())
        finally:
            if close_output:
                output_file.close()

    elapsed = max(time.monotonic() - started, 0.001)
    return {
        "schema": "enyo.bullet-tb-relabel.v2",
        "input": str(source),
        "output": str(destination),
        "records": read,
        "tb_pieces": tb_pieces,
        "tb_candidates": tb_candidates,
        "tb_hits": tb_hits,
        "tb_misses": tb_misses,
        "changed": changed,
        "score_changed": score_changed,
        "result_changed": result_changed,
        "win_floor": TB_WIN_FLOOR,
        "win_ceil": TB_WIN_CEIL,
        "draw_damp": TB_DRAW_DAMP,
        "elapsed_s": round(elapsed, 6),
        "rate": round(read / elapsed, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tb-pieces", type=int, default=6)
    parser.add_argument("--tb-dir", action="append", type=Path, default=[])
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    destination = args.output.expanduser().resolve()
    if source == destination:
        raise SystemExit("--input and --output must be different files")
    if source.suffix != ".bullet" or destination.suffix != ".bullet":
        raise SystemExit("--input and --output must end with '.bullet'")
    if not source.is_file():
        raise SystemExit(f"input does not exist: {source}")
    if destination.exists() and not args.replace:
        raise SystemExit(f"output exists: {destination}; pass --replace to overwrite")
    if args.tb_pieces < 3 or args.tb_pieces > 7:
        raise SystemExit("--tb-pieces must be between 3 and 7")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be non-negative")

    tb_dirs = [
        path.expanduser().resolve()
        for path in (args.tb_dir or list(DEFAULT_TB_DIRS))
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    tablebase = chess.syzygy.open_tablebase(str(tb_dirs[0]))
    try:
        for extra in tb_dirs[1:]:
            tablebase.add_directory(str(extra))
        stats = relabel(
            source,
            temporary,
            tablebase=tablebase,
            tb_pieces=args.tb_pieces,
            progress_every=args.progress_every,
        )
        os.chmod(temporary, stat.S_IMODE(source.stat().st_mode))
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        stats["output"] = str(destination)
        write_stats(args.stats.expanduser().resolve() if args.stats else None, stats)
        print(json.dumps(stats, sort_keys=True), flush=True)
    finally:
        tablebase.close()
        temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
