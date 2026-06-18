#!/usr/bin/env python3
"""Convert LCZero V6 training records to Enyo JSONL position rows."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import struct
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import chess


POLICY_SIZE = 1858
PLANES = 104
META_BYTES = 8
VALUE_FIELDS = (
    "root_q",
    "best_q",
    "root_d",
    "best_d",
    "root_m",
    "best_m",
    "plies_left",
    "result_q",
    "result_d",
    "played_q",
    "played_d",
    "played_m",
    "orig_q",
    "orig_d",
    "orig_m",
)
HEADER = struct.Struct("<II")
POLICY = struct.Struct(f"<{POLICY_SIZE}f")
PLANE_DATA = struct.Struct(f"<{PLANES}Q")
META = struct.Struct(f"<{META_BYTES}B")
VALUES = struct.Struct(f"<{len(VALUE_FIELDS)}f")
TAIL = struct.Struct("<IHHfI")
RECORD_SIZE = (
    HEADER.size + POLICY.size + PLANE_DATA.size + META.size
    + VALUES.size + TAIL.size
)
PIECE_TYPES = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)


@dataclass(frozen=True)
class Record:
    version: int
    input_format: int
    policy: tuple[float, ...]
    planes: tuple[int, ...]
    meta: tuple[int, ...]
    values: tuple[float, ...]
    visits: int
    played_idx: int
    best_idx: int
    policy_kld: float


@dataclass
class Stats:
    files: int = 0
    records: int = 0
    rows: int = 0
    invalid_records: int = 0
    unsupported_records: int = 0
    invalid_boards: int = 0
    illegal_played: int = 0
    illegal_best: int = 0
    illegal_top_policy: int = 0
    legal_played: int = 0
    legal_best: int = 0
    legal_top_policy: int = 0
    total_top_policy: int = 0
    tags: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.tags is None:
            self.tags = Counter()


def square_name(square: int) -> str:
    return chess.square_name(square)


def generate_policy_moves() -> list[str]:
    moves: list[str] = []
    for from_sq in range(64):
        from_file = chess.square_file(from_sq)
        from_rank = chess.square_rank(from_sq)
        for to_sq in range(64):
            if from_sq == to_sq:
                continue
            to_file = chess.square_file(to_sq)
            to_rank = chess.square_rank(to_sq)
            df = abs(to_file - from_file)
            dr = abs(to_rank - from_rank)
            queen_like = from_file == to_file or from_rank == to_rank or df == dr
            knight = (df, dr) in {(1, 2), (2, 1)}
            if queen_like or knight:
                moves.append(square_name(from_sq) + square_name(to_sq))

    for from_file in range(8):
        from_sq = chess.square(from_file, 6)
        for to_file in (from_file - 1, from_file, from_file + 1):
            if not 0 <= to_file < 8:
                continue
            to_sq = chess.square(to_file, 7)
            for promo in "qrb":
                moves.append(square_name(from_sq) + square_name(to_sq) + promo)

    if len(moves) != POLICY_SIZE:
        raise RuntimeError(f"LC0 policy size mismatch: {len(moves)}")
    return moves


POLICY_MOVES = generate_policy_moves()


def flip_rank(square: int) -> int:
    return chess.square(chess.square_file(square), 7 - chess.square_rank(square))


def flip_file(square: int) -> int:
    return chess.square(7 - chess.square_file(square), chess.square_rank(square))


def map_policy_uci(index: int, board: chess.Board) -> str | None:
    if not 0 <= index < len(POLICY_MOVES):
        return None
    move = POLICY_MOVES[index]
    from_sq = chess.parse_square(move[:2])
    to_sq = chess.parse_square(move[2:4])
    promo = move[4:] if len(move) > 4 else ""
    if board.turn == chess.BLACK:
        from_sq = flip_rank(from_sq)
        to_sq = flip_rank(to_sq)
    uci = chess.square_name(from_sq) + chess.square_name(to_sq) + promo
    piece = board.piece_at(from_sq)
    if piece and piece.piece_type == chess.PAWN and not promo:
        to_rank = chess.square_rank(to_sq)
        if (piece.color == chess.WHITE and to_rank == 7) or (
                piece.color == chess.BLACK and to_rank == 0):
            uci += "n"
    return uci


def bit_is_set(bitboard: int, square: int) -> bool:
    return bool(bitboard & (1 << square))


def decode_board(planes: tuple[int, ...], meta: tuple[int, ...]) -> chess.Board:
    side_black = bool(meta[4])
    board = chess.Board.empty()
    board.turn = chess.BLACK if side_black else chess.WHITE
    board.castling_rights = chess.BB_EMPTY
    board.ep_square = None
    board.halfmove_clock = int(meta[5])
    board.fullmove_number = 1

    for plane in range(12):
        ours = plane < 6
        piece_index = plane if ours else plane - 6
        piece_type = PIECE_TYPES[piece_index]
        bitboard = int(planes[plane])
        for square in range(64):
            if not bit_is_set(bitboard, square):
                continue
            internal_square = flip_file(square)
            actual_square = flip_rank(internal_square) if side_black else internal_square
            if side_black:
                color = chess.BLACK if ours else chess.WHITE
            else:
                color = chess.WHITE if ours else chess.BLACK
            board.set_piece_at(actual_square, chess.Piece(piece_type, color))

    we_000, we_00, they_000, they_00 = (bool(meta[i]) for i in range(4))
    if side_black:
        if they_00:
            board.castling_rights |= chess.BB_H1
        if they_000:
            board.castling_rights |= chess.BB_A1
        if we_00:
            board.castling_rights |= chess.BB_H8
        if we_000:
            board.castling_rights |= chess.BB_A8
    else:
        if we_00:
            board.castling_rights |= chess.BB_H1
        if we_000:
            board.castling_rights |= chess.BB_A1
        if they_00:
            board.castling_rights |= chess.BB_H8
        if they_000:
            board.castling_rights |= chess.BB_A8

    board.castling_rights = board.clean_castling_rights()
    return board


def parse_record(raw: bytes, offset: int) -> Record:
    cursor = offset
    version, input_format = HEADER.unpack_from(raw, cursor)
    cursor += HEADER.size
    policy = POLICY.unpack_from(raw, cursor)
    cursor += POLICY.size
    planes = PLANE_DATA.unpack_from(raw, cursor)
    cursor += PLANE_DATA.size
    meta = META.unpack_from(raw, cursor)
    cursor += META.size
    values = VALUES.unpack_from(raw, cursor)
    cursor += VALUES.size
    visits, played_idx, best_idx, policy_kld, _reserved = TAIL.unpack_from(raw, cursor)
    return Record(
        version=version,
        input_format=input_format,
        policy=policy,
        planes=planes,
        meta=meta,
        values=values,
        visits=visits,
        played_idx=played_idx,
        best_idx=best_idx,
        policy_kld=policy_kld,
    )


def read_member_payload(name: str, handle: BinaryIO) -> bytes:
    data = handle.read()
    if name.endswith(".gz"):
        return gzip.decompress(data)
    return data


def iter_payloads(path: Path) -> Iterable[tuple[str, bytes]]:
    if path.is_dir():
        for item in sorted(path.rglob("training.*")):
            if item.is_file():
                with item.open("rb") as handle:
                    yield str(item), read_member_payload(item.name, handle)
        return

    if tarfile.is_tarfile(path):
        with tarfile.open(path) as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = member.name
                base = Path(name).name
                if not base.startswith("training."):
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                yield name, read_member_payload(name, extracted)
        return

    with path.open("rb") as handle:
        yield str(path), read_member_payload(path.name, handle)


def top_policy_indices(policy: tuple[float, ...], count: int) -> list[int]:
    if count <= 0:
        return []
    return sorted(
        range(len(policy)),
        key=lambda idx: (-float(policy[idx]), idx),
    )[:count]


def clean_float(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(value)


def add_move(
    moves: dict[int, dict],
    *,
    index: int,
    board: chess.Board,
    policy: tuple[float, ...],
    role: str,
    rank: int | None = None,
) -> None:
    uci = map_policy_uci(index, board)
    if uci is None:
        return
    legal = False
    try:
        legal = chess.Move.from_uci(uci) in board.legal_moves
    except ValueError:
        legal = False
    item = moves.setdefault(index, {
        "idx": index,
        "move": uci,
        "policy": clean_float(policy[index]) if 0 <= index < len(policy) else 0.0,
        "roles": [],
        "legal": legal,
    })
    if role not in item["roles"]:
        item["roles"].append(role)
    if rank is not None:
        item["rank"] = min(int(item.get("rank", rank)), rank)


def record_to_row(
    record: Record,
    *,
    source_file: str,
    record_index: int,
    top_policy: int,
    stats: Stats,
) -> dict | None:
    if record.version != 6 or record.input_format != 1:
        stats.unsupported_records += 1
        return None

    board = decode_board(record.planes, record.meta)
    if board.status() != chess.STATUS_VALID:
        stats.invalid_boards += 1
        return None

    moves: dict[int, dict] = {}
    add_move(moves, index=record.played_idx, board=board,
             policy=record.policy, role="played")
    add_move(moves, index=record.best_idx, board=board,
             policy=record.policy, role="best")
    top_indices = top_policy_indices(record.policy, top_policy)
    for rank, index in enumerate(top_indices, 1):
        add_move(moves, index=index, board=board, policy=record.policy,
                 role="top_policy", rank=rank)

    legal_by_role: dict[str, bool] = {"played": False, "best": False}
    top_legal = 0
    top_total = 0
    for item in moves.values():
        roles = set(item["roles"])
        if "played" in roles:
            legal_by_role["played"] = bool(item["legal"])
        if "best" in roles:
            legal_by_role["best"] = bool(item["legal"])
        if "top_policy" in roles:
            top_total += 1
            top_legal += int(bool(item["legal"]))

    stats.legal_played += int(legal_by_role["played"])
    stats.illegal_played += int(not legal_by_role["played"])
    stats.legal_best += int(legal_by_role["best"])
    stats.illegal_best += int(not legal_by_role["best"])
    stats.legal_top_policy += top_legal
    stats.illegal_top_policy += top_total - top_legal
    stats.total_top_policy += top_total

    values = {
        key: clean_float(value)
        for key, value in zip(VALUE_FIELDS, record.values)
    }
    side = "b" if record.meta[4] else "w"
    tags = [
        "source:lc0",
        "lc0_format:v6",
        "lc0_input_format:1",
        "license:odbl",
    ]
    for tag in tags:
        stats.tags[tag] += 1

    return {
        "schema": "enyo.lc0.v1",
        "id": f"lc0-{Path(source_file).name}-{record_index}",
        "fen": board.fen(),
        "source": "lc0_training_data",
        "source_file": source_file,
        "record_index": record_index,
        "license": "ODbL-1.0",
        "lc0": {
            "version": record.version,
            "input_format": record.input_format,
            "meta": {
                "we_can_000": int(record.meta[0]),
                "we_can_00": int(record.meta[1]),
                "they_can_000": int(record.meta[2]),
                "they_can_00": int(record.meta[3]),
                "side_to_move": side,
                "rule50": int(record.meta[5]),
                "invariance": int(record.meta[6]),
                "reserved": int(record.meta[7]),
            },
            "visits": int(record.visits),
            "played_idx": int(record.played_idx),
            "best_idx": int(record.best_idx),
            "top_policy_idx": int(top_indices[0]) if top_indices else None,
            "policy_kld": clean_float(record.policy_kld),
            **values,
        },
        "moves": sorted(moves.values(), key=lambda item: (
            int(item.get("rank", 1000000)), int(item["idx"]))),
        "tags": tags,
    }


def convert(args: argparse.Namespace) -> Stats:
    stats = Stats()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    record_limit = args.max_records if args.max_records > 0 else None
    with args.out.open("w", encoding="utf-8") as out:
        for source_file, payload in iter_payloads(args.input):
            stats.files += 1
            if len(payload) % RECORD_SIZE != 0:
                stats.invalid_records += 1
                continue
            count = len(payload) // RECORD_SIZE
            for file_record_index in range(count):
                if record_limit is not None and stats.records >= record_limit:
                    return stats
                offset = file_record_index * RECORD_SIZE
                try:
                    record = parse_record(payload, offset)
                except struct.error:
                    stats.invalid_records += 1
                    continue
                stats.records += 1
                row = record_to_row(
                    record,
                    source_file=source_file,
                    record_index=file_record_index,
                    top_policy=args.top_policy,
                    stats=stats,
                )
                if row is None:
                    continue
                out.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                out.write("\n")
                stats.rows += 1
    return stats


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100.0 * numerator / denominator:.2f}%"


def pct_value(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def write_summary(path: Path, stats: Stats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"files={stats.files}",
        f"records={stats.records}",
        f"rows={stats.rows}",
        f"invalid_records={stats.invalid_records}",
        f"unsupported_records={stats.unsupported_records}",
        f"invalid_boards={stats.invalid_boards}",
        (
            f"played_legal={stats.legal_played}/"
            f"{stats.legal_played + stats.illegal_played} "
            f"({pct(stats.legal_played, stats.legal_played + stats.illegal_played)})"
        ),
        (
            f"best_legal={stats.legal_best}/"
            f"{stats.legal_best + stats.illegal_best} "
            f"({pct(stats.legal_best, stats.legal_best + stats.illegal_best)})"
        ),
        (
            f"top_policy_legal={stats.legal_top_policy}/"
            f"{stats.total_top_policy} "
            f"({pct(stats.legal_top_policy, stats.total_top_policy)})"
        ),
        "tags=" + ",".join(
            f"{key}:{value}" for key, value in sorted((stats.tags or {}).items())),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert LCZero V6 training records into Enyo JSONL rows.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--top-policy", type=int, default=8)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--min-played-legal-pct", type=float, default=99.0)
    parser.add_argument("--min-best-legal-pct", type=float, default=99.0)
    args = parser.parse_args()

    stats = convert(args)
    write_summary(args.summary, stats)
    if stats.rows < args.min_rows:
        raise SystemExit(
            f"rows={stats.rows} below min_rows={args.min_rows}")
    played_total = stats.legal_played + stats.illegal_played
    best_total = stats.legal_best + stats.illegal_best
    played_pct = pct_value(stats.legal_played, played_total)
    best_pct = pct_value(stats.legal_best, best_total)
    if played_pct < args.min_played_legal_pct:
        raise SystemExit(
            f"played_legal={played_pct:.2f}% below "
            f"{args.min_played_legal_pct:.2f}%")
    if best_pct < args.min_best_legal_pct:
        raise SystemExit(
            f"best_legal={best_pct:.2f}% below "
            f"{args.min_best_legal_pct:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
