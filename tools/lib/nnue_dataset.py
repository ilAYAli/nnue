"""Self-play JSONL dataset for the NNUE trainer."""
from __future__ import annotations

import json
import struct
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path

import numpy as np

from . import bullet_format
from . import enyo_nnue as nn2


BULLET_CHESSBOARD_STRUCT = struct.Struct("<Q16shBBB3s")
assert BULLET_CHESSBOARD_STRUCT.size == bullet_format.RECORD_BYTES


class SequentialLoader:
    """Minimal no-shuffle batch iterator, replacing torch's DataLoader."""

    def __init__(self, dataset: Sequence, batch_size: int, collate_fn: Callable):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn

    def __iter__(self) -> Iterator:
        n = len(self.dataset)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            yield self.collate_fn([self.dataset[i] for i in range(start, end)])


class FenScoreDataset:
    def __init__(self, rows: Iterable[dict], *,
                 input_buckets: int = nn2.DEFAULT_N_KING_BUCKETS,
                 feature_channels: int = nn2.DEFAULT_N_FEATURE_CHANNELS,
                 full_threats: bool = False,
                 slider_xray_threats: bool = False):
        self.items: list[
            tuple[list[int], list[int], int, int, float, float, float, int]
        ] = []
        source_map: dict[str, int] = {}
        for row in rows:
            fen = row["fen"]
            pieces, stm = nn2.parse_fen(fen)
            pieces.sort(key=lambda item: item[2])
            phase_scale = nn2.phase_scale_from_pieces(pieces)
            source_name = str(
                row.get("source_type") or row.get("teacher") or "unknown")
            if source_name not in source_map:
                source_map[source_name] = len(source_map)
            self.items.append((
                nn2.features_from_pieces(
                    pieces, nn2.WHITE, input_buckets, feature_channels,
                    full_threats, slider_xray_threats),
                nn2.features_from_pieces(
                    pieces, nn2.BLACK, input_buckets, feature_channels,
                    full_threats, slider_xray_threats),
                len(pieces),
                stm,
                float(row["score"]),
                float(row.get("wdl", 0.5)),
                phase_scale,
                source_map[source_name],
            ))

    @classmethod
    def from_jsonl(cls, path: str | Path, *,
                   limit: int = 0, skip: int = 0,
                   input_buckets: int = nn2.DEFAULT_N_KING_BUCKETS,
                   feature_channels: int = nn2.DEFAULT_N_FEATURE_CHANNELS,
                   full_threats: bool = False,
                   slider_xray_threats: bool = False,
                   ) -> "FenScoreDataset":
        rows = []
        with Path(path).open() as f:
            for i, line in enumerate(f):
                if i < skip:
                    continue
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if limit > 0 and len(rows) >= limit:
                    break
        return cls(
            rows,
            input_buckets=input_buckets,
            feature_channels=feature_channels,
            full_threats=full_threats,
            slider_xray_threats=slider_xray_threats)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        w, b, count, stm, score, wdl, phase_scale, source_id = self.items[idx]
        return (
            np.asarray(w, dtype=np.int64),
            np.asarray(b, dtype=np.int64),
            np.int64(count),
            np.int64(stm),
            np.float32(score),
            np.float32(wdl),
            np.float32(phase_scale),
            np.int64(source_id),
        )


class PackedFenScoreDataset:
    def __init__(self, path: str | Path, *,
                 limit: int = 0, skip: int = 0,
                 in_memory: bool = False) -> None:
        root = Path(path)
        mmap_mode = None if in_memory else "r"
        self.w_features = np.load(root / "white_features.npy", mmap_mode=mmap_mode)
        self.b_features = np.load(root / "black_features.npy", mmap_mode=mmap_mode)
        self.counts = np.load(root / "counts.npy", mmap_mode=mmap_mode)
        self.stms = np.load(root / "stm.npy", mmap_mode=mmap_mode)
        self.scores = np.load(root / "score.npy", mmap_mode=mmap_mode)
        self.wdls = np.load(root / "wdl.npy", mmap_mode=mmap_mode)
        self.phase_scales = np.load(root / "phase_scale.npy", mmap_mode=mmap_mode)
        source_id_path = root / "source_id.npy"
        self.source_ids = (
            np.load(source_id_path, mmap_mode=mmap_mode)
            if source_id_path.exists() else None)

        rows = len(self.counts)
        self.start = min(skip, rows)
        self.end = rows if limit <= 0 else min(rows, self.start + limit)

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, idx: int):
        real_idx = self.start + idx
        return (
            self.w_features[real_idx],
            self.b_features[real_idx],
            self.counts[real_idx],
            self.stms[real_idx],
            self.scores[real_idx],
            self.wdls[real_idx],
            self.phase_scales[real_idx],
            0 if self.source_ids is None else self.source_ids[real_idx],
        )


class BulletDataScoreDataset:
    def __init__(self, path: str | Path, *,
                 limit: int = 0, skip: int = 0,
                 input_buckets: int = nn2.DEFAULT_N_KING_BUCKETS,
                 feature_channels: int = nn2.DEFAULT_N_FEATURE_CHANNELS,
                 full_threats: bool = False,
                 slider_xray_threats: bool = False) -> None:
        self.path = Path(path)
        self.input_buckets = input_buckets
        self.feature_channels = feature_channels
        self.full_threats = full_threats
        self.slider_xray_threats = slider_xray_threats
        size = self.path.stat().st_size
        if size % bullet_format.RECORD_BYTES:
            raise ValueError(
                f"{self.path}: size {size} is not a multiple of "
                f"{bullet_format.RECORD_BYTES}")
        rows = size // bullet_format.RECORD_BYTES
        self.start = min(skip, rows)
        self.end = rows if limit <= 0 else min(rows, self.start + limit)
        self._data = np.memmap(self.path, dtype=np.uint8, mode="r")

    def __len__(self) -> int:
        return self.end - self.start

    @staticmethod
    def _trailing_zeros(value: int) -> int:
        if value == 0:
            raise ValueError("missing bit in BulletFormat record")
        return (value & -value).bit_length() - 1

    @staticmethod
    def _result_to_wdl(value: int) -> float:
        if value == 2:
            return 1.0
        if value == 1:
            return 0.5
        return 0.0

    def __getitem__(self, idx: int):
        real_idx = self.start + idx
        offset = real_idx * bullet_format.RECORD_BYTES
        raw = self._data[offset:offset + bullet_format.RECORD_BYTES].tobytes()
        occ, pieces_raw, score, result, _ksq, _opp_ksq, _extra = (
            BULLET_CHESSBOARD_STRUCT.unpack(raw))

        pieces: list[tuple[int, int, int]] = []
        remaining = int(occ)
        piece_idx = 0
        while remaining:
            square = self._trailing_zeros(remaining)
            remaining &= remaining - 1
            code = pieces_raw[piece_idx // 2]
            code = (code >> (4 * (piece_idx & 1))) & 0x0f
            color = nn2.BLACK if (code & 8) else nn2.WHITE
            piece_type = (code & 7) + 1
            # BulletFormat stores A1=0. Enyo uses h1=0, so mirror files.
            pieces.append((piece_type, color, square ^ 7))
            piece_idx += 1

        pieces.sort(key=lambda item: item[2])
        w_feats = nn2.features_from_pieces(
            pieces, nn2.WHITE, self.input_buckets, self.feature_channels,
            self.full_threats, self.slider_xray_threats)
        b_feats = nn2.features_from_pieces(
            pieces, nn2.BLACK, self.input_buckets, self.feature_channels,
            self.full_threats, self.slider_xray_threats)
        phase_scale = nn2.phase_scale_from_pieces(pieces)
        return (
            np.asarray(w_feats, dtype=np.int64),
            np.asarray(b_feats, dtype=np.int64),
            np.int64(len(pieces)),
            np.int64(nn2.WHITE),
            np.float32(score),
            np.float32(self._result_to_wdl(int(result))),
            np.float32(phase_scale),
            np.int64(0),
        )


def collate(batch):
    w_all, b_all = [], []
    w_offsets, b_offsets = [0], [0]
    stms, counts, scores, wdls, phase_scales, source_ids = [], [], [], [], [], []
    for w, b, count, stm, score, wdl, phase_scale, source_id in batch:
        w_all.append(w)
        b_all.append(b)
        w_offsets.append(w_offsets[-1] + len(w))
        b_offsets.append(b_offsets[-1] + len(b))
        counts.append(count)
        stms.append(stm)
        scores.append(score)
        wdls.append(wdl)
        phase_scales.append(phase_scale)
        source_ids.append(source_id)

    return (
        np.concatenate(w_all) if w_all else np.empty(0, dtype=np.int64),
        np.concatenate(b_all) if b_all else np.empty(0, dtype=np.int64),
        np.asarray(w_offsets[:-1], dtype=np.int64),
        np.asarray(b_offsets[:-1], dtype=np.int64),
        np.asarray(counts, dtype=np.int64),
        np.asarray(stms, dtype=np.int64),
        np.asarray(scores, dtype=np.float32),
        np.asarray(wdls, dtype=np.float32),
        np.asarray(phase_scales, dtype=np.float32),
        np.asarray(source_ids, dtype=np.int64),
    )


def collate_packed(batch):
    w_rows, b_rows, counts, stms, scores, wdls, phase_scales, source_ids = zip(
        *batch)
    w_padded = np.asarray(np.stack(w_rows), dtype=np.int64)
    b_padded = np.asarray(np.stack(b_rows), dtype=np.int64)
    counts_t = np.asarray(counts, dtype=np.int64)
    max_features = w_padded.shape[1]
    mask = np.arange(max_features)[None, :] < counts_t[:, None]
    offsets = np.concatenate((
        np.zeros(1, dtype=np.int64),
        np.cumsum(counts_t, dtype=np.int64)[:-1],
    ))
    return (
        w_padded[mask],
        b_padded[mask],
        offsets,
        offsets,
        counts_t,
        np.asarray(stms, dtype=np.int64),
        np.asarray(scores, dtype=np.float32),
        np.asarray(wdls, dtype=np.float32),
        np.asarray(phase_scales, dtype=np.float32),
        np.asarray(source_ids, dtype=np.int64),
    )


def count_rows(path: str | Path) -> int:
    p = Path(path)
    if p.is_dir():
        counts = np.load(p / "counts.npy", mmap_mode="r")
        return len(counts)
    if p.suffix == ".data":
        raise ValueError(
            f"deprecated BulletFormat extension .data is not accepted: {p}; "
            "rename it to .bullet"
        )
    if p.suffix == ".bullet":
        return bullet_format.record_count(p)

    n = 0
    with p.open() as handle:
        for line in handle:
            if line.strip():
                n += 1
    return n


def load_score_dataset(path: str | Path, *, limit: int = 0, skip: int = 0,
                       in_memory: bool = False,
                       input_buckets: int = nn2.DEFAULT_N_KING_BUCKETS,
                       feature_channels: int = nn2.DEFAULT_N_FEATURE_CHANNELS,
                       full_threats: bool = False,
                       slider_xray_threats: bool = False,
                       ) -> tuple[object, Callable]:
    p = Path(path)
    if p.is_dir():
        if full_threats or slider_xray_threats:
            raise ValueError("packed datasets do not include threat features")
        return PackedFenScoreDataset(
            p, limit=limit, skip=skip, in_memory=in_memory), collate_packed
    if p.suffix == ".data":
        raise ValueError(
            f"deprecated BulletFormat extension .data is not accepted: {p}; "
            "rename it to .bullet"
        )
    if p.suffix == ".bullet":
        return BulletDataScoreDataset(
            p,
            limit=limit,
            skip=skip,
            input_buckets=input_buckets,
            feature_channels=feature_channels,
            full_threats=full_threats,
            slider_xray_threats=slider_xray_threats), collate
    return FenScoreDataset.from_jsonl(
        p,
        limit=limit,
        skip=skip,
        input_buckets=input_buckets,
        feature_channels=feature_channels,
        full_threats=full_threats,
        slider_xray_threats=slider_xray_threats), collate
