"""Self-play JSONL dataset for the NNUE trainer."""
from __future__ import annotations

import json
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from . import bullet_format
from . import enyo_nnue as nn2


BULLET_CHESSBOARD_STRUCT = struct.Struct("<Q16shBBB3s")
assert BULLET_CHESSBOARD_STRUCT.size == bullet_format.RECORD_BYTES


class FenScoreDataset(Dataset):
    def __init__(self, rows: Iterable[dict], *, with_threats: bool = False):
        self.with_threats = with_threats
        self.items: list[
            tuple[
                list[int], list[int], list[int] | None, list[int] | None,
                int, int, float, float, float, int,
            ]
        ] = []
        source_map: dict[str, int] = {}
        for row in rows:
            fen = row["fen"]
            pieces, stm = nn2.parse_fen(fen)
            pieces.sort(key=lambda item: item[2])
            w_feats = nn2.features_from_pieces(pieces, nn2.WHITE)
            b_feats = nn2.features_from_pieces(pieces, nn2.BLACK)
            w_threats = b_threats = None
            if with_threats:
                w_threats = nn2.threat_features_from_pieces(pieces, nn2.WHITE)
                b_threats = nn2.threat_features_from_pieces(pieces, nn2.BLACK)
            phase_scale = nn2.phase_scale_from_pieces(pieces)
            source_name = str(
                row.get("source_type") or row.get("teacher") or "unknown")
            if source_name not in source_map:
                source_map[source_name] = len(source_map)
            self.items.append((
                w_feats,
                b_feats,
                w_threats,
                b_threats,
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
                   with_threats: bool = False) -> "FenScoreDataset":
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
        return cls(rows, with_threats=with_threats)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        (w, b, w_threats, b_threats, count, stm, score, wdl, phase_scale,
         source_id) = self.items[idx]
        if self.with_threats:
            assert w_threats is not None
            assert b_threats is not None
            return (
                torch.tensor(w, dtype=torch.long),
                torch.tensor(b, dtype=torch.long),
                torch.tensor(w_threats, dtype=torch.long),
                torch.tensor(b_threats, dtype=torch.long),
                torch.tensor(count, dtype=torch.long),
                torch.tensor(stm, dtype=torch.long),
                torch.tensor(score, dtype=torch.float32),
                torch.tensor(wdl, dtype=torch.float32),
                torch.tensor(phase_scale, dtype=torch.float32),
                torch.tensor(source_id, dtype=torch.long),
            )
        return (
            torch.tensor(w, dtype=torch.long),
            torch.tensor(b, dtype=torch.long),
            torch.tensor(count, dtype=torch.long),
            torch.tensor(stm, dtype=torch.long),
            torch.tensor(score, dtype=torch.float32),
            torch.tensor(wdl, dtype=torch.float32),
            torch.tensor(phase_scale, dtype=torch.float32),
            torch.tensor(source_id, dtype=torch.long),
        )


class PackedFenScoreDataset(Dataset):
    def __init__(self, path: str | Path, *,
                 limit: int = 0, skip: int = 0,
                 in_memory: bool = False,
                 with_threats: bool = False) -> None:
        root = Path(path)
        self.with_threats = with_threats
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
        self.w_threat_features = self.b_threat_features = self.threat_counts = None
        if with_threats:
            self.w_threat_features = np.load(
                root / "white_threat_features.npy", mmap_mode=mmap_mode)
            self.b_threat_features = np.load(
                root / "black_threat_features.npy", mmap_mode=mmap_mode)
            self.threat_counts = np.load(root / "threat_counts.npy",
                                         mmap_mode=mmap_mode)

        rows = len(self.counts)
        self.start = min(skip, rows)
        self.end = rows if limit <= 0 else min(rows, self.start + limit)

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, idx: int):
        real_idx = self.start + idx
        if self.with_threats:
            assert self.w_threat_features is not None
            assert self.b_threat_features is not None
            assert self.threat_counts is not None
            return (
                self.w_features[real_idx],
                self.b_features[real_idx],
                self.w_threat_features[real_idx],
                self.b_threat_features[real_idx],
                self.counts[real_idx],
                self.threat_counts[real_idx],
                self.stms[real_idx],
                self.scores[real_idx],
                self.wdls[real_idx],
                self.phase_scales[real_idx],
                0 if self.source_ids is None else self.source_ids[real_idx],
            )
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


class BulletDataScoreDataset(Dataset):
    def __init__(self, path: str | Path, *,
                 limit: int = 0, skip: int = 0,
                 with_threats: bool = False) -> None:
        self.path = Path(path)
        self.with_threats = with_threats
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
        w_feats = nn2.features_from_pieces(pieces, nn2.WHITE)
        b_feats = nn2.features_from_pieces(pieces, nn2.BLACK)
        phase_scale = nn2.phase_scale_from_pieces(pieces)
        if self.with_threats:
            return (
                torch.tensor(w_feats, dtype=torch.long),
                torch.tensor(b_feats, dtype=torch.long),
                torch.tensor(
                    nn2.threat_features_from_pieces(pieces, nn2.WHITE),
                    dtype=torch.long),
                torch.tensor(
                    nn2.threat_features_from_pieces(pieces, nn2.BLACK),
                    dtype=torch.long),
                torch.tensor(len(pieces), dtype=torch.long),
                torch.tensor(nn2.WHITE, dtype=torch.long),
                torch.tensor(float(score), dtype=torch.float32),
                torch.tensor(self._result_to_wdl(int(result)), dtype=torch.float32),
                torch.tensor(phase_scale, dtype=torch.float32),
                torch.tensor(0, dtype=torch.long),
            )
        return (
            torch.tensor(w_feats, dtype=torch.long),
            torch.tensor(b_feats, dtype=torch.long),
            torch.tensor(len(pieces), dtype=torch.long),
            torch.tensor(nn2.WHITE, dtype=torch.long),
            torch.tensor(float(score), dtype=torch.float32),
            torch.tensor(self._result_to_wdl(int(result)), dtype=torch.float32),
            torch.tensor(phase_scale, dtype=torch.float32),
            torch.tensor(0, dtype=torch.long),
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
        torch.cat(w_all) if w_all else torch.empty(0, dtype=torch.long),
        torch.cat(b_all) if b_all else torch.empty(0, dtype=torch.long),
        torch.tensor(w_offsets[:-1], dtype=torch.long),
        torch.tensor(b_offsets[:-1], dtype=torch.long),
        torch.stack(counts),
        torch.stack(stms),
        torch.stack(scores),
        torch.stack(wdls),
        torch.stack(phase_scales),
        torch.stack(source_ids),
    )


def collate_with_threats(batch):
    w_all, b_all = [], []
    wt_all, bt_all = [], []
    w_offsets, b_offsets = [0], [0]
    wt_offsets, bt_offsets = [0], [0]
    stms, counts, scores, wdls, phase_scales, source_ids = [], [], [], [], [], []
    for (w, b, wt, bt, count, stm, score, wdl, phase_scale,
         source_id) in batch:
        w_all.append(w)
        b_all.append(b)
        wt_all.append(wt)
        bt_all.append(bt)
        w_offsets.append(w_offsets[-1] + len(w))
        b_offsets.append(b_offsets[-1] + len(b))
        wt_offsets.append(wt_offsets[-1] + len(wt))
        bt_offsets.append(bt_offsets[-1] + len(bt))
        counts.append(count)
        stms.append(stm)
        scores.append(score)
        wdls.append(wdl)
        phase_scales.append(phase_scale)
        source_ids.append(source_id)

    return (
        torch.cat(w_all) if w_all else torch.empty(0, dtype=torch.long),
        torch.cat(b_all) if b_all else torch.empty(0, dtype=torch.long),
        torch.tensor(w_offsets[:-1], dtype=torch.long),
        torch.tensor(b_offsets[:-1], dtype=torch.long),
        torch.cat(wt_all) if wt_all else torch.empty(0, dtype=torch.long),
        torch.cat(bt_all) if bt_all else torch.empty(0, dtype=torch.long),
        torch.tensor(wt_offsets[:-1], dtype=torch.long),
        torch.tensor(bt_offsets[:-1], dtype=torch.long),
        torch.stack(counts),
        torch.stack(stms),
        torch.stack(scores),
        torch.stack(wdls),
        torch.stack(phase_scales),
        torch.stack(source_ids),
    )


def collate_packed(batch):
    w_rows, b_rows, counts, stms, scores, wdls, phase_scales, source_ids = zip(
        *batch)
    w_padded = torch.as_tensor(np.stack(w_rows), dtype=torch.long)
    b_padded = torch.as_tensor(np.stack(b_rows), dtype=torch.long)
    counts_t = torch.as_tensor(np.asarray(counts), dtype=torch.long)
    max_features = w_padded.shape[1]
    mask = (torch.arange(max_features).unsqueeze(0)
            < counts_t.unsqueeze(1))
    offsets = torch.cat((
        torch.zeros(1, dtype=torch.long),
        torch.cumsum(counts_t, dim=0)[:-1],
    ))
    return (
        w_padded[mask],
        b_padded[mask],
        offsets,
        offsets,
        counts_t,
        torch.as_tensor(np.asarray(stms), dtype=torch.long),
        torch.as_tensor(np.asarray(scores), dtype=torch.float32),
        torch.as_tensor(np.asarray(wdls), dtype=torch.float32),
        torch.as_tensor(np.asarray(phase_scales), dtype=torch.float32),
        torch.as_tensor(np.asarray(source_ids), dtype=torch.long),
    )


def collate_packed_with_threats(batch):
    (w_rows, b_rows, wt_rows, bt_rows, counts, threat_counts, stms, scores,
     wdls, phase_scales, source_ids) = zip(*batch)
    w_padded = torch.as_tensor(np.stack(w_rows), dtype=torch.long)
    b_padded = torch.as_tensor(np.stack(b_rows), dtype=torch.long)
    wt_padded = torch.as_tensor(np.stack(wt_rows), dtype=torch.long)
    bt_padded = torch.as_tensor(np.stack(bt_rows), dtype=torch.long)
    counts_t = torch.as_tensor(np.asarray(counts), dtype=torch.long)
    threat_counts_t = torch.as_tensor(np.asarray(threat_counts), dtype=torch.long)
    max_features = w_padded.shape[1]
    max_threats = wt_padded.shape[1]
    mask = (torch.arange(max_features).unsqueeze(0)
            < counts_t.unsqueeze(1))
    threat_mask = (torch.arange(max_threats).unsqueeze(0)
                   < threat_counts_t.unsqueeze(1))
    offsets = torch.cat((
        torch.zeros(1, dtype=torch.long),
        torch.cumsum(counts_t, dim=0)[:-1],
    ))
    threat_offsets = torch.cat((
        torch.zeros(1, dtype=torch.long),
        torch.cumsum(threat_counts_t, dim=0)[:-1],
    ))
    return (
        w_padded[mask],
        b_padded[mask],
        offsets,
        offsets,
        wt_padded[threat_mask],
        bt_padded[threat_mask],
        threat_offsets,
        threat_offsets,
        counts_t,
        torch.as_tensor(np.asarray(stms), dtype=torch.long),
        torch.as_tensor(np.asarray(scores), dtype=torch.float32),
        torch.as_tensor(np.asarray(wdls), dtype=torch.float32),
        torch.as_tensor(np.asarray(phase_scales), dtype=torch.float32),
        torch.as_tensor(np.asarray(source_ids), dtype=torch.long),
    )


def count_rows(path: str | Path) -> int:
    p = Path(path)
    if p.is_dir():
        counts = np.load(p / "counts.npy", mmap_mode="r")
        return len(counts)
    if p.suffix in {".bullet", ".data"}:
        return bullet_format.record_count(p)

    n = 0
    with p.open() as handle:
        for line in handle:
            if line.strip():
                n += 1
    return n


def load_score_dataset(path: str | Path, *, limit: int = 0, skip: int = 0,
                       in_memory: bool = False,
                       with_threats: bool = False,
                       ) -> tuple[Dataset, Callable]:
    p = Path(path)
    if p.is_dir():
        return PackedFenScoreDataset(
            p, limit=limit, skip=skip, in_memory=in_memory,
            with_threats=with_threats), (
                collate_packed_with_threats if with_threats else collate_packed)
    if p.suffix in {".bullet", ".data"}:
        return BulletDataScoreDataset(
            p, limit=limit, skip=skip, with_threats=with_threats), (
                collate_with_threats if with_threats else collate)
    return FenScoreDataset.from_jsonl(
        p, limit=limit, skip=skip, with_threats=with_threats), (
            collate_with_threats if with_threats else collate)
