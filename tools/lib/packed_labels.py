"""Compact labeled shard helpers for Enyo NNUE training data."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from . import enyo_nnue as nn2


FORMAT = "enyo-packed-label-shard-v1"


def source_name(row: dict) -> str:
    return str(row.get("source_type") or row.get("teacher") or "unknown")


def row_to_item(row: dict, source_map: dict[str, int]) -> tuple[
    list[int], list[int], int, int, float, float, float, int
]:
    pieces, stm = nn2.parse_fen(str(row["fen"]))
    pieces.sort(key=lambda item: item[2])
    w_features = nn2.features_from_pieces(pieces, nn2.WHITE)
    b_features = nn2.features_from_pieces(pieces, nn2.BLACK)
    name = source_name(row)
    if name not in source_map:
        source_map[name] = len(source_map)
    return (
        w_features,
        b_features,
        len(w_features),
        stm,
        float(row["score"]),
        float(row.get("wdl", 0.5)),
        nn2.phase_scale_from_pieces(pieces),
        source_map[name],
    )


def write_shard(path: str | Path, rows: Iterable[dict], *,
                max_features: int) -> dict[str, object]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_map: dict[str, int] = {}
    items = [row_to_item(row, source_map) for row in rows]
    row_count = len(items)
    shape = (row_count, max_features)

    w = np.zeros(shape, dtype=np.uint16)
    b = np.zeros(shape, dtype=np.uint16)
    counts = np.zeros(row_count, dtype=np.uint8)
    stms = np.zeros(row_count, dtype=np.uint8)
    scores = np.zeros(row_count, dtype=np.float32)
    wdls = np.zeros(row_count, dtype=np.float32)
    phase_scales = np.zeros(row_count, dtype=np.float32)
    source_ids = np.zeros(row_count, dtype=np.uint16)
    max_seen = 0

    for index, (
        w_features, b_features, count, stm, score, wdl, phase_scale, source_id
    ) in enumerate(items):
        if len(w_features) > max_features:
            raise ValueError(
                f"row {index}: {len(w_features)} features > max {max_features}"
            )
        n = len(w_features)
        max_seen = max(max_seen, n)
        w[index, :n] = np.asarray(w_features, dtype=np.uint16)
        b[index, :n] = np.asarray(b_features, dtype=np.uint16)
        counts[index] = count
        stms[index] = stm
        scores[index] = score
        wdls[index] = wdl
        phase_scales[index] = phase_scale
        source_ids[index] = source_id

    np.savez(
        output,
        format=np.asarray(FORMAT),
        white_features=w,
        black_features=b,
        counts=counts,
        stm=stms,
        score=scores,
        wdl=wdls,
        phase_scale=phase_scales,
        source_id=source_ids,
        source_map_json=np.asarray(json.dumps(source_map, sort_keys=True)),
    )
    return {
        "format": FORMAT,
        "output": str(output),
        "rows": row_count,
        "max_features": max_features,
        "max_features_seen": max_seen,
        "source_map": source_map,
    }


def load_source_map(npz: np.lib.npyio.NpzFile) -> dict[str, int]:
    raw = str(npz["source_map_json"].item())
    return {str(name): int(source_id) for name, source_id in json.loads(raw).items()}


def assert_format(npz: np.lib.npyio.NpzFile, path: Path) -> None:
    found = str(npz["format"].item()) if "format" in npz.files else ""
    if found != FORMAT:
        raise ValueError(f"{path}: unsupported packed shard format {found!r}")
