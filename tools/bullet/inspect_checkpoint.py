#!/usr/bin/env python3
"""Inspect the experimental Bullet spike checkpoint layout."""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Segment:
    name: str
    dtype: str
    shape: list[int]
    offset: int
    bytes: int
    count: int


TYPE_SIZE = {
    "i8": 1,
    "i16": 2,
    "f32": 4,
}


def product(values: list[int]) -> int:
    out = 1
    for value in values:
        out *= value
    return out


def add_segment(
    segments: list[Segment],
    name: str,
    dtype: str,
    shape: list[int],
    offset: int,
) -> int:
    count = product(shape)
    size = count * TYPE_SIZE[dtype]
    segments.append(Segment(name, dtype, shape, offset, size, count))
    return offset + size


def checkpoint_file(path: Path) -> Path:
    if path.is_dir():
        path = path / "quantised.bin"
    if not path.exists():
        raise SystemExit(f"missing checkpoint file: {path}")
    return path


def expected_segments(
    *,
    hidden: int,
    l2: int,
    input_buckets: int,
    output_buckets: int,
) -> list[Segment]:
    offset = 0
    segments: list[Segment] = []
    offset = add_segment(
        segments,
        "l0w",
        "i16",
        [hidden, 768 * input_buckets],
        offset,
    )
    offset = add_segment(segments, "l0b", "i16", [hidden], offset)
    offset = add_segment(
        segments,
        "l1w",
        "i8",
        [output_buckets * l2, hidden],
        offset,
    )
    offset = add_segment(segments, "l1b", "f32", [output_buckets * l2], offset)
    offset = add_segment(
        segments,
        "l2w",
        "f32",
        [output_buckets * 32, l2],
        offset,
    )
    offset = add_segment(segments, "l2b", "f32", [output_buckets * 32], offset)
    offset = add_segment(
        segments,
        "l3w",
        "f32",
        [output_buckets, 32],
        offset,
    )
    add_segment(segments, "l3b", "f32", [output_buckets], offset)
    return segments


def sample_segment(data: bytes, segment: Segment) -> str:
    start = segment.offset
    if segment.dtype == "i8":
        return ", ".join(str(x) for x in data[start : start + 8])
    if segment.dtype == "i16":
        values = struct.unpack_from("<" + "h" * min(8, segment.count), data, start)
        return ", ".join(str(x) for x in values)
    values = struct.unpack_from("<" + "f" * min(4, segment.count), data, start)
    return ", ".join(f"{x:.6g}" for x in values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the quantised.bin layout from tools/bullet/spike_trainer."
    )
    parser.add_argument("checkpoint", type=Path, help="Checkpoint directory or quantised.bin")
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--l2", type=int, default=16)
    parser.add_argument("--input-buckets", type=int, default=10)
    parser.add_argument("--output-buckets", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = checkpoint_file(args.checkpoint).expanduser()
    data = path.read_bytes()
    segments = expected_segments(
        hidden=args.hidden,
        l2=args.l2,
        input_buckets=args.input_buckets,
        output_buckets=args.output_buckets,
    )
    expected = segments[-1].offset + segments[-1].bytes
    padding = len(data) - expected
    ok = padding >= 0 and data[expected:] == (b"bullet" * ((padding + 5) // 6))[:padding]

    if args.json:
        print(
            json.dumps(
                {
                    "path": str(path),
                    "size": len(data),
                    "expected_without_padding": expected,
                    "padding": padding,
                    "padding_ok": ok,
                    "segments": [asdict(segment) for segment in segments],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if ok else 1

    print(f"path: {path}")
    print(f"size: {len(data)} bytes")
    print(f"expected without padding: {expected} bytes")
    print(f"padding: {padding} bytes ({'ok' if ok else 'unexpected'})")
    print()
    print("segments:")
    for segment in segments:
        print(
            f"  {segment.name:3} offset={segment.offset:8} "
            f"bytes={segment.bytes:8} dtype={segment.dtype:3} "
            f"shape={segment.shape} sample=[{sample_segment(data, segment)}]"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
