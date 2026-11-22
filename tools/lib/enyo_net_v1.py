"""Enyo net container, format version 1 (architecture 24).

Layout (little-endian):
  0   char[8] magic "ENYONNUE"
  8   u32 format_version (1)
  12  u32 architecture (24)
  16  u32 metadata_size
  20  u32 payload_size
  24  u64 FNV-1a 64 of the payload
  32  u8 int_l1_shift, u8 int_l2_shift, u8[6] zero
  40  metadata: UTF-8 "key=value" lines (name, trained)
  ..  payload, architecture 24:
        input weights i16[24576][1024], input biases i16[1024],
        L1 weights i8[16][2048], L1 biases i32[16],
        L2 weights i8[32][16], L2 squared weights i8[32][16],
        L2 biases i32[32] (linear + squared),
        output weights i32[8][32], output biases i32[8], skip weights i32[8][16]
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"ENYONNUE"
FORMAT_VERSION = 1
ARCHITECTURE = 24
HEADER = struct.Struct("<8sIIIIQBB6s")
N_FEATURES = 32 * 12 * 64
N_HIDDEN = 1024
N_L1 = 2 * N_HIDDEN
N_L2 = 16
N_L3 = 32
N_BUCKETS = 8

SECTIONS = (
    ("input_weights", np.int16, (N_FEATURES, N_HIDDEN)),
    ("input_biases", np.int16, (N_HIDDEN,)),
    ("l1_weights", np.int8, (N_L2, N_L1)),
    ("l1_biases", np.int32, (N_L2,)),
    ("l2_weights", np.int8, (N_L3, N_L2)),
    ("l2_squared_weights", np.int8, (N_L3, N_L2)),
    ("l2_biases", np.int32, (N_L3,)),
    ("output_weights", np.int32, (N_BUCKETS, N_L3)),
    ("output_biases", np.int32, (N_BUCKETS,)),
    ("skip_weights", np.int32, (N_BUCKETS, N_L2)),
)
PAYLOAD_SIZE = sum(int(np.prod(shape)) * np.dtype(dtype).itemsize for _, dtype, shape in SECTIONS)


def fnv1a64(data: bytes) -> int:
    # numpy-vectorised would need a scan; plain loop over 50 MB is slow, so
    # hash 8-byte words (the engine does the same).
    h = 0xCBF29CE484222325
    words = np.frombuffer(data[: len(data) // 8 * 8], dtype="<u8")
    prime = 0x100000001B3
    mask = (1 << 64) - 1
    for w in words.tolist():
        h = ((h ^ w) * prime) & mask
    for b in data[len(data) // 8 * 8:]:
        h = ((h ^ b) * prime) & mask
    return h


@dataclass
class NetV1:
    int_l1_shift: int
    int_l2_shift: int
    metadata: dict[str, str]
    arrays: dict[str, np.ndarray]


def write(path: str | Path, net: NetV1) -> None:
    payload = b"".join(
        np.ascontiguousarray(net.arrays[name], dtype=dtype).reshape(shape).tobytes()
        for name, dtype, shape in SECTIONS)
    assert len(payload) == PAYLOAD_SIZE
    meta = "".join(f"{k}={v}\n" for k, v in net.metadata.items()).encode()
    header = HEADER.pack(MAGIC, FORMAT_VERSION, ARCHITECTURE, len(meta), len(payload),
                         fnv1a64(payload), net.int_l1_shift, net.int_l2_shift, bytes(6))
    Path(path).write_bytes(header + meta + payload)


def read(path: str | Path) -> NetV1:
    data = Path(path).read_bytes()
    magic, version, arch, meta_size, payload_size, checksum, l1s, l2s, _ = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError(f"{path}: not an Enyo net")
    if version != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported net format version {version}")
    if arch != ARCHITECTURE:
        raise ValueError(f"{path}: unsupported architecture {arch}")
    start = HEADER.size + meta_size
    payload = data[start:]
    if payload_size != PAYLOAD_SIZE or len(payload) != PAYLOAD_SIZE:
        raise ValueError(f"{path}: payload size mismatch")
    if fnv1a64(payload) != checksum:
        raise ValueError(f"{path}: checksum mismatch")
    meta = dict(line.split("=", 1) for line in data[HEADER.size:start].decode().splitlines() if "=" in line)
    arrays, off = {}, 0
    for name, dtype, shape in SECTIONS:
        n = int(np.prod(shape)) * np.dtype(dtype).itemsize
        arrays[name] = np.frombuffer(payload, dtype=dtype, count=int(np.prod(shape)), offset=off).reshape(shape).copy()
        off += n
    return NetV1(l1s, l2s, meta, arrays)


def from_v8_integer(net) -> NetV1:
    """Convert a loaded v8 integer-dense enyo_nnue.Net (32x12, 1024, 8 buckets)."""
    if not getattr(net, "integer_dense", False):
        raise ValueError("source net is not integer-dense")
    if (net.input_buckets, net.feature_channels, net.output_buckets, net.full_heads,
            net.full_threats or net.slider_xray_threats or net.pawn_pairs) != (32, 12, 8, False, False):
        raise ValueError("source net does not have the architecture-24 layout")
    def ints(a, dtype):
        a = np.asarray(a, dtype=np.float64)
        if not np.all(a == np.round(a)):
            raise ValueError("non-integer dense weight")
        info = np.iinfo(dtype)
        if a.min() < info.min or a.max() > info.max:
            raise ValueError("dense weight out of range")
        return a.astype(dtype)
    arrays = {
        "input_weights": np.asarray(net.input_weights, dtype=np.int16),
        "input_biases": np.asarray(net.input_biases, dtype=np.int16),
        "l1_weights": np.asarray(net.l1_weights, dtype=np.int8).reshape(N_L2, N_L1),
        "l1_biases": np.asarray(net.l1_biases, dtype=np.int32).reshape(N_L2),
        "l2_weights": ints(np.asarray(net.l2_weights).reshape(N_L3, N_L2), np.int8),
        "l2_squared_weights": ints(np.asarray(net.l2_squared_weights).reshape(N_L3, N_L2), np.int8),
        "l2_biases": ints(np.asarray(net.l2_biases).reshape(N_L3) + np.asarray(net.l2_squared_biases).reshape(N_L3), np.int32),
        "output_weights": ints(np.asarray(net.output_weights)[:, :N_L3], np.int32),
        "output_biases": ints(np.asarray(net.output_biases).reshape(N_BUCKETS), np.int32),
        "skip_weights": ints(np.asarray(net.l2_output_skip_weights).reshape(N_BUCKETS, N_L2), np.int32),
    }
    return NetV1(net.int_l1_shift, net.int_l2_shift, {}, arrays)


if __name__ == "__main__":
    import argparse
    import datetime
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lib import enyo_nnue
    ap = argparse.ArgumentParser(description="convert a v8 integer-dense Enyo net to format 1 / architecture 24")
    ap.add_argument("source"); ap.add_argument("target")
    ap.add_argument("--name"); ap.add_argument("--trained")
    a = ap.parse_args()
    out = from_v8_integer(enyo_nnue.load_net(a.source))
    trained = a.trained or datetime.datetime.fromtimestamp(Path(a.source).stat().st_mtime, datetime.UTC).strftime("%Y-%m-%dT%H:%MZ")
    out.metadata = {"name": a.name or Path(a.target).stem, "trained": trained}
    write(a.target, out)
    back = read(a.target)
    assert all(np.array_equal(back.arrays[k], out.arrays[k]) for k in out.arrays)
    print(f"wrote {a.target}: architecture {ARCHITECTURE}, {back.metadata}")
