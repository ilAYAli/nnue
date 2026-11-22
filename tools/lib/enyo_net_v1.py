"""Enyo net container, format version 1 (architectures 24 and 25).

Header (little-endian):
  0   char[8] magic "ENYONNUE"
  8   u32 format_version (1)
  12  u32 architecture (24 or 25)
  16  u32 metadata_size
  20  u32 payload_size
  24  u64 FNV-1a 64 of the payload bytes as stored (8-byte LE words, then tail bytes)
  32  u8 int_l1_shift, u8 int_l2_shift, u8[6] zero
  40  metadata: UTF-8 "key=value" lines (name, trained)

Payload, architecture 24:
  input weights i16[24576][1024], input biases i16[1024],
  L1 weights i8[16][2048], L1 biases i32[16],
  L2 weights i8[32][16], L2 squared weights i8[32][16], L2 biases i32[32] (linear + squared),
  output weights i32[8][32], output biases i32[8], skip weights i32[8][16]
Payload, architecture 25 (= 24 + Stockfish-exact FullThreats inputs):
  u32 leb_size + input weights i16[24576][1024] as signed LEB128,
  threat weights i8[59808][1024],
  then input biases .. skip weights exactly as architecture 24.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"ENYONNUE"
FORMAT_VERSION = 1
ARCHITECTURES = (24, 25)
HEADER = struct.Struct("<8sIIIIQBB6s")
N_FEATURES = 32 * 12 * 64
N_THREAT_FEATURES = 59_808
N_HIDDEN = 1024
N_L1 = 2 * N_HIDDEN
N_L2 = 16
N_L3 = 32
N_BUCKETS = 8

TAIL = (
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
INPUT = ("input_weights", np.int16, (N_FEATURES, N_HIDDEN))
THREATS = ("threat_weights", np.int8, (N_THREAT_FEATURES, N_HIDDEN))


def fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    prime = 0x100000001B3
    mask = (1 << 64) - 1
    cut = len(data) // 8 * 8
    for w in np.frombuffer(data[:cut], dtype="<u8").tolist():
        h = ((h ^ w) * prime) & mask
    for b in data[cut:]:
        h = ((h ^ b) * prime) & mask
    return h


def leb128_encode(values: np.ndarray) -> bytes:
    v = np.asarray(values, dtype=np.int64).ravel()
    n = np.where((v >= -64) & (v < 64), 1, np.where((v >= -8192) & (v < 8192), 2, 3))
    if v.min(initial=0) < -(1 << 20) or v.max(initial=0) >= (1 << 20):
        raise ValueError("value out of 3-byte LEB128 range")
    start = np.cumsum(n) - n
    out = np.zeros(int(n.sum()), dtype=np.uint8)
    out[start] = (v & 0x7F) | np.where(n > 1, 0x80, 0)
    m2 = n >= 2
    out[start[m2] + 1] = ((v[m2] >> 7) & 0x7F) | np.where(n[m2] > 2, 0x80, 0)
    m3 = n == 3
    out[start[m3] + 2] = (v[m3] >> 14) & 0x7F
    return out.tobytes()


def leb128_decode(data: bytes, count: int) -> np.ndarray:
    a = np.frombuffer(data, dtype=np.uint8)
    ends = np.flatnonzero((a & 0x80) == 0)
    if len(ends) != count or (len(a) and ends[-1] != len(a) - 1):
        raise ValueError("invalid LEB128 input weights")
    starts = np.concatenate(([0], ends[:-1] + 1))
    length = ends - starts + 1
    if length.max(initial=1) > 3:
        raise ValueError("invalid LEB128 input weights")
    value = (a[starts] & 0x7F).astype(np.int64)
    for k in (1, 2):
        m = length > k
        value[m] |= (a[starts[m] + k] & 0x7F).astype(np.int64) << (7 * k)
    sign = (a[ends] & 0x40) != 0
    value[sign] -= 1 << (7 * length[sign])
    return value


@dataclass
class NetV1:
    architecture: int
    int_l1_shift: int
    int_l2_shift: int
    metadata: dict[str, str]
    arrays: dict[str, np.ndarray]


def _raw(arrays, sections) -> bytes:
    return b"".join(np.ascontiguousarray(arrays[n], dtype=d).reshape(s).tobytes() for n, d, s in sections)


def payload_bytes(net: NetV1) -> bytes:
    if net.architecture == 24:
        return _raw(net.arrays, (INPUT,) + TAIL)
    if net.architecture == 25:
        iw = np.ascontiguousarray(net.arrays["input_weights"], dtype=np.int16).reshape(INPUT[2])
        leb = leb128_encode(iw)
        return struct.pack("<I", len(leb)) + leb + _raw(net.arrays, (THREATS,) + TAIL)
    raise ValueError(f"unsupported architecture {net.architecture}")


def write(path: str | Path, net: NetV1) -> None:
    payload = payload_bytes(net)
    meta = "".join(f"{k}={v}\n" for k, v in net.metadata.items()).encode()
    header = HEADER.pack(MAGIC, FORMAT_VERSION, net.architecture, len(meta), len(payload),
                         fnv1a64(payload), net.int_l1_shift, net.int_l2_shift, bytes(6))
    Path(path).write_bytes(header + meta + payload)


def _take(payload, off, sections, arrays):
    for name, dtype, shape in sections:
        count = int(np.prod(shape))
        arrays[name] = np.frombuffer(payload, dtype=dtype, count=count, offset=off).reshape(shape).copy()
        off += count * np.dtype(dtype).itemsize
    return off


def read(path: str | Path) -> NetV1:
    data = Path(path).read_bytes()
    magic, version, arch, meta_size, payload_size, checksum, l1s, l2s, _ = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError(f"{path}: not an Enyo net")
    if version != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported net format version {version}")
    if arch not in ARCHITECTURES:
        raise ValueError(f"{path}: unsupported architecture {arch}")
    start = HEADER.size + meta_size
    payload = data[start:]
    if len(payload) != payload_size:
        raise ValueError(f"{path}: payload size mismatch")
    if fnv1a64(payload) != checksum:
        raise ValueError(f"{path}: checksum mismatch")
    meta = dict(line.split("=", 1) for line in data[HEADER.size:start].decode().splitlines() if "=" in line)
    arrays: dict[str, np.ndarray] = {}
    if arch == 24:
        off = _take(payload, 0, (INPUT,) + TAIL, arrays)
    else:
        (leb_size,) = struct.unpack_from("<I", payload, 0)
        arrays["input_weights"] = leb128_decode(payload[4:4 + leb_size], int(np.prod(INPUT[2]))).astype(np.int16).reshape(INPUT[2])
        off = _take(payload, 4 + leb_size, (THREATS,) + TAIL, arrays)
    if off != len(payload):
        raise ValueError(f"{path}: payload size mismatch")
    return NetV1(arch, l1s, l2s, meta, arrays)


def _ints(a, dtype):
    a = np.asarray(a, dtype=np.float64)
    if not np.all(a == np.round(a)):
        raise ValueError("non-integer dense weight")
    info = np.iinfo(dtype)
    if a.min() < info.min or a.max() > info.max:
        raise ValueError("dense weight out of range")
    return a.astype(dtype)


def from_enyo_net(net) -> NetV1:
    """Convert a loaded integer-dense enyo_nnue.Net: v8 -> architecture 24, v12 (FullThreats) -> 25."""
    if not getattr(net, "integer_dense", False):
        raise ValueError("source net is not integer-dense")
    if (net.input_buckets, net.feature_channels, net.output_buckets, net.full_heads,
            net.slider_xray_threats, net.pawn_pairs) != (32, 12, 8, False, False, False):
        raise ValueError("source net does not have the architecture-24/25 layout")
    iw = np.asarray(net.input_weights, dtype=np.int16)
    arrays = {
        "input_weights": iw[:N_FEATURES],
        "input_biases": np.asarray(net.input_biases, dtype=np.int16),
        "l1_weights": np.asarray(net.l1_weights, dtype=np.int8).reshape(N_L2, N_L1),
        "l1_biases": np.asarray(net.l1_biases, dtype=np.int32).reshape(N_L2),
        "l2_weights": _ints(np.asarray(net.l2_weights).reshape(N_L3, N_L2), np.int8),
        "l2_squared_weights": _ints(np.asarray(net.l2_squared_weights).reshape(N_L3, N_L2), np.int8),
        "l2_biases": _ints(np.asarray(net.l2_biases).reshape(N_L3) + np.asarray(net.l2_squared_biases).reshape(N_L3), np.int32),
        "output_weights": _ints(np.asarray(net.output_weights)[:, :N_L3], np.int32),
        "output_biases": _ints(np.asarray(net.output_biases).reshape(N_BUCKETS), np.int32),
        "skip_weights": _ints(np.asarray(net.l2_output_skip_weights).reshape(N_BUCKETS, N_L2), np.int32),
    }
    arch = 24
    if net.full_threats:
        threats = iw[N_FEATURES:N_FEATURES + N_THREAT_FEATURES]
        if threats.shape != THREATS[2]:
            raise ValueError("unexpected threat-row count")
        clipped = int(np.count_nonzero((threats > 127) | (threats < -128)))
        if clipped:
            print(f"warning: clipped {clipped} of {threats.size} threat weights to int8")
        arrays["threat_weights"] = np.clip(threats, -128, 127).astype(np.int8)
        arch = 25
    return NetV1(arch, net.int_l1_shift, net.int_l2_shift, {}, arrays)


from_v8_integer = from_enyo_net  # backwards-compatible name


if __name__ == "__main__":
    import argparse
    import datetime
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lib import enyo_nnue
    ap = argparse.ArgumentParser(description="convert an integer-dense Enyo v8/v12 net to format 1 (architecture 24/25)")
    ap.add_argument("source"); ap.add_argument("target")
    ap.add_argument("--name"); ap.add_argument("--trained")
    a = ap.parse_args()
    out = from_enyo_net(enyo_nnue.load_net(a.source))
    trained = a.trained or datetime.datetime.fromtimestamp(Path(a.source).stat().st_mtime, datetime.UTC).strftime("%Y-%m-%dT%H:%MZ")
    out.metadata = {"name": a.name or Path(a.target).stem, "trained": trained}
    write(a.target, out)
    back = read(a.target)
    assert all(np.array_equal(back.arrays[k], out.arrays[k]) for k in out.arrays)
    print(f"wrote {a.target}: architecture {out.architecture}, {Path(a.target).stat().st_size} bytes, {back.metadata}")
