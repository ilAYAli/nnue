"""Enyo net container, format version 1 (architectures 24, 25 and 26).

Header (little-endian):
  0   char[8] magic "ENYONNUE"
  8   u32 format_version (1)
  12  u32 architecture (24, 25 or 26)
  16  u32 metadata_size
  20  u32 payload_size
  24  u64 FNV-1a 64 of the payload bytes as stored (8-byte LE words, then tail bytes)
  32  architectures 24/25: u8 int_l1_shift, u8 int_l2_shift, u8[6] zero
      architecture 26: u8[8] zero (reserved)
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
Payload, architecture 26 (layer stacks; the engine's native architecture):
  layout block, u32 each: input_set, threats, hidden (H), psqt_buckets (P), stacks,
    fc0_out, fc1_out, bucket_scheme. The engine loads only
    input_set=0 (HalfKAv2_hm as Stockfish, F = 22528 rows) or 1 (Enyo HalfKA 32 king
    buckets x 12 planes, the architecture-24/25 layout, F = 24576 rows; engine
    src/nnue/enyo/enyo_halfka.hpp), threats=1 (Stockfish FullThreats), H=1024, P=8,
    stacks=8, fc0_out=32 or 64 (incl. the passthrough pair), fc1_out=32,
    bucket_scheme=0 ((pieces - 1) / 4) or 1 (game phase: min(7, phase * 8 / 25),
    phase = min(24, N + B + 2 R + 4 Q) over both sides). Only input_set 0,
    fc0_out 32, bucket_scheme 0 has a Stockfish .nnue form.
  feature biases  i16[H]
  HalfKA weights  leb i16[F][H]
  HalfKA PSQT     leb i32[F][P]
  threat weights  i8[59808][H]
  threat PSQT     leb i32[59808][P]
  per stack: fc0 biases i32[fc0_out], fc0 weights i8[fc0_out][H],
             fc1 biases i32[fc1_out], fc1 weights i8[fc1_out][2 * fc0_out],
             fc2 bias i32, fc2 weights i8[2 * fc0_out + 2 * fc1_out]
  "leb" is u32 byte count + one minimal signed LEB128 varint per value.
  Values and quantisation are those of the Stockfish layer-stacks .nnue
  (HalfKAv2_hm + FullThreats, sfnnv15), so an input_set-0 net converts
  losslessly both ways; input_set 1 has no .nnue form (the trainer writes it).

Command line:
  enyo_net_v1.py SOURCE TARGET [--name N] [--trained T] [--description D]
    SOURCE Stockfish layer-stacks .nnue (current variant) -> TARGET architecture 26
    SOURCE architecture-26 net                            -> TARGET Stockfish .nnue
    SOURCE integer-dense Enyo v8/v12 net                  -> TARGET architecture 24/25
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = b"ENYONNUE"
FORMAT_VERSION = 1
ARCHITECTURES = (24, 25, 26)
LAYER_STACKS = 26
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

# Architecture 26.
LAYOUT_FIELDS = ("input_set", "threats", "hidden", "psqt_buckets", "stacks",
                 "fc0_out", "fc1_out", "bucket_scheme")
LAYOUT = struct.Struct("<8I")
SUPPORTED_LAYOUT = {"input_set": 0, "threats": 1, "hidden": 1024, "psqt_buckets": 8,
                    "stacks": 8, "fc0_out": 32, "fc1_out": 32, "bucket_scheme": 0}
# Fields that may take other values than SUPPORTED_LAYOUT's.
LAYOUT_CHOICES = {"input_set": (0, 1), "fc0_out": (32, 64), "bucket_scheme": (0, 1)}
HALFKA_V2_FEATURES = 22_528
# King rows per input_set: HalfKAv2_hm, Enyo HalfKA 32x12.
INPUT_SET_ROWS = {0: HALFKA_V2_FEATURES, 1: N_FEATURES}
LEB = "leb"
RAW = "raw"


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


def leb128_encode(values: np.ndarray, max_bytes: int = 3) -> bytes:
    """Minimal signed LEB128, one varint per value."""
    v = np.asarray(values, dtype=np.int64).ravel()
    limit = 1 << (7 * max_bytes - 1)
    if v.min(initial=0) < -limit or v.max(initial=0) >= limit:
        raise ValueError(f"value out of {max_bytes}-byte LEB128 range")
    n = np.ones(v.size, dtype=np.int64)
    for k in range(1, max_bytes):
        bound = 1 << (7 * k - 1)
        n += (v < -bound) | (v >= bound)
    start = np.cumsum(n) - n
    out = np.zeros(int(n.sum()), dtype=np.uint8)
    for k in range(max_bytes):
        m = n > k
        out[start[m] + k] = ((v[m] >> (7 * k)) & 0x7F) | np.where(n[m] > k + 1, 0x80, 0)
    return out.tobytes()


def leb128_decode(data: bytes, count: int, max_bytes: int = 3, what: str = "input weights") -> np.ndarray:
    """Strict: exactly `count` varints of at most `max_bytes` bytes filling `data`."""
    a = np.frombuffer(data, dtype=np.uint8)
    ends = np.flatnonzero((a & 0x80) == 0)
    if len(ends) != count or (len(a) and (not len(ends) or ends[-1] != len(a) - 1)):
        raise ValueError(f"invalid LEB128 {what}")
    starts = np.concatenate(([0], ends[:-1] + 1))
    length = ends - starts + 1
    if length.max(initial=1) > max_bytes:
        raise ValueError(f"invalid LEB128 {what}")
    value = (a[starts] & 0x7F).astype(np.int64)
    for k in range(1, max_bytes):
        m = length > k
        value[m] |= (a[starts[m] + k] & 0x7F).astype(np.int64) << (7 * k)
    sign = (a[ends] & 0x40) != 0
    value[sign] -= np.left_shift(np.int64(1), 7 * length[sign])
    return value


def _leb_bytes(dtype) -> int:
    return (8 * np.dtype(dtype).itemsize + 6) // 7


def _checked(values: np.ndarray, dtype, what: str) -> np.ndarray:
    info = np.iinfo(dtype)
    if values.size and (values.min() < info.min or values.max() > info.max):
        raise ValueError(f"{what} out of {np.dtype(dtype).name} range")
    return values.astype(dtype)


@dataclass
class NetV1:
    architecture: int
    int_l1_shift: int
    int_l2_shift: int
    metadata: dict[str, str]
    arrays: dict[str, np.ndarray]
    layout: dict[str, int] = field(default_factory=dict)


def _raw(arrays, sections) -> bytes:
    return b"".join(np.ascontiguousarray(arrays[n], dtype=d).reshape(s).tobytes() for n, d, s in sections)


# ---------------------------------------------------------------------------
# Architecture 26 sections

def feature_sections(layout: dict[str, int]):
    h, p = layout["hidden"], layout["psqt_buckets"]
    rows = INPUT_SET_ROWS[layout["input_set"]]
    return (
        ("ft_biases", np.int16, (h,), RAW),
        ("halfka_weights", np.int16, (rows, h), LEB),
        ("halfka_psqt", np.int32, (rows, p), LEB),
        ("threat_weights", np.int8, (N_THREAT_FEATURES, h), RAW),
        ("threat_psqt", np.int32, (N_THREAT_FEATURES, p), LEB),
    )


def stack_sections(layout: dict[str, int]):
    h, f0, f1 = layout["hidden"], layout["fc0_out"], layout["fc1_out"]
    return (
        ("fc0_biases", np.int32, (f0,)),
        ("fc0_weights", np.int8, (f0, h)),
        ("fc1_biases", np.int32, (f1,)),
        ("fc1_weights", np.int8, (f1, 2 * f0)),
        ("fc2_bias", np.int32, (1,)),
        ("fc2_weights", np.int8, (2 * f0 + 2 * f1,)),
    )


def check_layout(layout: dict[str, int]) -> None:
    fixed = {k: v for k, v in layout.items() if k not in LAYOUT_CHOICES}
    if (any(layout[k] not in choices for k, choices in LAYOUT_CHOICES.items())
            or fixed != {k: v for k, v in SUPPORTED_LAYOUT.items() if k not in LAYOUT_CHOICES}):
        got = " ".join(f"{k}={layout[k]}" for k in LAYOUT_FIELDS)
        want = " ".join(f"{k}={'|'.join(map(str, LAYOUT_CHOICES[k]))}" if k in LAYOUT_CHOICES
                        else f"{k}={SUPPORTED_LAYOUT[k]}" for k in LAYOUT_FIELDS)
        raise ValueError(f"unsupported layout {got} (supported: {want})")


def _layer_stacks_payload(net: NetV1) -> bytes:
    layout = net.layout
    check_layout(layout)
    a = net.arrays
    parts = [LAYOUT.pack(*(layout[k] for k in LAYOUT_FIELDS))]
    for name, dtype, shape, encoding in feature_sections(layout):
        values = np.ascontiguousarray(a[name], dtype=dtype).reshape(shape)
        if encoding == LEB:
            leb = leb128_encode(values, _leb_bytes(dtype))
            parts += [struct.pack("<I", len(leb)), leb]
        else:
            parts.append(values.tobytes())
    stacks = layout["stacks"]
    for s in range(stacks):
        for name, dtype, shape in stack_sections(layout):
            stacked = np.ascontiguousarray(a[name], dtype=dtype).reshape((stacks,) + shape)
            parts.append(stacked[s].tobytes())
    return b"".join(parts)


def _read_layer_stacks(payload: bytes, path) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    if len(payload) < LAYOUT.size:
        raise ValueError(f"{path}: payload too small for the layout block")
    layout = dict(zip(LAYOUT_FIELDS, LAYOUT.unpack_from(payload, 0)))
    check_layout(layout)
    off = LAYOUT.size
    arrays: dict[str, np.ndarray] = {}
    for name, dtype, shape, encoding in feature_sections(layout):
        count = int(np.prod(shape))
        if encoding == LEB:
            (size,) = struct.unpack_from("<I", payload, off)
            off += 4
            values = leb128_decode(payload[off:off + size], count, _leb_bytes(dtype), name)
            arrays[name] = _checked(values, dtype, name).reshape(shape)
            off += size
        else:
            arrays[name] = np.frombuffer(payload, dtype=dtype, count=count, offset=off).reshape(shape).copy()
            off += count * np.dtype(dtype).itemsize
    stacks = layout["stacks"]
    sections = stack_sections(layout)
    for name, dtype, shape in sections:
        arrays[name] = np.empty((stacks,) + shape, dtype=dtype)
    for s in range(stacks):
        for name, dtype, shape in sections:
            count = int(np.prod(shape))
            arrays[name][s] = np.frombuffer(payload, dtype=dtype, count=count, offset=off).reshape(shape)
            off += count * np.dtype(dtype).itemsize
    if off != len(payload):
        raise ValueError(f"{path}: payload size mismatch")
    return layout, arrays


# ---------------------------------------------------------------------------
# Stockfish layer-stacks .nnue (current variant: HalfKAv2_hm + FullThreats)

SF_VERSION = 0x6A448AFA
SF_LEB128_MAGIC = b"COMPRESSED_LEB128"
_MASK32 = 0xFFFFFFFF


def _sf_hashes(layout: dict[str, int]) -> tuple[int, int, int]:
    """(network hash, feature transformer hash, per-stack hash) as Stockfish writes them."""
    hidden2 = 2 * layout["hidden"]
    ft = (((0x2E6B9D04 << 1) | (0x2E6B9D04 >> 31)) & _MASK32) ^ 0x7F234CB8 ^ hidden2
    fc = 0xEC42E90D ^ hidden2
    for outputs in (layout["fc0_out"], layout["fc1_out"], 1):
        fc = ((0xCC03DAE4 + outputs) ^ (fc >> 1) ^ (fc << 31)) & _MASK32
        if outputs != 1:
            fc = (fc + 0x538D24C7) & _MASK32
    return ft ^ fc, ft, fc


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.off = 0

    def take(self, n: int, what: str) -> bytes:
        if self.off + n > len(self.data):
            raise ValueError(f"truncated .nnue in {what}")
        out = self.data[self.off:self.off + n]
        self.off += n
        return out

    def u32(self, what: str) -> int:
        return struct.unpack("<I", self.take(4, what))[0]

    def expect(self, value: int, what: str) -> None:
        got = self.u32(what)
        if got != value:
            raise ValueError(f"{what}: expected {value:08x}, got {got:08x} "
                             "(only the HalfKAv2_hm + FullThreats layer-stacks .nnue converts)")

    def array(self, dtype, shape, what: str) -> np.ndarray:
        count = int(np.prod(shape))
        raw = self.take(count * np.dtype(dtype).itemsize, what)
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()

    def leb(self, dtype, shape, what: str) -> np.ndarray:
        if self.take(len(SF_LEB128_MAGIC), what) != SF_LEB128_MAGIC:
            raise ValueError(f"{what}: missing LEB128 magic")
        size = self.u32(what)
        values = leb128_decode(self.take(size, what), int(np.prod(shape)), _leb_bytes(dtype), what)
        return _checked(values, dtype, what).reshape(shape)


def _sf_leb(values: np.ndarray, dtype) -> bytes:
    leb = leb128_encode(values, _leb_bytes(dtype))
    return SF_LEB128_MAGIC + struct.pack("<I", len(leb)) + leb


def from_nnue(data: bytes, metadata: dict[str, str] | None = None) -> tuple[NetV1, str]:
    """Stockfish .nnue bytes -> (architecture-26 net, the .nnue description)."""
    layout = dict(SUPPORTED_LAYOUT)
    network_hash, ft_hash, fc_hash = _sf_hashes(layout)
    h, p = layout["hidden"], layout["psqt_buckets"]
    r = _Reader(data)
    r.expect(SF_VERSION, "version")
    r.expect(network_hash, "network hash")
    description = r.take(r.u32("description size"), "description").decode("utf-8", "surrogateescape")
    r.expect(ft_hash, "feature transformer hash")
    a: dict[str, np.ndarray] = {}
    a["ft_biases"] = r.leb(np.int16, (h,), "feature biases")
    a["threat_weights"] = r.array(np.int8, (N_THREAT_FEATURES, h), "threat weights")
    a["threat_psqt"] = r.leb(np.int32, (N_THREAT_FEATURES, p), "threat PSQT")
    a["halfka_weights"] = r.leb(np.int16, (HALFKA_V2_FEATURES, h), "HalfKA weights")
    a["halfka_psqt"] = r.leb(np.int32, (HALFKA_V2_FEATURES, p), "HalfKA PSQT")
    stacks = layout["stacks"]
    sections = stack_sections(layout)
    for name, dtype, shape in sections:
        a[name] = np.empty((stacks,) + shape, dtype=dtype)
    for s in range(stacks):
        r.expect(fc_hash, "layer stack hash")
        for name, dtype, shape in sections:
            a[name][s] = r.array(dtype, shape, name)
    if r.off != len(data):
        raise ValueError(f"{len(data) - r.off} trailing bytes after the .nnue payload")
    return NetV1(LAYER_STACKS, 0, 0, dict(metadata or {}), a, layout), description


def to_nnue(net: NetV1, description: str) -> bytes:
    """Architecture-26 net -> Stockfish .nnue bytes."""
    if net.architecture != LAYER_STACKS:
        raise ValueError(f"architecture {net.architecture} has no .nnue form")
    layout = net.layout
    check_layout(layout)
    if layout["input_set"] != 0:
        raise ValueError(f"input_set {layout['input_set']} (Enyo HalfKA 32x12) has no Stockfish .nnue form")
    if layout["fc0_out"] != SUPPORTED_LAYOUT["fc0_out"] or layout["bucket_scheme"] != SUPPORTED_LAYOUT["bucket_scheme"]:
        raise ValueError(f"fc0_out {layout['fc0_out']} / bucket_scheme {layout['bucket_scheme']} "
                         "has no Stockfish .nnue form (only 32 / 0)")
    network_hash, ft_hash, fc_hash = _sf_hashes(layout)
    a = net.arrays
    desc = description.encode("utf-8", "surrogateescape")
    parts = [struct.pack("<III", SF_VERSION, network_hash, len(desc)), desc,
             struct.pack("<I", ft_hash),
             _sf_leb(a["ft_biases"], np.int16),
             np.ascontiguousarray(a["threat_weights"], dtype=np.int8).tobytes(),
             _sf_leb(a["threat_psqt"], np.int32),
             _sf_leb(a["halfka_weights"], np.int16),
             _sf_leb(a["halfka_psqt"], np.int32)]
    for s in range(layout["stacks"]):
        parts.append(struct.pack("<I", fc_hash))
        for name, dtype, shape in stack_sections(layout):
            parts.append(np.ascontiguousarray(a[name][s], dtype=dtype).reshape(shape).tobytes())
    return b"".join(parts)


def is_nnue(data: bytes) -> bool:
    return len(data) >= 4 and struct.unpack_from("<I", data)[0] == SF_VERSION


# ---------------------------------------------------------------------------
# Container

def payload_bytes(net: NetV1) -> bytes:
    if net.architecture == 24:
        return _raw(net.arrays, (INPUT,) + TAIL)
    if net.architecture == 25:
        iw = np.ascontiguousarray(net.arrays["input_weights"], dtype=np.int16).reshape(INPUT[2])
        leb = leb128_encode(iw)
        return struct.pack("<I", len(leb)) + leb + _raw(net.arrays, (THREATS,) + TAIL)
    if net.architecture == LAYER_STACKS:
        return _layer_stacks_payload(net)
    raise ValueError(f"unsupported architecture {net.architecture}")


def container_bytes(net: NetV1) -> bytes:
    payload = payload_bytes(net)
    meta = "".join(f"{k}={v}\n" for k, v in net.metadata.items()).encode()
    shifts = (0, 0) if net.architecture == LAYER_STACKS else (net.int_l1_shift, net.int_l2_shift)
    header = HEADER.pack(MAGIC, FORMAT_VERSION, net.architecture, len(meta), len(payload),
                         fnv1a64(payload), *shifts, bytes(6))
    return header + meta + payload


def write(path: str | Path, net: NetV1) -> None:
    Path(path).write_bytes(container_bytes(net))


def _take(payload, off, sections, arrays):
    for name, dtype, shape in sections:
        count = int(np.prod(shape))
        arrays[name] = np.frombuffer(payload, dtype=dtype, count=count, offset=off).reshape(shape).copy()
        off += count * np.dtype(dtype).itemsize
    return off


def read(path: str | Path) -> NetV1:
    return parse(Path(path).read_bytes(), path)


def parse(data: bytes, path="net") -> NetV1:
    if len(data) < HEADER.size:
        raise ValueError(f"{path}: truncated header")
    magic, version, arch, meta_size, payload_size, checksum, l1s, l2s, reserved = HEADER.unpack_from(data)
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
    if arch == LAYER_STACKS:
        if l1s or l2s or any(reserved):
            raise ValueError(f"{path}: header reserved bytes are not zero")
        layout, arrays = _read_layer_stacks(payload, path)
        return NetV1(arch, 0, 0, meta, arrays, layout)
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


def _main() -> None:
    import argparse
    import datetime
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    ap = argparse.ArgumentParser(
        description="convert between Enyo format-1 nets and their sources: Stockfish layer-stacks "
                    ".nnue <-> architecture 26, integer-dense Enyo v8/v12 -> architecture 24/25")
    ap.add_argument("source")
    ap.add_argument("target")
    ap.add_argument("--name", help="metadata name (default: target file stem)")
    ap.add_argument("--trained", help="metadata trained time (default: source mtime, UTC)")
    ap.add_argument("--description", help=".nnue description when writing a .nnue "
                                          "(default: 'Enyo net NAME, trained TRAINED.')")
    a = ap.parse_args()
    source = Path(a.source)
    data = source.read_bytes()
    trained = a.trained or datetime.datetime.fromtimestamp(source.stat().st_mtime, datetime.UTC).strftime("%Y-%m-%dT%H:%MZ")
    metadata = {"name": a.name or Path(a.target).stem, "trained": trained}

    if data[:8] == MAGIC:
        net = parse(data, source)
        if net.architecture != LAYER_STACKS:
            raise SystemExit(f"{source}: architecture {net.architecture} converts only from its v8/v12 source")
        name = net.metadata.get("name", "unnamed")
        description = a.description or f"Enyo net {name}, trained {net.metadata.get('trained', 'unknown')}."
        out = to_nnue(net, description)
        back, _ = from_nnue(out)
        assert all(np.array_equal(back.arrays[k], net.arrays[k]) for k in net.arrays)
        Path(a.target).write_bytes(out)
        print(f"wrote {a.target}: Stockfish .nnue, {len(out)} bytes, description {description!r}")
        return

    if is_nnue(data):
        net, description = from_nnue(data, metadata)
        if to_nnue(net, description) != data:
            raise SystemExit(f"{source}: does not re-export byte-identically (non-minimal LEB128?)")
        del data
    else:
        from lib import enyo_nnue
        net = from_enyo_net(enyo_nnue.load_net(source))
        net.metadata = metadata
    out = container_bytes(net)
    back = parse(out, a.target)
    assert all(np.array_equal(back.arrays[k], net.arrays[k]) for k in net.arrays)
    Path(a.target).write_bytes(out)
    print(f"wrote {a.target}: architecture {net.architecture}, {len(out)} bytes, {back.metadata}")


if __name__ == "__main__":
    _main()
