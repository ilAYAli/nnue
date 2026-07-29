#!/usr/bin/env python3
"""Map BulletFormat search-score labels onto this lineage's static-eval scale.

The Stockfish master binpacks are labeled with search scores (nodes5000pv2 and
friends). This lineage's nets are trained and judged against Stockfish's *static*
evaluation: tools/posgen/relabel_with_stockfish.py builds its labels with the UCI
`eval` command, and tools/validate/structural_net_audit.py scores a .nnue subject
the same way.

Those are not the same quantity, and the difference is not a constant. Measured
on 6000 positions drawn from a converted binpack, paired against the static eval
of nn-0ee0657fb25e on the identical position:

    |static eval|      label / static
         0 -   50            3.56
        50 -  100            2.13
       100 -  300            1.68
       300 -  800            1.42
        800+                 1.24

Search resolves tactics that static eval cannot see, so the gap is widest in
quiet positions and narrows as the position becomes decisively won. Overall
slope(label ~ static) is 1.277, which is why every net trained on this data
lands near slope 1.2 against the reference regardless of dose, learning rate or
eval_scale, and why no single global output rescale ever corrected it: the
distortion is magnitude-dependent, so a uniform factor is wrong everywhere
except at one crossing point.

This applies the inverse map to the labels instead, so the net learns
search-derived rankings expressed in the coordinate system the lineage is
measured in. Interpolation is piecewise linear on |label|, anchored at the
measured bucket means and extrapolated along the final segment.

Only the two-byte score field changes; every other byte is copied verbatim.
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import struct
import sys
from pathlib import Path
from typing import BinaryIO

RECORD_BYTES = 32
SCORE = struct.Struct("<h")
SCORE_OFFSET = 24

# (mean |label|, mean |static eval|) at each measured bucket, ascending.
CALIBRATION = (
    (47.6, 13.4),
    (155.1, 72.7),
    (313.0, 186.1),
    (713.5, 501.4),
    (1343.4, 1086.1),
)

ENYO_RUNTIME_SCORE_CLAMP = 2045


def recalibrate(score: int) -> int:
    """Map one search-score label onto the static-eval scale."""
    magnitude = abs(score)
    points = CALIBRATION
    if magnitude <= points[0][0]:
        # Below the first anchor, interpolate toward the origin so that a label
        # of zero stays zero rather than acquiring an offset.
        label, static = points[0]
        mapped = magnitude * (static / label)
    else:
        for (label_lo, static_lo), (label_hi, static_hi) in zip(points, points[1:]):
            if magnitude <= label_hi:
                span = label_hi - label_lo
                position = (magnitude - label_lo) / span if span else 0.0
                mapped = static_lo + position * (static_hi - static_lo)
                break
        else:
            # Extrapolate along the slope of the final segment.
            (label_lo, static_lo), (label_hi, static_hi) = points[-2], points[-1]
            slope = (static_hi - static_lo) / (label_hi - label_lo)
            mapped = static_hi + (magnitude - label_hi) * slope
    mapped = min(mapped, ENYO_RUNTIME_SCORE_CLAMP)
    result = int(round(mapped))
    return result if score >= 0 else -result


def rewrite(
    source: Path,
    destination: Path,
    *,
    progress_every: int = 20_000_000,
    output_stream: BinaryIO | None = None,
) -> dict[str, object]:
    size = source.stat().st_size
    if size % RECORD_BYTES:
        raise ValueError(f"{source}: size {size} is not divisible by {RECORD_BYTES}")

    read = changed = 0
    before = after = 0
    output_file = output_stream or destination.open("wb")
    close_output = output_stream is None
    try:
        with source.open("rb") as input_file:
            while True:
                chunk = input_file.read(RECORD_BYTES * 65536)
                if not chunk:
                    break
                if len(chunk) % RECORD_BYTES:
                    raise ValueError(f"{source}: truncated BulletFormat record")
                buffer = bytearray(chunk)
                for offset in range(0, len(buffer), RECORD_BYTES):
                    original = SCORE.unpack_from(buffer, offset + SCORE_OFFSET)[0]
                    mapped = recalibrate(original)
                    read += 1
                    before += abs(original)
                    after += abs(mapped)
                    if mapped != original:
                        SCORE.pack_into(buffer, offset + SCORE_OFFSET, mapped)
                        changed += 1
                output_file.write(buffer)
                if progress_every > 0 and read % progress_every < 65536:
                    print(f"read={read}/{size // RECORD_BYTES} changed={changed}", flush=True)
        output_file.flush()
        if close_output:
            os.fsync(output_file.fileno())
    finally:
        if close_output:
            output_file.close()

    return {
        "schema": "enyo.bullet-recalibrate-labels.v1",
        "input": str(source),
        "output": str(destination),
        "records": read,
        "changed": changed,
        "mean_abs_before": round(before / read, 3) if read else 0.0,
        "mean_abs_after": round(after / read, 3) if read else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--progress-every", type=int, default=20_000_000)
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

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        stats = rewrite(source, temporary, progress_every=args.progress_every)
        os.chmod(temporary, stat.S_IMODE(source.stat().st_mode))
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        stats["output"] = str(destination)
        if args.stats:
            args.stats.expanduser().write_text(
                json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(json.dumps(stats, sort_keys=True), flush=True)
    finally:
        temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
