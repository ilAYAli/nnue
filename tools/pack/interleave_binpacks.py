#!/usr/bin/env python3
"""Interleave Stockfish binpacks chunk by chunk into one file.

A binpack is a sequence of independent chunks: b"BINP", a little-endian u32
payload size, then the payload. The sfbinpack loader reads its input files one
after another, so a long run over several sources would train on one source at
a time. This writes a single file whose chunks are drawn from all inputs in
random order, each input picked in proportion to its remaining bytes, so every
part of the output holds the same source mix. Chunks are copied verbatim.

Usage: interleave_binpacks.py --output OUT.binpack [--seed N] IN.binpack...
"""

import argparse
import os
import random
import struct
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("inputs", nargs="+")
    args = ap.parse_args()

    if os.path.exists(args.output):
        sys.exit(f"error: output exists: {args.output}")
    rng = random.Random(args.seed)
    files = [open(p, "rb", buffering=1 << 24) for p in args.inputs]
    remaining = [os.path.getsize(p) for p in args.inputs]
    total = sum(remaining)
    chunks = [0] * len(files)
    written = 0
    start = time.time()
    tmp = args.output + ".tmp"
    with open(tmp, "wb", buffering=1 << 24) as out:
        while any(remaining):
            i = rng.choices(range(len(files)), weights=remaining)[0]
            header = files[i].read(8)
            if len(header) != 8 or header[:4] != b"BINP":
                sys.exit(f"error: bad chunk header in {args.inputs[i]} "
                         f"with {remaining[i]} bytes left")
            size = struct.unpack("<I", header[4:])[0]
            payload = files[i].read(size)
            if len(payload) != size:
                sys.exit(f"error: truncated chunk in {args.inputs[i]}")
            out.write(header)
            out.write(payload)
            remaining[i] -= 8 + size
            chunks[i] += 1
            written += 8 + size
            if sum(chunks) % 20000 == 0:
                rate = written / (time.time() - start) / 1e6
                print(f"{written / 1e9:7.1f} / {total / 1e9:.1f} GB  {rate:.0f} MB/s",
                      flush=True)
    os.rename(tmp, args.output)
    for path, n in zip(args.inputs, chunks):
        print(f"{n:9d} chunks  {os.path.basename(path)}")
    print(f"wrote {written} bytes to {args.output} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
