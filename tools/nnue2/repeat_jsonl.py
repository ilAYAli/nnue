from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--repeat", type=int, required=True)
    args = ap.parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")

    lines = [line for line in args.input.read_text().splitlines() if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for _ in range(args.repeat):
            for line in lines:
                out.write(line)
                out.write("\n")
    print(f"wrote {len(lines) * args.repeat} rows to {args.output}")


if __name__ == "__main__":
    main()
