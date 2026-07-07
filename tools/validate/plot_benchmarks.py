#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{line_number}: {error}") from error
    if not rows:
        raise SystemExit(f"{path}: no benchmark rows")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot NNUE benchmark progress")
    parser.add_argument("input", type=Path, help="benchmark JSONL file")
    parser.add_argument("-o", "--output", type=Path, help="output PNG path")
    args = parser.parse_args()

    rows = load_rows(args.input)
    output = args.output or args.input.with_suffix(".png")
    labels = [row["candidate"] for row in rows]
    elo = [float(row["elo"]) for row in rows]
    ci = [float(row["ci"]) for row in rows]
    x = range(len(rows))

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(rows)), 5.5))
    ax.errorbar(x, elo, yerr=ci, fmt="o-", capsize=5, linewidth=2)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(list(x), labels, rotation=30, ha="right")
    ax.set_ylabel("Elo vs Stockfish net")
    ax.set_title("NNUE Stockfish-net benchmark progress")
    ax.grid(axis="y", alpha=0.3)

    for index, value in enumerate(elo):
        ax.annotate(f"{value:+.1f}", (index, value), xytext=(0, 9),
                    textcoords="offset points", ha="center")

    references = ", ".join(sorted({row["reference"] for row in rows}))
    engines = ", ".join(sorted({row["engine"] for row in rows}))
    fig.text(0.5, 0.01, f"Reference: {references} | Engine: {engines}", ha="center")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
