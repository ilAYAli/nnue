from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import chess


ISSUE_RE = re.compile(
    r"^\s*(?P<severity>inaccuracy|mistake|blunder):\s*"
    r"\[(?P<ply>\d+)/(?P<total>\d+)\]\s+"
    r"(?P<played>[a-h][1-8][a-h][1-8][qrbn]?)\b.*?"
    r"\bbest:\s*(?P<best>[a-h][1-8][a-h][1-8][qrbn]?)\s+"
    r"loss:\s*(?P<loss>\d+)cp\s+FEN:\s+(?P<fen>.+?)\s*$",
    re.IGNORECASE,
)


def iter_analysis_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.analysis")))
            files.extend(sorted(path.glob("*_analysis")))
        elif path.exists():
            files.append(path)
        else:
            raise SystemExit(f"missing analysis path: {path}")
    return files


def legal_case(row: dict[str, object]) -> tuple[bool, str]:
    try:
        board = chess.Board(str(row["fen"]))
        played = chess.Move.from_uci(str(row["played"]))
        best = chess.Move.from_uci(str(row["best"]))
    except Exception as exc:
        return False, str(exc)
    if played not in board.legal_moves:
        return False, f"played move is illegal: {played}"
    if best not in board.legal_moves:
        return False, f"best move is illegal: {best}"
    return True, ""


def parse_file(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_no, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        match = ISSUE_RE.match(line)
        if not match:
            continue
        row: dict[str, object] = {
            "source": str(path),
            "line": line_no,
            "severity": match.group("severity").lower(),
            "ply": int(match.group("ply")),
            "total_plies": int(match.group("total")),
            "played": match.group("played").lower(),
            "best": match.group("best").lower(),
            "loss_cp": int(match.group("loss")),
            "fen": match.group("fen"),
        }
        ok, reason = legal_case(row)
        if ok:
            rows.append(row)
        else:
            print(f"skip {path}:{line_no}: {reason}", file=sys.stderr)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis", nargs="+", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--min-loss", type=int, default=60)
    ap.add_argument("--max-loss", type=int, default=1000)
    ap.add_argument("--include-oot", action="store_true")
    ap.add_argument("--dedupe", action="store_true", default=True)
    args = ap.parse_args()

    rows: list[dict[str, object]] = []
    for path in iter_analysis_files(args.analysis):
        if not args.include_oot and "_oot" in path.name:
            continue
        rows.extend(parse_file(path))

    filtered = [
        row for row in rows
        if int(row["loss_cp"]) >= args.min_loss
        and int(row["loss_cp"]) <= args.max_loss
    ]

    if args.dedupe:
        seen: set[tuple[str, str, str]] = set()
        deduped: list[dict[str, object]] = []
        for row in filtered:
            key = (str(row["fen"]), str(row["played"]), str(row["best"]))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        filtered = deduped

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row in filtered:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    by_severity: dict[str, int] = {}
    for row in filtered:
        severity = str(row["severity"])
        by_severity[severity] = by_severity.get(severity, 0) + 1
    print(
        f"wrote {len(filtered)} cases to {args.output} "
        f"from {len(rows)} parsed issues; severity={by_severity}"
    )


if __name__ == "__main__":
    main()
