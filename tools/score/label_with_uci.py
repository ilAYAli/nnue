"""Relabel JSONL positions with a UCI teacher engine."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import bullet_format, bullet_text
from lib.hashing import sha256_file


SCORE_RE = re.compile(r"\bscore\s+(cp|mate)\s+(-?\d+)\b")
EVAL_RE = re.compile(r"^eval\s+(-?\d+)\s*$")
ENGINE_ID_NAME_RE = re.compile(r"^id name (.+)$")
NET_LOAD_ERROR_RE = re.compile(r"^info string ERROR:", re.IGNORECASE)
NET_LOAD_FALLBACK_RE = re.compile(r"falling back|source=embedded", re.IGNORECASE)
# Different engines report a successful net load differently: Stockfish
# names the file ("NNUE evaluation using nn-<hash>.nnue"), Enyo reports its
# content hash ("evaluator=enyo-nnue path='...' sha256=<hash> ..."). Neither
# format is trusted on its own; confirmation requires the reported name or
# hash to match the net this process actually asked for.
NET_LOAD_STOCKFISH_CONFIRM_RE = re.compile(r"^info string NNUE evaluation using (\S+)")
NET_LOAD_SHA256_RE = re.compile(r"\bsha256=([0-9a-f]{16,64})\b", re.IGNORECASE)
NET_LOAD_PROBE_MAX_LINES = 200


class EngineTimeout(RuntimeError):
    pass


class UciEngine:
    def __init__(
        self,
        path: str,
        *,
        threads: int,
        hash_mb: int,
        net: str | None = None,
        net_option: str = "nnue_file",
    ) -> None:
        self.path = path
        self.threads = threads
        self.hash_mb = hash_mb
        self.net = net
        self.net_option = net_option
        self.output_history: list[str] = []
        self.net_load_confirmed = False
        self.net_load_confirmation: str | None = None
        self.start()

    def start(self) -> None:
        self.proc = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.reader = threading.Thread(target=self.read_stdout, daemon=True)
        self.reader.start()
        self.net_load_confirmed = False
        self.net_load_confirmation = None
        try:
            self.send("uci")
            self.wait_for("uciok", timeout_s=60.0)
            self._verify_engine_identity()
            self.setoption("Threads", str(self.threads))
            self.setoption("Hash", str(self.hash_mb))
            if self.net:
                self.setoption(self.net_option, self.net)
            self.send("isready")
            if self.net:
                self._verify_net_loaded()
            else:
                self.wait_for("readyok", timeout_s=60.0)
        except Exception:
            # A crashed process leaves a reader thread and pipes behind. Make
            # a failed handshake fully disposable before a restart attempt.
            self.close()
            raise

    def _verify_engine_identity(self) -> None:
        """Sanity-check the handshake actually produced an engine identity."""
        name = ""
        for line in self.output_history:
            match = ENGINE_ID_NAME_RE.match(line)
            if match:
                name = match.group(1)
        if not name:
            raise RuntimeError("engine did not report an id name during the UCI handshake")

    def _verify_net_loaded(self, *, timeout_s: float = 30.0) -> None:
        """Prove the requested net actually loaded rather than trusting readyok.

        A UCI engine's handshake answers readyok even when the net option
        points at a missing, corrupt, or architecturally incompatible file.
        Engines differ in *when* they report the outcome: Enyo reports it
        eagerly, as an info string between isready and readyok; Stockfish
        reports it lazily, only the first time the net is actually used.
        They also differ in *how* they confirm success: Stockfish names the
        file, Enyo reports a content hash. This must be called right after
        sending isready (not after wait_for("readyok"), which would consume
        and discard an eager engine's report before this ever sees it).
        """
        assert self.net is not None
        expected_name = Path(self.net).name
        expected_sha256 = sha256_file(Path(self.net))
        reported: list[str] = []

        def scan(line: str) -> bool:
            if NET_LOAD_ERROR_RE.match(line) or NET_LOAD_FALLBACK_RE.search(line):
                raise RuntimeError(f"engine failed to load the requested net {self.net!r}: {line}")
            sha_match = NET_LOAD_SHA256_RE.search(line)
            if sha_match:
                reported.append(sha_match.group(1))
                if sha_match.group(1).lower() == expected_sha256.lower():
                    self.net_load_confirmed = True
                    self.net_load_confirmation = line
                    return True
                return False
            name_match = NET_LOAD_STOCKFISH_CONFIRM_RE.match(line)
            if name_match:
                reported.append(name_match.group(1))
                if name_match.group(1) == expected_name:
                    self.net_load_confirmed = True
                    self.net_load_confirmation = line
                    return True
            return False

        deadline = time.monotonic() + timeout_s
        saw_readyok = False
        for _ in range(NET_LOAD_PROBE_MAX_LINES):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EngineTimeout("engine did not answer readyok during net-load verification")
            line = self.readline(timeout_s=remaining)
            if line == "readyok":
                saw_readyok = True
                break
            if scan(line):
                self.wait_for("readyok", timeout_s=max(0.0, deadline - time.monotonic()))
                return
        if not saw_readyok:
            raise RuntimeError(f"engine did not answer readyok while loading net {self.net!r}")

        # Not confirmed (or refuted) before readyok: this engine reports
        # lazily. Force the report with a throwaway eval instead of
        # discovering a silent mislabel after a whole shard has been written.
        self.send("position startpos")
        self.send("eval")
        for _ in range(NET_LOAD_PROBE_MAX_LINES):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EngineTimeout("engine did not respond to the net-load verification probe")
            line = self.readline(timeout_s=remaining)
            if scan(line):
                return
        raise RuntimeError(
            f"engine did not confirm loading the requested net {self.net!r}; "
            f"reported {reported or 'nothing'}"
        )

    def close(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is None:
            return
        try:
            if proc.poll() is None:
                self.send("quit")
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def restart(self, *, attempts: int = 3, delay_s: float = 1.0) -> None:
        if attempts <= 0:
            raise ValueError("engine restart attempts must be positive")
        self.close()
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                self.start()
                return
            except (OSError, EngineTimeout, RuntimeError) as exc:
                last_error = exc
                self.close()
                if attempt + 1 < attempts and delay_s > 0:
                    time.sleep(delay_s * (attempt + 1))
        assert last_error is not None
        raise RuntimeError(
            f"engine restart failed after {attempts} attempts: {last_error}"
        ) from last_error

    def read_stdout(self) -> None:
        if self.proc.stdout is None:
            self.lines.put(None)
            return
        for line in self.proc.stdout:
            text = line.strip()
            self.output_history.append(text)
            self.lines.put(text)
        self.lines.put(None)

    def send(self, command: str) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("engine stdin closed")
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def readline(self, *, timeout_s: float | None = None) -> str:
        try:
            line = self.lines.get(
                timeout=max(0.0, timeout_s) if timeout_s is not None else None,
            )
        except queue.Empty:
            raise EngineTimeout(f"engine timed out after {timeout_s:.1f}s") from None
        if line is None:
            raise RuntimeError("engine exited")
        return line

    def wait_for(self, token: str, *, timeout_s: float | None = None) -> None:
        deadline = time.monotonic() + timeout_s if timeout_s is not None else None
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise EngineTimeout(f"engine did not send {token!r}")
            if self.readline(timeout_s=remaining) == token:
                return

    def setoption(self, name: str, value: str) -> None:
        self.send(f"setoption name {name} value {value}")

    def label(self, fen: str, *, depth: int, timeout_s: float) -> tuple[int | None, str | None]:
        self.send(f"position fen {fen}")
        self.send(f"go depth {depth}")
        deadline = time.monotonic() + timeout_s if timeout_s > 0 else None
        score_cp: int | None = None
        mate: str | None = None
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise EngineTimeout(f"engine timed out labeling {fen}")
            line = self.readline(timeout_s=remaining)
            match = SCORE_RE.search(line)
            if match:
                kind, value = match.groups()
                if kind == "cp":
                    score_cp = int(value)
                    mate = None
                else:
                    mate = value
            if line.startswith("bestmove "):
                return score_cp, mate

    def static_eval(self, fen: str, *, timeout_s: float) -> int | None:
        self.send(f"position fen {fen}")
        self.send("eval")
        deadline = time.monotonic() + timeout_s if timeout_s > 0 else None
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise EngineTimeout(f"engine timed out on static eval {fen}")
            line = self.readline(timeout_s=remaining)
            match = EVAL_RE.match(line)
            if match:
                return int(match.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--engine", default="stockfish")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash", type=int, default=128)
    ap.add_argument("--engine-timeout-s", type=float, default=30.0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum input rows to consider before sharding; 0 means all rows.",
    )
    ap.add_argument("--max-abs-cp", type=int, default=1600)
    ap.add_argument("--progress", type=int, default=1000)
    ap.add_argument("--output-format", default="jsonl",
                    choices=["jsonl", "packed", "bullet-text", "bullet-data"])
    ap.add_argument("--max-features", type=int, default=32)
    ap.add_argument("--enyo-runtime-target", action="store_true")
    args = ap.parse_args()

    if not (0 <= args.shard_index < args.shard_count):
        raise ValueError("invalid shard index/count")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "input": args.input,
        "output": args.output,
        "engine": args.engine,
        "depth": args.depth,
        "threads": args.threads,
        "hash": args.hash,
        "engine_timeout_s": args.engine_timeout_s,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "limit": args.limit,
        "read": 0,
        "selected": 0,
        "written": 0,
        "skipped_mate": 0,
        "skipped_no_score": 0,
        "skipped_timeout": 0,
        "skipped_cp": 0,
        "output_format": args.output_format,
    }

    engine = UciEngine(args.engine, threads=args.threads, hash_mb=args.hash)
    start = time.monotonic()
    rows: list[dict] = []
    text_output = args.output_format in {"jsonl", "bullet-text"}
    binary_output = args.output_format == "bullet-data"
    output_context = (
        out.open("w")
        if text_output else
        out.open("wb")
        if binary_output else
        nullcontext()
    )
    try:
        with Path(args.input).open() as src, output_context as dst:
            for idx, line in enumerate(src):
                if args.limit > 0 and idx >= args.limit:
                    break
                stats["read"] += 1
                if idx % args.shard_count != args.shard_index:
                    continue
                line = line.strip()
                if not line:
                    continue
                stats["selected"] += 1
                row = json.loads(line)
                try:
                    score, mate = engine.label(
                        row["fen"],
                        depth=args.depth,
                        timeout_s=args.engine_timeout_s,
                    )
                except EngineTimeout as exc:
                    stats["skipped_timeout"] += 1
                    print(
                        f"timeout shard {args.shard_index}/{args.shard_count}: "
                        f"selected={stats['selected']} {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    engine.restart()
                    continue
                if mate is not None:
                    stats["skipped_mate"] += 1
                    continue
                if score is None:
                    stats["skipped_no_score"] += 1
                    continue
                if (
                    args.output_format not in {"bullet-text", "bullet-data"}
                    and args.max_abs_cp > 0
                    and abs(score) > args.max_abs_cp
                ):
                    stats["skipped_cp"] += 1
                    continue

                if "score" in row:
                    row["source_score"] = row["score"]
                row["score"] = score
                row["teacher"] = Path(args.engine).name
                row["teacher_depth"] = args.depth
                if args.output_format == "jsonl":
                    assert dst is not None
                    dst.write(json.dumps(row, separators=(",", ":")))
                    dst.write("\n")
                elif args.output_format == "bullet-text":
                    assert dst is not None
                    white_score = bullet_text.white_score_from_row(
                        row,
                        enyo_runtime_target=args.enyo_runtime_target,
                    )
                    if args.max_abs_cp > 0 and abs(white_score) > args.max_abs_cp:
                        stats["skipped_cp"] += 1
                        continue
                    dst.write(bullet_text.row_to_text(
                        row,
                        enyo_runtime_target=args.enyo_runtime_target,
                    ))
                    dst.write("\n")
                elif args.output_format == "bullet-data":
                    assert dst is not None
                    white_score = bullet_text.white_score_from_row(
                        row,
                        enyo_runtime_target=args.enyo_runtime_target,
                    )
                    if args.max_abs_cp > 0 and abs(white_score) > args.max_abs_cp:
                        stats["skipped_cp"] += 1
                        continue
                    bullet_format.write_row(
                        dst,
                        row,
                        enyo_runtime_target=args.enyo_runtime_target,
                    )
                else:
                    rows.append(row)
                stats["written"] += 1

                if args.progress > 0 and stats["selected"] % args.progress == 0:
                    elapsed = max(1e-6, time.monotonic() - start)
                    rate = stats["selected"] / elapsed
                    print(
                        f"shard {args.shard_index}/{args.shard_count}: "
                        f"selected={stats['selected']} written={stats['written']} "
                        f"rate={rate:.1f}/s",
                        flush=True,
                    )
    finally:
        engine.close()

    if args.output_format == "packed":
        from lib import packed_labels

        packed = packed_labels.write_shard(
            out,
            rows,
            max_features=args.max_features,
        )
        stats["packed"] = packed
    elif args.output_format == "bullet-data":
        stats["bullet_format"] = {
            "format": bullet_format.FORMAT,
            "record_bytes": bullet_format.RECORD_BYTES,
            "records": bullet_format.record_count(out),
        }

    stats["elapsed_s"] = round(time.monotonic() - start, 3)
    stats_path = out.with_suffix(out.suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
