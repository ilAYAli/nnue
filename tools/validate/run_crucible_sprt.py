#!/usr/bin/env python3
"""Run an NNUE SPRT through Crucible and aggregate the shard logs."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.events import emit_event


SCORE_RE = re.compile(
    r"Score of candidate vs reference:\s+"
    r"(?P<wins>\d+)\s+-\s+(?P<losses>\d+)\s+-\s+(?P<draws>\d+)"
    r"\s+\[[^\]]+\]\s+(?P<games>\d+)"
)
TRANSIENT_STARTUP_PATTERNS = (
    "Engine didn't respond to uciok after startup",
    "engine startup failure",
    "Engine reference is not responsive",
    "Engine candidate is not responsive",
)


@dataclass(frozen=True)
class SprtAggregate:
    wins: int
    losses: int
    draws: int
    games: int

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0

    @property
    def elo(self) -> float:
        score = min(max(self.score, 1e-6), 1.0 - 1e-6)
        return -400.0 * math.log10(1.0 / score - 1.0)

    @property
    def ci95(self) -> float:
        if self.games <= 1:
            return float("nan")
        values = [1.0] * self.wins + [0.0] * self.losses + [0.5] * self.draws
        mean = self.score
        variance = sum((value - mean) ** 2 for value in values) / (self.games - 1)
        score_stderr = math.sqrt(variance / self.games)
        score = min(max(mean, 1e-6), 1.0 - 1e-6)
        elo_derivative = 400.0 / (math.log(10.0) * score * (1.0 - score))
        return 1.96 * elo_derivative * score_stderr

    @property
    def los(self) -> float:
        if self.games <= 1:
            return 50.0
        values = [1.0] * self.wins + [0.0] * self.losses + [0.5] * self.draws
        mean = self.score
        variance = sum((value - mean) ** 2 for value in values) / (self.games - 1)
        stderr = math.sqrt(variance / self.games)
        if stderr <= 0.0:
            return 100.0 if mean > 0.5 else 0.0 if mean < 0.5 else 50.0
        z = (mean - 0.5) / stderr
        return 100.0 * 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def expand_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def shell_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


def shell_quote_expand_home(value: str) -> str:
    if "$HOME/" not in value:
        return shell_quote(value)
    before, after = value.split("$HOME/", 1)
    if "$HOME/" in after:
        return shell_quote(value)
    escaped_before = before.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`").replace("$", "\\$")
    escaped_after = after.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`").replace("$", "\\$")
    return f'"{escaped_before}$HOME/{escaped_after}"'


def safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return text or "sprt"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_cached(source: Path, cache_dir: Path, prefix: str) -> Path:
    digest = file_sha256(source)[:12]
    target = cache_dir / f"{safe_name(prefix)}-{digest}{source.suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_size != source.stat().st_size:
        shutil.copy2(source, target)
    return target


def tilde_path(path: Path) -> str:
    home = Path.home().resolve()
    try:
        return "~/" + str(path.resolve().relative_to(home))
    except ValueError:
        return str(path)


def worker_command_path(path: str) -> str:
    if path.startswith("~/"):
        return "$HOME/" + path[2:]
    return path


def add_path_map(path_maps: dict[str, dict[str, str]], profile: str, src: str, dst: str) -> None:
    if not profile or not src or not dst or src == dst:
        return
    path_maps.setdefault(profile, {}).setdefault(src, dst)


def cache_runs_dir(cache_dir: str) -> str:
    return str(PurePath(cache_dir) / "runs")


def worker_cache_dir(worker: dict[str, Any]) -> str:
    cache = str(worker.get("cache_dir") or "").strip()
    if cache:
        return cache
    host = str(worker.get("host") or "").strip()
    if host in ("localhost", "127.0.0.1") or bool(worker.get("local", False)):
        return "~/.cache/crucible"
    return ".cache/crucible"


def worker_profiles(path: Path) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return {}
    workers = payload.get("workers", [])
    if not isinstance(workers, list):
        return {}
    profiles: dict[str, str] = {}
    for item in workers:
        if isinstance(item, str):
            host, _, profile = item.partition("=")
            profile = profile or host
            cache = "~/.cache/crucible"
        elif isinstance(item, dict):
            if not bool(item.get("enabled", True)):
                continue
            host = str(item.get("host") or "").strip()
            profile = str(item.get("profile") or host).strip()
            cache = worker_cache_dir(item)
        else:
            continue
        if profile and profile not in profiles:
            profiles[profile] = cache
    return profiles


def worker_path_maps(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    maps: dict[str, dict[str, str]] = {}
    workers = worker_profiles(expand_path(args.workers))
    coordinator_runs = str(expand_path(args.crucible_runs_dir))
    coordinator_cache = str(expand_path(args.cache_dir))
    cache_text = str(args.cache_dir)
    cache_tilde = tilde_path(expand_path(args.cache_dir))
    cache_arg_home = worker_command_path(cache_text)
    cache_home = worker_command_path(cache_tilde)
    for profile, cache in workers.items():
        add_path_map(maps, profile, coordinator_runs, cache_runs_dir(cache))
        add_path_map(maps, profile, coordinator_cache, cache)
        add_path_map(maps, profile, cache_text, cache)
        add_path_map(maps, profile, cache_arg_home, cache)
        add_path_map(maps, profile, cache_tilde, cache)
        add_path_map(maps, profile, cache_home, cache)
    return maps


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_streamed(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(shell_quote(part) for part in command), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return proc.wait()


def run_capture(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def task_command(
    *,
    index: int,
    name_prefix: str,
    candidate_net: str,
    reference_net: str,
    book: str,
    games: int,
    concurrency: int,
    threads: int,
    hash_mb: int,
    tc: str,
    elo0: float,
    elo1: float,
    seed: int,
) -> str:
    task_name = f"{safe_name(name_prefix)}-sprt_{index:04d}"
    candidate_arg = f"nnue_file={worker_command_path(candidate_net)}"
    reference_arg = f"nnue_file={worker_command_path(reference_net)}"
    book_arg = worker_command_path(book)
    parts = [
        "set -euo pipefail",
        "out={output}",
        'logdir="$(dirname "$out")/logs"',
        f"name={shell_quote(task_name)}",
        'rm -rf "$logdir"',
        'mkdir -p "$logdir"',
        (
            "$HOME/.local/bin/sprt "
            "--candidate $HOME/assets/engines/reference "
            "--reference $HOME/assets/engines/reference "
            "--candidate-uci use_syzygy=false "
            "--reference-uci use_syzygy=false "
            f"--candidate-uci {shell_quote_expand_home(candidate_arg)} "
            f"--reference-uci {shell_quote_expand_home(reference_arg)} "
            f"--candidate-uci Hash={hash_mb} "
            f"--reference-uci Hash={hash_mb} "
            f"--games {games} "
            f"--concurrency {concurrency} "
            f"--threads {threads} "
            f"--tc {shell_quote(tc)} "
            f"--elo0 {elo0} "
            f"--elo1 {elo1} "
            "--restart off "
            "--fastchess $HOME/.local/bin/fastchess "
            f"--book {shell_quote_expand_home(book_arg)} "
            '--log-dir "$logdir" '
            '--name "$name" '
            "--order random "
            f"--srand {seed + index} "
            "--no-ntfy "
            "--no-eta"
        ),
        'cp "$logdir/$name.log" "$out"',
    ]
    return "; ".join(parts)


def build_manifest(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    validate_game_counts(args)
    run_name = safe_name(args.tag)
    chunk_count = (args.games + args.chunk_games - 1) // args.chunk_games
    if chunk_count <= 0:
        raise SystemExit("--games and --chunk-games must produce at least one task")

    cache = expand_path(args.cache_dir)
    candidate_cache = copy_cached(expand_path(args.net), cache / "nnue-nets", f"{args.tag}-candidate")
    reference_cache = copy_cached(
        expand_path(args.reference_net), cache / "nnue-nets", f"{args.tag}-reference"
    )
    book_cache = copy_cached(expand_path(args.book), cache / "books", Path(args.book).stem)

    candidate_net = tilde_path(candidate_cache)
    reference_net = tilde_path(reference_cache)
    book = tilde_path(book_cache)

    tasks: list[dict[str, Any]] = []
    for index in range(chunk_count):
        games = min(args.chunk_games, args.games - index * args.chunk_games)
        task_id = f"sprt_{index:04d}"
        tasks.append({
            "id": task_id,
            "index": index,
            "label": f"sprt:{games}",
            "timeout_s": args.task_timeout_seconds,
            "progress": {
                "log_regex": (
                    r"\[\s*(?P<done>\d+)\s*/\s*(?P<total>\d+)\]\s+"
                    r"Elo\s+(?P<elo>[+-]?(?:\d+(?:\.\d+)?|inf))\s+"
                    r"\+/-\s+[+-]?(?:\d+(?:\.\d+)?|nan|-nan)\s+\|\s+"
                    r"LLR\s+[^\|]*\|\s+LOS\s+(?P<los>[0-9.]+%)\s+\|\s+"
                    r"draw\s+(?P<draw>[0-9.]+%)"
                ),
                "total": games,
                "unit": "games",
                "display_fields": {
                    "elo": "elo~",
                    "los": "los",
                    "draw": "draw",
                },
            },
            "command": task_command(
                index=index,
                name_prefix=args.tag,
                candidate_net=candidate_net,
                reference_net=reference_net,
                book=book,
                games=games,
                concurrency=args.chunk_concurrency,
                threads=args.threads,
                hash_mb=args.hash,
                tc=args.tc,
                elo0=args.elo0,
                elo1=args.elo1,
                seed=args.seed,
            ),
            "outputs": [
                f"{{output_dir}}/{task_id}/sprt.log",
            ],
        })

    timeout_cmd = args.timeout_command
    candidate_probe_net = worker_command_path(candidate_net)
    reference_probe_net = worker_command_path(reference_net)
    probe = (
        f"{timeout_cmd} 30s bash -lc "
        + shell_quote(
            "printf '%s\n' uci "
            "'setoption name use_syzygy value false' "
            f"'setoption name nnue_file value {candidate_probe_net}' "
            "isready 'position startpos' 'go depth 1' quit | $HOME/assets/engines/reference"
        )
    )
    reference_probe = (
        f"{timeout_cmd} 30s bash -lc "
        + shell_quote(
            "printf '%s\n' uci "
            "'setoption name use_syzygy value false' "
            f"'setoption name nnue_file value {reference_probe_net}' "
            "isready 'position startpos' 'go depth 1' quit | $HOME/assets/engines/reference"
        )
    )

    manifest: dict[str, Any] = {
        "schema": "crucible.task.v1",
        "name": args.tag,
        "kind": "sprt",
        "description": args.description or f"{args.tag}: distributed SPRT",
        "work_dir": str(expand_path(args.crucible_runs_dir) / run_name),
        "requirements": {
            "python_min": "3.11",
            "git_min": "2.38",
            "commands": [
                {
                    "name": "sprt",
                    "command": "~/.local/bin/sprt --help",
                    "contains": "--no-ntfy",
                    "timeout_s": 10,
                },
                {
                    "name": "fastchess",
                    "command": "~/.local/bin/fastchess -version",
                    "timeout_s": 10,
                },
                {
                    "name": "engine-candidate-search",
                    "command": probe,
                    "contains": "bestmove",
                    "timeout_s": 40,
                },
                {
                    "name": "engine-reference-search",
                    "command": reference_probe,
                    "contains": "bestmove",
                    "timeout_s": 40,
                },
            ],
            "inputs": [
                {"path": candidate_net},
                {"path": reference_net},
                {"path": book},
            ],
        },
        "path_maps": worker_path_maps(args),
        "tasks": tasks,
    }
    for mapping in args.path_map:
        profile, sep, rest = mapping.partition(":")
        if not sep or "=" not in rest:
            raise SystemExit(f"--path-map must be profile:from=to, got {mapping!r}")
        src, dst = rest.split("=", 1)
        manifest["path_maps"].setdefault(profile, {})[src] = dst
    write_json(run_dir / "manifest.template.json", manifest)
    return manifest


def status_json(crucible: str, run_name: str) -> dict[str, Any]:
    result = run_capture([crucible, "status", run_name, "--json"])
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        pass
    if result.returncode != 0:
        raise RuntimeError(result.stdout)
    return json.loads(result.stdout)


def failed_rows(status: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in status.get("rows", []) if row.get("state") == "fail"]


def task_log(run_dir: Path, task_id: str) -> str:
    candidates = [
        run_dir / "logs" / f"{task_id}.log",
        run_dir / "outputs" / task_id / "sprt.log",
    ]
    for path in candidates:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    return ""


def is_transient_startup_failure(text: str) -> bool:
    return any(pattern in text for pattern in TRANSIENT_STARTUP_PATTERNS)


def retry_transient_failures(args: argparse.Namespace, run_name: str, run_dir: Path) -> int:
    status = status_json(args.crucible, run_name)
    rows = failed_rows(status)
    if not rows:
        return 0
    retried = 0
    manifest = run_dir / "manifest.json"
    for row in rows:
        task_id = str(row["id"])
        log_text = task_log(run_dir, task_id)
        if not is_transient_startup_failure(log_text):
            print(f"{task_id}: failure is not classified as transient startup", flush=True)
            continue
        print(f"{task_id}: retrying transient engine startup failure", flush=True)
        rc = run_streamed(
            [
                args.crucible,
                "run-task",
                "--manifest", str(manifest),
                "--index", str(row["index"]),
                "--force",
            ],
            run_dir / "retry.log",
        )
        if rc == 0:
            retried += 1
    return retried


def aggregate_sprt_logs(run_dir: Path) -> SprtAggregate:
    wins = losses = draws = games = 0
    logs = sorted((run_dir / "outputs").glob("*/sprt.log"))
    for log in logs:
        text = log.read_text(encoding="utf-8", errors="replace")
        matches = list(SCORE_RE.finditer(text))
        if not matches:
            raise RuntimeError(f"{log}: no final SPRT score line")
        match = matches[-1]
        wins += int(match.group("wins"))
        losses += int(match.group("losses"))
        draws += int(match.group("draws"))
        games += int(match.group("games"))
    return SprtAggregate(wins=wins, losses=losses, draws=draws, games=games)


def conclusion_text(args: argparse.Namespace, aggregate: SprtAggregate, run_dir: Path) -> str:
    return "\n".join([
        "<output>",
        "Distributed SPRT",
        f"  • Run: {args.tag}",
        f"  • Candidate: {args.net}",
        f"  • Reference: {args.reference_net}",
        f"  • Score: {aggregate.wins} - {aggregate.losses} - {aggregate.draws} [{aggregate.score:.4f}] {aggregate.games}",
        f"  • Elo: {aggregate.elo:.2f} +/- {aggregate.ci95:.2f}, LOS: {aggregate.los:.1f}%",
        f"  • Run dir: {run_dir}",
        "</output>",
    ])


def notify_stdout(message: str, *, enabled: bool) -> None:
    print(message, flush=True)
    if not enabled:
        return
    script = repo_root() / "tools" / "events" / "ai_stdout.sh"
    subprocess.run([str(script)], input=message + "\n", text=True, check=False)


def load_ntfy_auth() -> str:
    if os.environ.get("NTFY_AUTH"):
        return os.environ["NTFY_AUTH"]
    if os.environ.get("LICHESS_NTFY_AUTH"):
        return os.environ["LICHESS_NTFY_AUTH"]

    ntfy_file = Path.home() / ".ntfy"
    if not ntfy_file.is_file():
        return ""

    for line in ntfy_file.read_text(errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:]
        if line.startswith("NTFY_AUTH="):
            value = line.split("=", 1)[1].strip()
        elif line.startswith("LICHESS_NTFY_AUTH="):
            value = line.split("=", 1)[1].strip()
        else:
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        return value
    return ""


def post_sprt_ntfy(url: str, message: str, *, title: str) -> bool:
    if not url:
        return True

    req = urllib.request.Request(url, data=message.encode(), method="POST")
    if title:
        req.add_header("Title", title)
    auth = load_ntfy_auth()
    if auth:
        token = base64.b64encode(auth.encode()).decode()
        req.add_header("Authorization", f"Basic {token}")

    try:
        urllib.request.urlopen(req, timeout=10).close()
    except (OSError, urllib.error.URLError):
        print(f"sprt ntfy post failed: {url}", file=sys.stderr)
        return False
    return True


def notify_sprt(args: argparse.Namespace, message: str, *, title: str) -> None:
    post_sprt_ntfy(args.sprt_ntfy_url, message, title=title)


def emit_completion_event(
    args: argparse.Namespace,
    rc: int,
    run_dir: Path,
    message: str,
    *,
    log_path: Path | None = None,
) -> None:
    if not args.notify:
        return
    hook = getattr(args, "event_command", None) or (
        f"NNUE_NTFY_EVENTS= "
        f"NNUE_AI_STDIN_EVENTS=done,fail {repo_root() / 'tools' / 'events' / 'nnue_event_ntfy.sh'}"
    )
    emit_event(
        args.nnue_run or run_dir,
        "done" if rc == 0 else "fail",
        stage="validate_crucible_sprt",
        status="ok" if rc == 0 else "failed",
        rc=rc,
        log=log_path or (run_dir / "deploy.log"),
        message=message,
        hook_command=hook,
        extra={"tag": args.tag, "games": args.games},
    )


def cmd_run(args: argparse.Namespace) -> int:
    run_name = safe_name(args.tag)
    local_run_dir = expand_path(args.output_dir) if args.output_dir else (
        repo_root() / "runs" / "validation" / run_name
    )
    local_run_dir.mkdir(parents=True, exist_ok=True)
    build_manifest(args, local_run_dir)

    command = [
        args.crucible,
        "deploy",
        str(expand_path(args.workers)),
        str(local_run_dir / "manifest.template.json"),
        "--run", run_name,
        "--jobs", str(args.jobs),
        "--remote-timeout-seconds", str(args.remote_timeout_seconds),
    ]
    if args.crucible_notify:
        command.extend(["--notify-command", str(args.crucible_notify_command)])
    if args.replace:
        command.append("--replace")
    elif args.resume:
        command.append("--resume")
    if args.coordinator_host:
        command.extend(["--coordinator-host", args.coordinator_host])
    if args.verbose:
        command.append("--verbose")

    deploy_log = local_run_dir / "deploy.log"
    rc = run_streamed(command, deploy_log)
    run_dir = expand_path(args.crucible_runs_dir) / run_name

    if rc != 0:
        try:
            status = status_json(args.crucible, run_name)
            counts = status.get("counts", {})
            if counts.get("done") == args.task_count and counts.get("fail", 0) == 0:
                rc = 0
        except Exception as exc:
            print(f"{run_name}: cannot read Crucible status after deploy failure: {exc}", flush=True)

    if rc != 0 and args.retry_startup_failures:
        retried = retry_transient_failures(args, run_name, run_dir)
        if retried:
            status = status_json(args.crucible, run_name)
            counts = status.get("counts", {})
            rc = 0 if counts.get("done") == args.task_count and counts.get("fail", 0) == 0 else 1

    if rc != 0:
        message = f"Distributed SPRT failed: run={run_name} log={local_run_dir / 'deploy.log'}"
        notify_stdout(message, enabled=args.notify)
        notify_sprt(args, message, title="SPRT failed")
        emit_completion_event(args, rc, run_dir, message, log_path=deploy_log)
        return rc

    status = status_json(args.crucible, run_name)
    counts = status.get("counts", {})
    if counts.get("done") != args.task_count or counts.get("fail", 0) != 0:
        message = (
            f"Distributed SPRT incomplete: run={run_name} "
            f"done={counts.get('done')}/{args.task_count} fail={counts.get('fail', 0)}"
        )
        notify_stdout(message, enabled=args.notify)
        notify_sprt(args, message, title="SPRT failed")
        emit_completion_event(args, 1, run_dir, message, log_path=deploy_log)
        return 1

    aggregate = aggregate_sprt_logs(run_dir)
    if aggregate.games != args.games:
        message = f"Distributed SPRT game count mismatch: got={aggregate.games} expected={args.games}"
        notify_stdout(message, enabled=args.notify)
        notify_sprt(args, message, title="SPRT failed")
        emit_completion_event(args, 1, run_dir, message, log_path=deploy_log)
        return 1

    message = conclusion_text(args, aggregate, run_dir)
    notify_stdout(message, enabled=args.notify)
    notify_sprt(args, message, title="SPRT finished")
    emit_completion_event(args, 0, run_dir, message, log_path=deploy_log)
    return 0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def validate_game_counts(args: argparse.Namespace) -> None:
    for attr in ("games", "chunk_games"):
        value = int(getattr(args, attr))
        if value % 2 != 0:
            option = "--" + attr.replace("_", "-")
            raise SystemExit(
                f"{option} must be even because SPRT plays paired openings"
            )
    if args.chunk_games < 50:
        raise SystemExit(
            "--chunk-games must be at least 50; smaller chunks do not "
            "produce reliable exact game counts"
        )
    if args.games % args.chunk_games != 0:
        raise SystemExit("--games must be divisible by --chunk-games")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a distributed NNUE SPRT through Crucible."
    )
    parser.add_argument("--net", required=True, help="Candidate NNUE file.")
    parser.add_argument("--reference-net", required=True, help="Reference NNUE file.")
    parser.add_argument("--tag", required=True, help="Crucible run name and SPRT tag.")
    parser.add_argument("--description", default="")
    parser.add_argument("--workers", default="~/code/cpp/chess/crucible/runs/workers.json")
    parser.add_argument("--crucible", default="crucible")
    parser.add_argument("--crucible-runs-dir", default="~/code/cpp/chess/crucible/runs")
    parser.add_argument(
        "--crucible-notify-command",
        default="~/code/cpp/chess/crucible/scripts/crucible_event_ntfy.sh",
    )
    parser.add_argument("--work-dir", default="~/code/cpp/chess/nnue")
    parser.add_argument("--output-dir")
    parser.add_argument("--nnue-run", help="NNUE run dir used for done/fail wake events.")
    parser.add_argument("--event-command", help="Event hook command for done/fail wake events.")
    parser.add_argument("--cache-dir", default="~/.cache/crucible")
    parser.add_argument("--book", default="~/code/cpp/chess/assets/books/UHO_Lichess_4852_v1.epd")
    parser.add_argument("--games", type=positive_int, default=4000)
    parser.add_argument(
        "--chunk-games",
        type=positive_int,
        default=100,
        help=(
            "Games per Crucible task. Must be even, at least 50, and divide --games."
        ),
    )
    parser.add_argument("--chunk-concurrency", type=positive_int, default=4)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--hash", type=positive_int, default=512)
    parser.add_argument("--tc", default="2+0.02")
    parser.add_argument("--elo0", type=float, default=0.0)
    parser.add_argument("--elo1", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=int(time.strftime("%Y%m%d00")))
    parser.add_argument("--jobs", type=positive_int, default=8)
    parser.add_argument("--task-timeout-seconds", type=positive_int, default=1800)
    parser.add_argument("--remote-timeout-seconds", type=positive_int, default=1800)
    parser.add_argument("--timeout-command", default="timeout")
    parser.add_argument("--path-map", action="append", default=[])
    parser.add_argument("--coordinator-host")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--crucible-notify", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--retry-startup-failures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--notify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--sprt-ntfy-url",
        default=os.environ.get("SPRT_NTFY_URL", ""),
        help="Aggregate SPRT ntfy URL. Empty disables SPRT ntfy.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.crucible_notify_command = str(expand_path(args.crucible_notify_command))
    validate_game_counts(args)
    args.task_count = (args.games + args.chunk_games - 1) // args.chunk_games
    if args.resume and args.replace:
        parser.error("--resume and --replace are mutually exclusive")
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
