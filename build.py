#!/usr/bin/env python3
"""High-level Enyo NNUE candidate workflow command."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))

from lib.defaults import DEFAULTS, repo_root


def expand_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def expand_user(value: str | Path) -> Path:
    return Path(value).expanduser()


def tool(path: str) -> str:
    return str(repo_root() / "tools" / path)


def run(command: list[str], *, dry_run: bool = False) -> int:
    print(" ".join(command), flush=True)
    if dry_run:
        return 0
    proc = subprocess.Popen(command, cwd=repo_root())
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        raise


def default_name() -> str:
    return time.strftime("candidate_%Y%m%d_%H%M%S")


def run_dir_for(name: str, run_dir: str | None) -> Path:
    if run_dir:
        return expand_path(run_dir)
    return expand_path(DEFAULTS.run_base) / name


def write_config(run_dir: Path, config: dict) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def create_config(args: argparse.Namespace) -> dict:
    name = args.name or default_name()
    run_dir = run_dir_for(name, args.run_dir)
    candidate_dir = f"{{train}}/{name}"
    steps = [
        {
            "name": "posgen_selfplay",
            "command": [
                tool("posgen/posgen.py"), "selfplay",
                "--runner", str(expand_path(args.runner)),
                "--engine", str(expand_path(args.engine)),
                "--nnue-file", str(expand_path(args.nnue_file)),
                "--book", str(expand_path(args.book)),
                "--output", "{posgen}/selfplay.pgn",
                "--games", str(args.selfplay_games),
                "--shard-games", str(args.selfplay_shard_games),
                "--concurrency", str(args.selfplay_concurrency),
                "--threads", str(args.selfplay_threads),
                "--depth", str(args.selfplay_depth),
                "--srand", str(args.selfplay_seed),
                "--restart", "off",
                "--engine-option", f"Hash={args.selfplay_hash}",
            ],
        },
        {
            "name": "posgen_extract",
            "command": [
                tool("posgen/posgen.py"), "extract",
                "{posgen}/selfplay.pgn",
                "--output", "{posgen}/positions.jsonl",
                "--stats", "{posgen}/extract_stats.json",
                "--skip-plies", str(args.skip_plies),
                "--min-depth", str(args.selfplay_depth),
                "--max-abs-cp", str(args.source_max_abs_cp),
            ],
        },
        {
            "name": "posgen_sample",
            "command": [
                tool("posgen/posgen.py"), "sample",
                "--input", "{posgen}/positions.jsonl",
                "--output", "{posgen}/source.jsonl",
                "--preset", args.sample_preset,
                "--unique-fen",
                "--seed", str(args.selfplay_seed),
            ],
        },
    ]

    for shard in range(args.score_shards):
        steps.append({
            "name": f"score_{shard:02d}",
            "command": [
                tool("score/score.py"), "uci",
                "--input", "{posgen}/source.jsonl",
                "--output", f"{{score}}/shards/label.{shard}.jsonl",
                "--engine", str(expand_path(args.score_engine)),
                "--depth", str(args.score_depth),
                "--threads", str(args.score_threads),
                "--hash", str(args.score_hash),
                "--shard-count", str(args.score_shards),
                "--shard-index", str(shard),
                "--max-abs-cp", str(args.score_max_abs_cp),
                "--progress", str(args.score_progress),
            ],
        })

    steps.extend([
        {
            "name": "score_merge",
            "command": [
                "bash", "-lc",
                "cat \"$1\"/shards/label.*.jsonl > \"$1\"/labeled.jsonl && wc -l \"$1\"/labeled.jsonl > \"$1\"/labeled.wc",
                "merge-score", "{score}",
            ],
        },
        {
            "name": "pack",
            "command": [
                tool("pack/pack.py"), "build",
                "--input", "{score}/labeled.jsonl",
                "--out-dir", "{pack}/train",
                "--max-features", str(args.max_features),
                "--progress", str(args.pack_progress),
                "--python", str(expand_user(args.python)),
            ],
        },
        {
            "name": "train",
            "command": [
                tool("train/train.py"), "run",
                "--data", "{pack}/train",
                "--init-from-nn", str(expand_path(args.init_net)),
                "--objective", args.objective,
                "--huber-beta", str(args.huber_beta),
                "--select-metric", args.select_metric,
                "--wdl-lambda", str(args.wdl_lambda),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--lr", str(args.lr),
                "--weight-decay", str(args.weight_decay),
                "--target-clamp", str(args.target_clamp),
                "--device", args.device,
                "--workers", str(args.workers),
                "--patience", str(args.patience),
                "--val-rows", str(args.val_rows),
                "--trainable", args.trainable,
                "--python", str(expand_user(args.python)),
                "--out", f"{candidate_dir}/model.pt",
                "--out-nn", f"{candidate_dir}/model.nn",
            ],
        },
    ])

    config = {
        "name": name,
        "run": str(run_dir),
        "vars": {
            "candidate": name,
        },
        "create_args": {
            key: value
            for key, value in vars(args).items()
            if key not in {"func"}
        },
        "steps": steps,
    }
    if args.event_command:
        config["hooks"] = {
            "event_command": args.event_command,
        }
    return config


def cmd_create(args: argparse.Namespace) -> int:
    config = create_config(args)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return 0
    run_dir = expand_path(config["run"])
    config_path = write_config(run_dir, config)
    print(f"wrote {config_path}")
    command = [sys.executable, tool("pipeline/pipeline.py"), "launch", str(config_path)]
    if args.force:
        command.append("--force")
    return run(command)


def cmd_resume(args: argparse.Namespace) -> int:
    run_dir = expand_path(args.run)
    config = run_dir / "config.json"
    if not config.exists():
        config = run_dir / "config.yml"
    if not config.exists():
        raise SystemExit(f"missing config.json/config.yml in {run_dir}")
    command = [sys.executable, tool("pipeline/pipeline.py"), "launch", str(config)]
    if args.event_command:
        command += ["--on-event", args.event_command]
    if args.force:
        command.append("--force")
    return run(command)


def cmd_status(args: argparse.Namespace) -> int:
    return run([
        sys.executable, tool("pipeline/pipeline.py"), "status",
        str(expand_path(args.run)),
        "--tail", str(args.tail),
    ])


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = expand_path(args.run)
    print(f"run={run_dir}")
    for net in sorted(run_dir.glob("train/*/model.nn")):
        print(f"candidate={net}")
    for summary in sorted(run_dir.glob("validate/**/summary.txt")):
        print(f"summary={summary}")
        print(summary.read_text(encoding="utf-8", errors="replace").strip())
    for sprt_log in sorted(run_dir.glob("**/*sprt*.log")):
        print(f"sprt_log={sprt_log}")
        lines = sprt_log.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines[-args.tail:]:
            print(line)
    return 0


def add_create_args(parser: argparse.ArgumentParser) -> None:
    d = DEFAULTS
    parser.add_argument("--name")
    parser.add_argument("--run-dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--event-command",
        help="Optional event hook command. Event JSON is passed on stdin and in NNUE_RUN_EVENT_JSON.",
    )

    parser.add_argument("--engine", default=d.engine)
    parser.add_argument("--nnue-file", default=d.nnue_file)
    parser.add_argument("--book", default=d.book)
    parser.add_argument("--runner", default=d.runner)
    parser.add_argument("--python", default=d.python)

    parser.add_argument("--selfplay-games", type=int, default=d.selfplay_games)
    parser.add_argument("--selfplay-shard-games", type=int, default=d.selfplay_shard_games)
    parser.add_argument("--selfplay-concurrency", type=int, default=d.selfplay_concurrency)
    parser.add_argument("--selfplay-threads", type=int, default=d.selfplay_threads)
    parser.add_argument("--selfplay-hash", type=int, default=d.selfplay_hash)
    parser.add_argument("--selfplay-depth", type=int, default=d.selfplay_depth)
    parser.add_argument("--selfplay-seed", type=int, default=d.selfplay_seed)

    parser.add_argument("--skip-plies", type=int, default=d.skip_plies)
    parser.add_argument("--source-max-abs-cp", type=int, default=d.source_max_abs_cp)
    parser.add_argument("--sample-preset", default=d.sample_preset)

    parser.add_argument("--score-engine", default=d.score_engine)
    parser.add_argument("--score-depth", type=int, default=d.score_depth)
    parser.add_argument("--score-shards", type=int, default=d.score_shards)
    parser.add_argument("--score-threads", type=int, default=d.score_threads)
    parser.add_argument("--score-hash", type=int, default=d.score_hash)
    parser.add_argument("--score-max-abs-cp", type=int, default=d.score_max_abs_cp)
    parser.add_argument("--score-progress", type=int, default=d.score_progress)

    parser.add_argument("--max-features", type=int, default=d.max_features)
    parser.add_argument("--pack-progress", type=int, default=d.pack_progress)

    parser.add_argument("--init-net", default=d.init_net)
    parser.add_argument("--objective", default=d.objective,
                        choices=["mse", "huber", "mpe25"])
    parser.add_argument("--target-clamp", type=int, default=d.target_clamp)
    parser.add_argument("--huber-beta", type=int, default=d.huber_beta)
    parser.add_argument("--wdl-lambda", type=float, default=d.wdl_lambda)
    parser.add_argument("--lr", type=float, default=d.lr)
    parser.add_argument("--epochs", type=int, default=d.epochs)
    parser.add_argument("--batch-size", type=int, default=d.batch_size)
    parser.add_argument("--device", default=d.device)
    parser.add_argument("--workers", type=int, default=d.workers)
    parser.add_argument("--val-rows", type=int, default=d.val_rows)
    parser.add_argument("--patience", type=int, default=d.patience)
    parser.add_argument("--select-metric", default=d.select_metric,
                        choices=["loss", "mse", "mae", "sign"])
    parser.add_argument("--weight-decay", type=float, default=d.weight_decay)
    parser.add_argument("--trainable", default=d.trainable,
                        choices=["all", "float-head", "output"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create, resume, and inspect Enyo NNUE candidate runs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Create/train a candidate.")
    add_create_args(create)
    create.set_defaults(func=cmd_create)

    resume = subparsers.add_parser("resume", help="Resume a candidate run.")
    resume.add_argument("run")
    resume.add_argument("--force", action="store_true")
    resume.add_argument(
        "--event-command",
        help="Optional event hook command. Event JSON is passed on stdin and in NNUE_RUN_EVENT_JSON.",
    )
    resume.set_defaults(func=cmd_resume)

    status = subparsers.add_parser("status", help="Show candidate run status.")
    status.add_argument("run")
    status.add_argument("--tail", type=int, default=0)
    status.set_defaults(func=cmd_status)

    report = subparsers.add_parser("report", help="Print candidate run report.")
    report.add_argument("run")
    report.add_argument("--tail", type=int, default=20)
    report.set_defaults(func=cmd_report)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
