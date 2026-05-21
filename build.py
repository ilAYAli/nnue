#!/usr/bin/env python3
"""High-level Enyo NNUE candidate workflow command."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))

from lib.defaults import DEFAULTS, repo_root


def expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser().resolve()


def expand_user(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser()


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


def normalize_key(key: str) -> str:
    return key.strip().lstrip("-").replace("-", "_")


def create_config_path(argv: list[str]) -> str | None:
    if len(argv) < 2 or argv[1] != "create":
        return None
    args = argv[2:]
    for i, item in enumerate(args):
        if item in {"-c", "--config"}:
            if i + 1 >= len(args):
                raise SystemExit(f"{item} requires a path")
            return args[i + 1]
        if item.startswith("--config="):
            return item.split("=", 1)[1]
    return None


def normalize_argv(argv: list[str]) -> list[str]:
    if len(argv) > 1 and (argv[1] in {"-c", "--config"} or argv[1].startswith("--config=")):
        return [argv[0], "create", *argv[1:]]
    return argv


def load_create_arg_defaults(path: str | Path | None) -> dict[str, object]:
    if not path:
        return {}

    config_path = expand_path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{config_path}: build config must be a JSON object")

    validation = data.get("validation", {})
    validation_engine = ""
    if isinstance(validation, dict):
        validation_engine = str(validation.get("engine", ""))

    if "create" in data:
        create = data["create"]
    elif "create_args" in data:
        create = data["create_args"]
    else:
        create = {
            key: value for key, value in data.items()
            if not key.startswith("_")
            and key not in {
                "description",
                "notes",
                "rationale",
                "validation",
                "metadata",
            }
        }

    if not isinstance(create, dict):
        raise SystemExit(f"{config_path}: 'create' must be a JSON object")

    allowed = {field.name for field in fields(DEFAULTS)}
    allowed.update({
        "name",
        "run_dir",
        "dry_run",
        "force",
        "event_command",
        "validation_engine",
    })
    out: dict[str, object] = {"config": str(config_path)}
    if validation_engine:
        out["validation_engine"] = validation_engine
    for raw_key, value in create.items():
        key = normalize_key(str(raw_key))
        if key in {"command", "config", "func"}:
            continue
        if key not in allowed:
            raise SystemExit(f"{config_path}: unknown create argument '{raw_key}'")
        out[key] = value
    return out


def config_default(overrides: dict[str, object], key: str, fallback: object) -> object:
    return overrides.get(key, fallback)


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
            if key not in {"command", "config", "func"}
        },
        "steps": steps,
    }
    if args.validation_engine:
        config["validation"] = {
            "engine": args.validation_engine,
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


def add_create_args(
    parser: argparse.ArgumentParser,
    overrides: dict[str, object] | None = None,
) -> None:
    d = DEFAULTS
    cfg = overrides or {}
    value = lambda key, fallback: config_default(cfg, key, fallback)

    parser.add_argument("-c", "--config", default=value("config", None),
                        help="JSON create-argument config. CLI args override it.")
    parser.add_argument("--name", default=value("name", None))
    parser.add_argument("--run-dir", default=value("run_dir", None))
    parser.add_argument("--dry-run", action="store_true",
                        default=value("dry_run", False))
    parser.add_argument("--force", action="store_true",
                        default=value("force", False))
    parser.add_argument(
        "--event-command",
        default=value("event_command", None),
        help="Optional event hook command. Event JSON is passed on stdin and in NNUE_RUN_EVENT_JSON.",
    )
    parser.add_argument(
        "--validation-engine",
        default=value("validation_engine", ""),
        help="Engine path used in generated validation commands.",
    )

    parser.add_argument("--engine", default=value("engine", d.engine))
    parser.add_argument("--nnue-file", default=value("nnue_file", d.nnue_file))
    parser.add_argument("--book", default=value("book", d.book))
    parser.add_argument("--runner", default=value("runner", d.runner))
    parser.add_argument("--python", default=value("python", d.python))

    parser.add_argument("--selfplay-games", type=int, default=value("selfplay_games", d.selfplay_games))
    parser.add_argument("--selfplay-shard-games", type=int, default=value("selfplay_shard_games", d.selfplay_shard_games))
    parser.add_argument("--selfplay-concurrency", type=int, default=value("selfplay_concurrency", d.selfplay_concurrency))
    parser.add_argument("--selfplay-threads", type=int, default=value("selfplay_threads", d.selfplay_threads))
    parser.add_argument("--selfplay-hash", type=int, default=value("selfplay_hash", d.selfplay_hash))
    parser.add_argument("--selfplay-depth", type=int, default=value("selfplay_depth", d.selfplay_depth))
    parser.add_argument("--selfplay-seed", type=int, default=value("selfplay_seed", d.selfplay_seed))

    parser.add_argument("--skip-plies", type=int, default=value("skip_plies", d.skip_plies))
    parser.add_argument("--source-max-abs-cp", type=int, default=value("source_max_abs_cp", d.source_max_abs_cp))
    parser.add_argument("--sample-preset", default=value("sample_preset", d.sample_preset))

    parser.add_argument("--score-engine", default=value("score_engine", d.score_engine))
    parser.add_argument("--score-depth", type=int, default=value("score_depth", d.score_depth))
    parser.add_argument("--score-shards", type=int, default=value("score_shards", d.score_shards))
    parser.add_argument("--score-threads", type=int, default=value("score_threads", d.score_threads))
    parser.add_argument("--score-hash", type=int, default=value("score_hash", d.score_hash))
    parser.add_argument("--score-max-abs-cp", type=int, default=value("score_max_abs_cp", d.score_max_abs_cp))
    parser.add_argument("--score-progress", type=int, default=value("score_progress", d.score_progress))

    parser.add_argument("--max-features", type=int, default=value("max_features", d.max_features))
    parser.add_argument("--pack-progress", type=int, default=value("pack_progress", d.pack_progress))

    parser.add_argument("--init-net", default=value("init_net", d.init_net))
    parser.add_argument("--objective", default=value("objective", d.objective),
                        choices=["mse", "huber", "mpe25"])
    parser.add_argument("--target-clamp", type=int, default=value("target_clamp", d.target_clamp))
    parser.add_argument("--huber-beta", type=int, default=value("huber_beta", d.huber_beta))
    parser.add_argument("--wdl-lambda", type=float, default=value("wdl_lambda", d.wdl_lambda))
    parser.add_argument("--lr", type=float, default=value("lr", d.lr))
    parser.add_argument("--epochs", type=int, default=value("epochs", d.epochs))
    parser.add_argument("--batch-size", type=int, default=value("batch_size", d.batch_size))
    parser.add_argument("--device", default=value("device", d.device))
    parser.add_argument("--workers", type=int, default=value("workers", d.workers))
    parser.add_argument("--val-rows", type=int, default=value("val_rows", d.val_rows))
    parser.add_argument("--patience", type=int, default=value("patience", d.patience))
    parser.add_argument("--select-metric", default=value("select_metric", d.select_metric),
                        choices=["loss", "mse", "mae", "sign"])
    parser.add_argument("--weight-decay", type=float, default=value("weight_decay", d.weight_decay))
    parser.add_argument("--trainable", default=value("trainable", d.trainable),
                        choices=["all", "float-head", "output", "phase-head"])


def build_parser(create_defaults: dict[str, object] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create and inspect Enyo NNUE candidate runs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Create/train a candidate.")
    add_create_args(create, create_defaults)
    create.set_defaults(func=cmd_create)

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
    argv = normalize_argv(sys.argv)
    create_defaults = load_create_arg_defaults(create_config_path(argv))
    parser = build_parser(create_defaults)
    args = parser.parse_args(argv[1:])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
