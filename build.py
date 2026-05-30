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
    allowed.update({"name", "run_dir", "dry_run", "force", "event_command"})
    out: dict[str, object] = {"config": str(config_path)}
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
    python = str(expand_user(args.python))
    steps: list[dict] = []

    if args.backend == "pytorch":
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
                    "--python", python,
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
                    "--prefetch-factor", str(args.prefetch_factor),
                    "--amp", args.amp,
                    "--torch-compile" if args.torch_compile else "--no-torch-compile",
                    "--dataset-in-memory" if args.dataset_in_memory else "--no-dataset-in-memory",
                    "--patience", str(args.patience),
                    "--val-rows", str(args.val_rows),
                    "--trainable", args.trainable,
                    "--python", python,
                    "--out", f"{candidate_dir}/model.pt",
                    "--out-nn", f"{candidate_dir}/model.nn",
                ],
            },
        ])
    elif args.backend == "bullet":
        if args.bullet_loader == "direct":
            if args.bullet_data:
                bullet_data = str(expand_user(args.bullet_data))
            else:
                if not args.bullet_source_jsonl:
                    raise SystemExit(
                        "backend=bullet direct loader requires bullet_source_jsonl or bullet_data"
                    )
                bullet_text = "{assets}/bullet.txt"
                bullet_data = "{assets}/bullet.data"
                steps.extend([
                    {
                        "name": "bullet_text",
                        "command": [
                            python, tool("bullet/jsonl_to_bullet_text.py"),
                            "--input", str(expand_path(args.bullet_source_jsonl)),
                            "--output", bullet_text,
                            "--limit", str(args.bullet_limit),
                            "--max-abs-cp", str(args.bullet_max_abs_cp),
                            *(
                                ["--enyo-runtime-target"]
                                if args.bullet_enyo_runtime_target else []
                            ),
                        ],
                    },
                    {
                        "name": "bullet_format",
                        "command": [
                            python, tool("bullet/bullet.py"), "format",
                            "--input", bullet_text,
                            "--output", bullet_data,
                            "--bullet-manifest", str(expand_path(args.bullet_manifest)),
                            "--validate",
                        ],
                    },
                ])
        elif args.bullet_loader == "sfbinpack":
            if not args.bullet_data:
                raise SystemExit("backend=bullet sfbinpack loader requires bullet_data")
            bullet_data = str(expand_user(args.bullet_data))
        else:
            raise SystemExit(f"unsupported bullet_loader={args.bullet_loader}")

        steps.append({
            "name": "bullet_train",
            "command": [
                python, tool("bullet/bullet.py"), "train",
                "--data", bullet_data,
                "--loader", args.bullet_loader,
                "--sfbinpack-buffer-mb", str(args.bullet_sfbinpack_buffer_mb),
                "--sfbinpack-min-ply", str(args.bullet_sfbinpack_min_ply),
                "--sfbinpack-max-abs-cp", str(args.bullet_sfbinpack_max_abs_cp),
                (
                    "--sfbinpack-quiet-only"
                    if args.bullet_sfbinpack_quiet_only
                    else "--no-sfbinpack-quiet-only"
                ),
                "--out-dir", f"{candidate_dir}/checkpoints",
                "--net-id", name,
                "--cargo-target-dir", "{run}/cargo-target",
                "--mode", args.bullet_mode,
                "--accelerator", args.bullet_accelerator,
                "--cuda-path", args.bullet_cuda_path,
                "--cuda-arch", args.bullet_cuda_arch,
                "--hidden", str(args.bullet_hidden),
                "--l2", str(args.bullet_l2),
                "--batch-size", str(args.bullet_batch_size),
                "--batches", str(args.bullet_batches),
                "--superbatches", str(args.bullet_superbatches),
                "--threads", str(args.bullet_threads),
                "--wdl", str(args.bullet_wdl),
                "--lr", str(args.bullet_lr),
                "--final-lr", str(args.bullet_final_lr),
                "--enyo-l0-std", str(args.bullet_enyo_l0_std),
                "--enyo-l1-std", str(args.bullet_enyo_l1_std),
                "--enyo-l1-export-scale", str(args.bullet_enyo_l1_export_scale),
                (
                    "--enyo-input-factorizer"
                    if args.bullet_enyo_input_factorizer
                    else "--no-enyo-input-factorizer"
                ),
                "--enyo-input-buckets", str(args.bullet_enyo_input_buckets),
                "--enyo-runtime-input-buckets",
                str(args.bullet_enyo_runtime_input_buckets),
                "--eval-scale", str(args.bullet_eval_scale),
                "--save-rate", str(args.bullet_save_rate),
                *(
                    ["--init-weights", str(expand_path(args.bullet_init_weights))]
                    if args.bullet_init_weights else []
                ),
                "--trainable", args.bullet_trainable,
                "--weight-decay", str(args.bullet_weight_decay),
                *(["--export-init-only"] if args.bullet_export_init_only else []),
            ],
        })
        if args.bullet_static_data:
            steps.append({
                "name": "validate_bullet_static",
                "command": [
                    python, tool("validate/eval_dataset.py"),
                    "--net", f"{candidate_dir}/model.nn",
                    "--data", str(expand_path(args.bullet_static_data)),
                    "--rows", str(args.bullet_static_rows),
                    "--buckets",
                    "--sources",
                ],
            })
    else:
        raise SystemExit(f"unsupported backend={args.backend}")

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

    parser.add_argument("--engine", default=value("engine", d.engine))
    parser.add_argument("--nnue-file", default=value("nnue_file", d.nnue_file))
    parser.add_argument("--book", default=value("book", d.book))
    parser.add_argument("--runner", default=value("runner", d.runner))
    parser.add_argument("--python", default=value("python", d.python))
    parser.add_argument("--backend", default=value("backend", d.backend),
                        choices=["pytorch", "bullet"])

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
    parser.add_argument("--prefetch-factor", type=int, default=value("prefetch_factor", d.prefetch_factor))
    parser.add_argument("--amp", default=value("amp", d.amp), choices=["off", "bf16"])
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction,
                        default=value("torch_compile", d.torch_compile))
    parser.add_argument("--dataset-in-memory", action=argparse.BooleanOptionalAction,
                        default=value("dataset_in_memory", d.dataset_in_memory))
    parser.add_argument("--val-rows", type=int, default=value("val_rows", d.val_rows))
    parser.add_argument("--patience", type=int, default=value("patience", d.patience))
    parser.add_argument("--select-metric", default=value("select_metric", d.select_metric),
                        choices=["loss", "mse", "mae", "sign"])
    parser.add_argument("--weight-decay", type=float, default=value("weight_decay", d.weight_decay))
    parser.add_argument("--trainable", default=value("trainable", d.trainable),
                        choices=["all", "float-head", "output"])

    parser.add_argument("--bullet-source-jsonl", default=value("bullet_source_jsonl", d.bullet_source_jsonl))
    parser.add_argument("--bullet-data", default=value("bullet_data", d.bullet_data))
    parser.add_argument("--bullet-manifest", default=value("bullet_manifest", d.bullet_manifest))
    parser.add_argument("--bullet-loader", default=value("bullet_loader", d.bullet_loader),
                        choices=["direct", "sfbinpack"])
    parser.add_argument("--bullet-limit", type=int, default=value("bullet_limit", d.bullet_limit))
    parser.add_argument("--bullet-max-abs-cp", type=int, default=value("bullet_max_abs_cp", d.bullet_max_abs_cp))
    parser.add_argument("--bullet-enyo-runtime-target", action=argparse.BooleanOptionalAction,
                        default=value("bullet_enyo_runtime_target", d.bullet_enyo_runtime_target))
    parser.add_argument("--bullet-sfbinpack-buffer-mb", type=int, default=value("bullet_sfbinpack_buffer_mb", d.bullet_sfbinpack_buffer_mb))
    parser.add_argument("--bullet-sfbinpack-min-ply", type=int, default=value("bullet_sfbinpack_min_ply", d.bullet_sfbinpack_min_ply))
    parser.add_argument("--bullet-sfbinpack-max-abs-cp", type=int, default=value("bullet_sfbinpack_max_abs_cp", d.bullet_sfbinpack_max_abs_cp))
    parser.add_argument("--bullet-sfbinpack-quiet-only", action=argparse.BooleanOptionalAction,
                        default=value("bullet_sfbinpack_quiet_only", d.bullet_sfbinpack_quiet_only))
    parser.add_argument("--bullet-mode", default=value("bullet_mode", d.bullet_mode),
                        choices=["reckless", "enyo"])
    parser.add_argument("--bullet-accelerator", default=value("bullet_accelerator", d.bullet_accelerator),
                        choices=["cuda", "rocm"])
    parser.add_argument("--bullet-cuda-path", default=value("bullet_cuda_path", d.bullet_cuda_path))
    parser.add_argument("--bullet-cuda-arch", default=value("bullet_cuda_arch", d.bullet_cuda_arch))
    parser.add_argument("--bullet-hidden", type=int, default=value("bullet_hidden", d.bullet_hidden))
    parser.add_argument("--bullet-l2", type=int, default=value("bullet_l2", d.bullet_l2))
    parser.add_argument("--bullet-batch-size", type=int, default=value("bullet_batch_size", d.bullet_batch_size))
    parser.add_argument("--bullet-batches", type=int, default=value("bullet_batches", d.bullet_batches))
    parser.add_argument("--bullet-superbatches", type=int, default=value("bullet_superbatches", d.bullet_superbatches))
    parser.add_argument("--bullet-threads", type=int, default=value("bullet_threads", d.bullet_threads))
    parser.add_argument("--bullet-wdl", type=float, default=value("bullet_wdl", d.bullet_wdl))
    parser.add_argument("--bullet-lr", type=float, default=value("bullet_lr", d.bullet_lr))
    parser.add_argument("--bullet-final-lr", type=float, default=value("bullet_final_lr", d.bullet_final_lr))
    parser.add_argument("--bullet-enyo-l0-std", type=float, default=value("bullet_enyo_l0_std", d.bullet_enyo_l0_std))
    parser.add_argument("--bullet-enyo-l1-std", type=float, default=value("bullet_enyo_l1_std", d.bullet_enyo_l1_std))
    parser.add_argument("--bullet-enyo-l1-export-scale", type=float, default=value("bullet_enyo_l1_export_scale", d.bullet_enyo_l1_export_scale))
    parser.add_argument("--bullet-enyo-input-factorizer", action=argparse.BooleanOptionalAction,
                        default=value("bullet_enyo_input_factorizer", d.bullet_enyo_input_factorizer))
    parser.add_argument("--bullet-enyo-input-buckets", type=int,
                        default=value("bullet_enyo_input_buckets", d.bullet_enyo_input_buckets),
                        choices=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--bullet-enyo-runtime-input-buckets", type=int,
                        default=value("bullet_enyo_runtime_input_buckets", d.bullet_enyo_runtime_input_buckets),
                        choices=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--bullet-eval-scale", type=float, default=value("bullet_eval_scale", d.bullet_eval_scale))
    parser.add_argument("--bullet-save-rate", type=int, default=value("bullet_save_rate", d.bullet_save_rate))
    parser.add_argument("--bullet-init-weights", default=value("bullet_init_weights", d.bullet_init_weights))
    parser.add_argument("--bullet-trainable", default=value("bullet_trainable", d.bullet_trainable),
                        choices=["all", "input", "float-head", "output"])
    parser.add_argument("--bullet-weight-decay", type=float, default=value("bullet_weight_decay", d.bullet_weight_decay))
    parser.add_argument("--bullet-export-init-only", action="store_true",
                        default=value("bullet_export_init_only", d.bullet_export_init_only))
    parser.add_argument("--bullet-static-data", default=value("bullet_static_data", d.bullet_static_data))
    parser.add_argument("--bullet-static-rows", type=int, default=value("bullet_static_rows", d.bullet_static_rows))


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
