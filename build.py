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
from lib.events import emit_event


def expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser().resolve()


def expand_user(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser()


def tool(path: str) -> str:
    return str(repo_root() / "tools" / path)


def run(command: list[str], *, dry_run: bool = False, echo: bool = False) -> int:
    if dry_run or echo:
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
        config_path = None
        if argv[1].startswith("--config="):
            config_path = argv[1].split("=", 1)[1]
        elif len(argv) > 2:
            config_path = argv[2]
        command = "create"
        if config_path:
            data = json.loads(expand_path(config_path).read_text(encoding="utf-8"))
            if "target_build" in data and "create" not in data and "create_args" not in data:
                command = "target-build"
            elif "target_score" in data and "create" not in data and "create_args" not in data:
                command = "target-score"
        return [argv[0], command, *argv[1:]]
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

    if create.get("disabled"):
        description = str(data.get("description") or "no active create config")
        raise SystemExit(f"{config_path}: create config is disabled: {description}")

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


def default_build_config_path() -> Path:
    return repo_root() / "build.json"


def run_dir_from_build_config(path: str | Path | None = None) -> Path:
    config_path = expand_path(path or default_build_config_path())
    if not config_path.exists():
        raise SystemExit(
            f"{config_path}: missing; pass a run path or create build.json")

    create_args = load_create_arg_defaults(config_path)
    name = create_args.get("name")
    if not name:
        raise SystemExit(f"{config_path}: create.name is required")

    run_dir = create_args.get("run_dir")
    return run_dir_for(str(name), str(run_dir) if run_dir else None)


def resolve_run_arg(run: str | None) -> Path:
    if run:
        return expand_path(run)
    return run_dir_from_build_config()


def write_config(run_dir: Path, config: dict) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def create_config(args: argparse.Namespace) -> dict:
    name = args.name or default_name()
    run_dir = run_dir_for(name, args.run_dir)
    candidate_dir = f"{{train}}/{name}"
    labeled_jsonl = str(args.labeled_jsonl or "")
    external_bullet_data = args.backend == "bullet" and (
        bool(args.bullet_data) or bool(args.bullet_lichess_eval_input)
    )
    pack_input = "{score}/labeled.jsonl"
    steps = []

    if labeled_jsonl:
        pack_input = str(expand_path(labeled_jsonl))
    elif external_bullet_data:
        pass
    else:
        steps.extend([
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
        ])

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

        steps.append({
            "name": "score_merge",
            "command": [
                "bash", "-lc",
                "cat \"$1\"/shards/label.*.jsonl > \"$1\"/labeled.jsonl && wc -l \"$1\"/labeled.jsonl > \"$1\"/labeled.wc",
                "merge-score", "{score}",
            ],
        })

    if args.blend_extra_jsonl:
        blended_input = "{assets}/labeled_blend.jsonl"
        steps.append({
            "name": "blend_labeled",
            "command": [
                tool("posgen/blend_labeled_jsonl.py"),
                "--base", pack_input,
                "--extra", str(expand_path(args.blend_extra_jsonl)),
                "--output", blended_input,
                "--base-limit", str(args.blend_base_limit),
                "--extra-repeat", str(args.blend_extra_repeat),
                "--shuffle-seed", str(args.blend_shuffle_seed),
            ],
        })
        pack_input = blended_input

    if args.backend == "pytorch":
        train_command = [
            tool("train/train.py"), "run",
            "--data", "{pack}/train",
        ]
        if args.init_net:
            train_command.extend([
                "--init-from-nn", str(expand_path(args.init_net)),
            ])
        else:
            train_command.extend([
                "--init", args.init,
            ])
        train_command.extend([
            "--objective", args.objective,
            "--forward", args.forward,
            "--huber-beta", str(args.huber_beta),
            "--select-metric", args.select_metric,
            "--wdl-lambda", str(args.wdl_lambda),
            "--sign-loss-weight", str(args.sign_loss_weight),
            "--sign-loss-scale", str(args.sign_loss_scale),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--weight-decay", str(args.weight_decay),
            "--input-lr-mult", str(args.input_lr_mult),
            "--l1-lr-mult", str(args.l1_lr_mult),
            "--dense-lr-mult", str(args.dense_lr_mult),
            "--target-clamp", str(args.target_clamp),
            "--device", args.device,
            "--workers", str(args.workers),
            "--patience", str(args.patience),
            "--max-rows", str(args.max_rows),
            "--skip-rows", str(args.skip_rows),
            "--val-rows", str(args.val_rows),
            "--trainable", args.trainable,
            "--grad-norm-every", str(args.grad_norm_every),
            "--python", str(expand_user(args.python)),
            "--out", f"{candidate_dir}/model.pt",
            "--out-nn", f"{candidate_dir}/model.nn",
        ])

        steps.extend([
            {
                "name": "pack",
                "command": [
                    tool("pack/pack.py"), "build",
                    "--input", pack_input,
                    "--out-dir", "{pack}/train",
                    "--skip", str(args.pack_skip),
                    "--limit", str(args.pack_limit),
                    "--max-features", str(args.max_features),
                    "--progress", str(args.pack_progress),
                    "--python", str(expand_user(args.python)),
                ],
            },
            {
                "name": "train",
                "command": train_command,
            },
        ])
    elif args.backend == "material-head":
        if not args.init_net:
            raise SystemExit("backend=material-head requires --init-net")
        material_trainable = "output" if args.trainable == "all" else args.trainable
        if material_trainable not in {"output", "float-head"}:
            raise SystemExit(
                "backend=material-head supports --trainable output|float-head")
        train_command = [
            str(expand_user(args.python)),
            tool("train/train_material_head.py"),
            "--data", "{pack}/train",
            "--init-from-nn", str(expand_path(args.init_net)),
            "--out", f"{candidate_dir}/model.pt",
            "--out-nn", f"{candidate_dir}/model.nn",
            "--bucket-mode", args.bucket_mode,
            "--trainable", material_trainable,
            "--objective", args.objective,
            "--huber-beta", str(args.huber_beta),
            "--select-metric", args.select_metric,
            "--wdl-lambda", str(args.wdl_lambda),
            "--sign-loss-weight", str(args.sign_loss_weight),
            "--sign-loss-scale", str(args.sign_loss_scale),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--weight-decay", str(args.weight_decay),
            "--target-clamp", str(args.target_clamp),
            "--device", args.device,
            "--workers", str(args.workers),
            "--patience", str(args.patience),
            "--max-rows", str(args.max_rows),
            "--skip-rows", str(args.skip_rows),
            "--val-rows", str(args.val_rows),
        ]

        steps.extend([
            {
                "name": "pack",
                "command": [
                    tool("pack/pack.py"), "build",
                    "--input", pack_input,
                    "--out-dir", "{pack}/train",
                    "--skip", str(args.pack_skip),
                    "--limit", str(args.pack_limit),
                    "--max-features", str(args.max_features),
                    "--progress", str(args.pack_progress),
                    "--python", str(expand_user(args.python)),
                ],
            },
            {
                "name": "train_material_head",
                "command": train_command,
            },
        ])
    elif args.backend == "pairwise":
        if not args.pairwise_scores_csv and not args.pairwise_pairs_jsonl:
            raise SystemExit(
                "backend=pairwise requires pairwise_scores_csv or pairwise_pairs_jsonl")

        train_command = [
            str(expand_user(args.python)),
            tool("train/train_pairwise.py"),
            "--data", "{pack}/train",
            "--out", f"{candidate_dir}/model.pt",
            "--out-nn", f"{candidate_dir}/model.nn",
            "--forward", args.forward,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--pair-batch-size", str(args.pairwise_pair_batch_size),
            "--lr", str(args.lr),
            "--weight-decay", str(args.weight_decay),
            "--input-lr-mult", str(args.input_lr_mult),
            "--l1-lr-mult", str(args.l1_lr_mult),
            "--dense-lr-mult", str(args.dense_lr_mult),
            "--huber-beta", str(args.huber_beta),
            "--broad-weight", str(args.pairwise_broad_weight),
            "--broad-target", str(args.pairwise_broad_target),
            "--pair-beta", str(args.pairwise_pair_beta),
            "--pair-weight", str(args.pairwise_pair_weight),
            "--target-clamp", str(args.target_clamp),
            "--min-target-margin", str(args.pairwise_min_target_margin),
            "--max-target-margin", str(args.pairwise_max_target_margin),
            "--device", args.device,
            "--workers", str(args.workers),
            "--max-rows", str(args.max_rows),
            "--skip-rows", str(args.skip_rows),
            "--grad-norm-every", str(args.grad_norm_every),
        ]
        if args.init_net:
            train_command.extend([
                "--init-from-nn", str(expand_path(args.init_net)),
            ])
        else:
            train_command.extend([
                "--init", args.init,
            ])
        if args.pairwise_scores_csv:
            train_command.extend([
                "--scores-csv", str(expand_path(args.pairwise_scores_csv)),
            ])
        if args.pairwise_candidate_moves_csv:
            train_command.extend([
                "--candidate-moves-csv",
                str(expand_path(args.pairwise_candidate_moves_csv)),
            ])
        if args.pairwise_pairs_jsonl:
            train_command.extend([
                "--pairs", str(expand_path(args.pairwise_pairs_jsonl)),
            ])
        if args.pairwise_loss_weight_by_cp:
            train_command.append("--loss-weight-by-cp")

        steps.extend([
            {
                "name": "pack",
                "command": [
                    tool("pack/pack.py"), "build",
                    "--input", pack_input,
                    "--out-dir", "{pack}/train",
                    "--skip", str(args.pack_skip),
                    "--limit", str(args.pack_limit),
                    "--max-features", str(args.max_features),
                    "--progress", str(args.pack_progress),
                    "--python", str(expand_user(args.python)),
                ],
            },
            {
                "name": "train_pairwise",
                "command": train_command,
            },
        ])
    elif args.backend == "search-aware":
        if not args.search_targets_jsonl:
            raise SystemExit("backend=search-aware requires --search-targets-jsonl")

        train_command = [
            str(expand_user(args.python)),
            tool("train/train_search_aware.py"),
            "--data", "{pack}/train",
            "--targets", str(expand_path(args.search_targets_jsonl)),
            "--out", f"{candidate_dir}/model.pt",
            "--out-nn", f"{candidate_dir}/model.nn",
            "--forward", args.forward,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--search-target-batch-size", str(args.search_target_batch_size),
            "--lr", str(args.lr),
            "--weight-decay", str(args.weight_decay),
            "--input-lr-mult", str(args.input_lr_mult),
            "--l1-lr-mult", str(args.l1_lr_mult),
            "--dense-lr-mult", str(args.dense_lr_mult),
            "--huber-beta", str(args.huber_beta),
            "--target-clamp", str(args.target_clamp),
            "--search-broad-weight", str(args.search_broad_weight),
            "--search-broad-target", args.search_broad_target,
            "--search-target-warmup-epochs", str(args.search_target_warmup_epochs),
            "--search-warmup-broad-weight", str(args.search_warmup_broad_weight),
            "--search-broad-ramp-epochs", str(args.search_broad_ramp_epochs),
            "--search-margin-weight", str(args.search_margin_weight),
            "--search-policy-weight", str(args.search_policy_weight),
            "--search-margin-beta", str(args.search_margin_beta),
            "--search-policy-temperature-cp", str(args.search_policy_temperature_cp),
            "--search-score-mode", args.search_score_mode,
            "--search-max-gap-cp", str(args.search_max_gap_cp),
            "--search-max-moves", str(args.search_max_moves),
            "--search-target-limit", str(args.search_target_limit),
            "--search-required-tags", str(args.search_required_tags),
            "--search-tag-weights", str(args.search_tag_weights),
            "--device", args.device,
            "--workers", str(args.workers),
            "--max-rows", str(args.max_rows),
            "--skip-rows", str(args.skip_rows),
            "--grad-norm-every", str(args.grad_norm_every),
            "--seed", str(args.selfplay_seed),
            "--trainable", args.trainable,
        ]
        if args.search_target_shuffle:
            train_command.append("--search-target-shuffle")
        if args.search_select_best_target:
            train_command.append("--search-select-best-target")
        if args.init_net:
            train_command.extend([
                "--init-from-nn", str(expand_path(args.init_net)),
            ])
        else:
            train_command.extend([
                "--init", args.init,
            ])
        if args.search_broad_target_net:
            train_command.extend([
                "--search-broad-target-net",
                str(expand_path(args.search_broad_target_net)),
            ])

        steps.extend([
            {
                "name": "pack",
                "command": [
                    tool("pack/pack.py"), "build",
                    "--input", pack_input,
                    "--out-dir", "{pack}/train",
                    "--skip", str(args.pack_skip),
                    "--limit", str(args.pack_limit),
                    "--max-features", str(args.max_features),
                    "--progress", str(args.pack_progress),
                    "--python", str(expand_user(args.python)),
                ],
            },
            {
                "name": "train_search_aware",
                "command": train_command,
            },
            {
                "name": "validate_search_targets_model",
                "command": [
                    str(expand_user(args.python)),
                    tool("validate/search_target_model_gate.py"),
                    "--targets", str(expand_path(args.search_targets_jsonl)),
                    "--net", f"{candidate_dir}/model.nn",
                    "--pt", f"{candidate_dir}/model.pt",
                    "--forward", args.forward,
                    "--search-score-mode", args.search_score_mode,
                    "--device", args.search_model_gate_device,
                    "--batch-size", str(args.search_target_batch_size),
                    "--target-limit", str(args.search_target_limit),
                    "--required-tags", str(args.search_required_tags),
                    "--tag-weights", str(args.search_tag_weights),
                    "--max-moves", str(args.search_max_moves),
                    "--max-gap-cp", str(args.search_max_gap_cp),
                    "--fail-if-net-top1-below", str(args.search_model_gate_min_top1),
                    "--fail-if-pt-top1-below", str(args.search_model_gate_min_top1),
                ],
            },
        ])
    elif args.backend == "bullet":
        bullet_text = "{pack}/bullet/enyo.txt"
        lichess_eval_input = str(args.bullet_lichess_eval_input or "")
        if args.bullet_data and lichess_eval_input:
            raise SystemExit(
                "bullet_data and bullet_lichess_eval_input are mutually exclusive")
        if lichess_eval_input and args.bullet_loader != "direct":
            raise SystemExit(
                "bullet_lichess_eval_input produces BulletFormat data; use "
                "bullet_loader=direct")
        bullet_data = str(args.bullet_data or "{pack}/bullet/enyo.data")
        bullet_cargo_target_dir = (
            str(expand_user(args.bullet_cargo_target_dir))
            if args.bullet_cargo_target_dir
            else "{run}/cargo-target"
        )
        bullet_train = [
            tool("bullet/bullet.py"), "train",
            "--data", bullet_data,
            "--loader", str(args.bullet_loader),
            "--sfbinpack-buffer-mb", str(args.bullet_sfbinpack_buffer_mb),
            "--sfbinpack-min-ply", str(args.bullet_sfbinpack_min_ply),
            "--sfbinpack-max-abs-cp", str(args.bullet_sfbinpack_max_abs_cp),
            "--sfbinpack-quiet-only" if args.bullet_sfbinpack_quiet_only else "--no-sfbinpack-quiet-only",
            "--out-dir", f"{candidate_dir}/checkpoints",
            "--net-id", name,
            "--cargo-target-dir", bullet_cargo_target_dir,
            "--mode", str(args.bullet_mode),
            "--cuda-arch", str(args.bullet_cuda_arch),
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
            "--enyo-input-buckets", str(args.bullet_enyo_input_buckets),
            "--eval-scale", str(args.bullet_eval_scale),
            "--save-rate", str(args.bullet_save_rate),
            "--trainable", str(args.trainable),
            "--weight-decay", str(args.weight_decay),
        ]
        if args.bullet_export_init_only:
            bullet_train.append("--export-init-only")
        if args.init_net and args.bullet_mode == "enyo":
            bullet_init_weights = "{assets}/bullet_init_weights.bin"
            steps.append({
                "name": "bullet_init_weights",
                "command": [
                    str(expand_user(args.python)),
                    tool("bullet/enyo_nn_to_bullet_weights.py"),
                    "--input", str(expand_path(args.init_net)),
                    "--output", bullet_init_weights,
                    "--eval-scale", str(args.bullet_eval_scale),
                    "--l1-export-scale", str(args.bullet_enyo_l1_export_scale),
                ] + (["--legacy-inputs"] if args.bullet_enyo_input_buckets == 16 else []),
            })
            bullet_train.extend(["--init-weights", bullet_init_weights])
        elif args.init_net and args.bullet_mode != "enyo":
            raise SystemExit(
                "Bullet init-net is currently supported only for bullet-mode=enyo; "
                "Reckless-like checkpoint layout cannot preserve Enyo .nn weights yet.")
        if args.bullet_cuda_path:
            bullet_train.extend(["--cuda-path", str(expand_user(args.bullet_cuda_path))])

        if args.bullet_mode == "enyo" and not args.bullet_data and not lichess_eval_input:
            steps.append({
                "name": "pack",
                "command": [
                    tool("pack/pack.py"), "build",
                    "--input", pack_input,
                    "--out-dir", "{pack}/train",
                    "--skip", str(args.pack_skip),
                    "--limit", str(args.bullet_rows),
                    "--max-features", str(args.max_features),
                    "--progress", str(args.pack_progress),
                    "--python", str(expand_user(args.python)),
                ],
            })

        if not args.bullet_data:
            if lichess_eval_input:
                bucket_specs = str(args.bullet_lichess_eval_buckets or "")
                if isinstance(args.bullet_lichess_eval_buckets, list):
                    bucket_list = [str(x) for x in args.bullet_lichess_eval_buckets]
                else:
                    bucket_list = [
                        item.strip() for item in bucket_specs.split(";")
                        if item.strip()
                    ]
                bullet_text_command = [
                    str(expand_user(args.python)),
                    tool("score/import_lichess_eval.py"),
                    "--input", str(expand_path(lichess_eval_input)),
                    "--output", bullet_text,
                    "--rows", str(args.bullet_rows),
                    "--min-depth", str(args.bullet_lichess_eval_min_depth),
                    "--min-knodes", str(args.bullet_lichess_eval_min_knodes),
                    "--max-abs-cp", str(args.bullet_max_abs_cp),
                    "--max-input-rows", str(args.bullet_lichess_eval_max_input_rows),
                    "--seed", str(args.bullet_lichess_eval_seed),
                    "--progress", str(args.pack_progress),
                    "--output-format", "bullet-text",
                ] + (["--enyo-runtime-target"] if args.bullet_mode == "enyo" else [])
                if args.bullet_lichess_eval_unique_fen:
                    bullet_text_command.append("--unique-fen")
                if args.bullet_lichess_eval_stop_when_full:
                    bullet_text_command.append("--stop-when-full")
                for bucket in bucket_list:
                    bullet_text_command.extend(["--bucket", bucket])
            else:
                bullet_text_command = [
                    tool("bullet/jsonl_to_bullet_text.py"),
                    "--input", pack_input,
                    "--output", bullet_text,
                    "--limit", str(args.bullet_rows),
                    "--max-abs-cp", str(args.bullet_max_abs_cp),
                ] + (["--enyo-runtime-target"] if args.bullet_mode == "enyo" else [])
            steps.extend([
                {
                    "name": "bullet_text",
                    "command": bullet_text_command,
                },
                {
                    "name": "bullet_format",
                    "command": [
                        tool("bullet/bullet.py"), "format",
                        "--input", bullet_text,
                        "--output", bullet_data,
                        "--bullet-manifest", str(expand_user(args.bullet_manifest)),
                        "--validate",
                    ],
                },
            ])
        steps.append({
            "name": "bullet_train",
            "command": bullet_train,
        })
    else:
        raise SystemExit(f"unknown backend: {args.backend}")

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
    run_dir = resolve_run_arg(args.run)
    return run([
        sys.executable, tool("pipeline/pipeline.py"), "status",
        str(run_dir),
        "--tail", str(args.tail),
    ])


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = resolve_run_arg(args.run)
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


def target_build_run_dir(config: dict) -> Path:
    metadata = config.get("metadata", {})
    target_build = config.get("target_build", {})
    if not isinstance(metadata, dict) or not isinstance(target_build, dict):
        raise SystemExit("build config requires metadata and target_build objects")
    run_dir = metadata.get("run_dir") or target_build.get("run_dir")
    if run_dir:
        return expand_path(str(run_dir))
    output = target_build.get("output_deduped") or target_build.get("output_full")
    if output:
        return expand_path(str(output)).parent
    experiment = metadata.get("experiment", "target-build")
    return expand_path(DEFAULTS.run_base) / str(experiment)


def target_score_run_dir(config: dict) -> Path:
    metadata = config.get("metadata", {})
    target_score = config.get("target_score", {})
    if not isinstance(metadata, dict) or not isinstance(target_score, dict):
        raise SystemExit("build config requires metadata and target_score objects")
    run_dir = metadata.get("run_dir") or target_score.get("run_dir")
    if run_dir:
        return expand_path(str(run_dir))
    output = target_score.get("search_targets_jsonl") or target_score.get("scores_csv")
    if output:
        return expand_path(str(output)).parent
    experiment = metadata.get("experiment", "target-score")
    return expand_path(DEFAULTS.run_base) / str(experiment)


def target_score_positions_path(target_score: dict, run_dir: Path) -> Path | None:
    positions = target_score.get("positions_jsonl")
    if positions:
        return expand_path(str(positions))
    if target_score.get("lichess_eval_input"):
        return run_dir / "positions.jsonl"
    return None


def target_score_bucket_specs(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value or "").split(";") if item.strip()]


def target_score_import_command(target_score: dict, run_dir: Path) -> list[str] | None:
    lichess_input = target_score.get("lichess_eval_input")
    if not lichess_input:
        return None
    if target_score.get("targets_csv"):
        raise SystemExit("target_score accepts targets_csv or lichess_eval_input, not both")
    output = target_score_positions_path(target_score, run_dir)
    if output is None:
        raise SystemExit("target_score lichess_eval_input requires an output positions path")
    command = [
        str(expand_user(target_score.get("python", DEFAULTS.python))),
        tool("score/import_lichess_eval.py"),
        "--input", str(expand_path(str(lichess_input))),
        "--output", str(output),
        "--rows", str(target_score.get("lichess_eval_rows", 0)),
        "--min-depth", str(target_score.get("lichess_eval_min_depth", 18)),
        "--min-knodes", str(target_score.get("lichess_eval_min_knodes", 0)),
        "--max-abs-cp", str(target_score.get(
            "lichess_eval_max_abs_cp", target_score.get("max_abs_cp", 1600))),
        "--max-input-rows", str(target_score.get("lichess_eval_max_input_rows", 0)),
        "--seed", str(target_score.get("lichess_eval_seed", target_score.get("seed", 1))),
        "--progress", str(target_score.get("lichess_eval_progress", 100000)),
        "--output-format", "jsonl",
    ]
    if target_score.get("lichess_eval_unique_fen", True):
        command.append("--unique-fen")
    if target_score.get("lichess_eval_stop_when_full", False):
        command.append("--stop-when-full")
    for bucket in target_score_bucket_specs(target_score.get("lichess_eval_buckets")):
        command.extend(["--bucket", bucket])
    return command


def target_build_command(
    target_build: dict,
    *,
    output_key: str,
    summary_key: str,
    run_dir: Path,
    dedupe: bool,
    event_command: str,
) -> list[str]:
    output = target_build.get(output_key)
    if not output:
        raise SystemExit(f"target_build.{output_key} is required")
    summary = target_build.get(summary_key) or str(Path(str(output)).with_suffix(".summary.txt"))
    sources = target_build.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise SystemExit("target_build.sources must be a non-empty object")
    command = [
        sys.executable,
        tool("validate/validate.py"),
        "search-targets",
        "--output", str(expand_path(str(output))),
        "--summary", str(expand_path(str(summary))),
        "--max-moves", str(target_build.get("max_moves", 0)),
        "--policy-temperature-cp", str(target_build.get("policy_temperature_cp", 200.0)),
        "--mate-gap-cp", str(target_build.get("mate_gap_cp", 30000)),
        "--run", str(run_dir),
    ]
    if event_command:
        command += ["--event-command", event_command]
    if dedupe:
        command.append("--dedupe-fen")
    for name, path in sources.items():
        command += ["--scores", f"{name}={expand_user(str(path))}"]
    return command


def target_score_command(target_score: dict, run_dir: Path) -> list[str]:
    output = target_score.get("scores_csv")
    if not output:
        raise SystemExit("target_score.scores_csv is required")
    command = [
        str(expand_user(target_score.get("python", DEFAULTS.python))),
        tool("validate/score_tail_targets.py"),
        "--out", str(expand_path(str(output))),
        "--engine", str(expand_user(target_score.get("engine", DEFAULTS.score_engine))),
        "--nodes", str(target_score.get("nodes", 200000)),
        "--hash", str(target_score.get("hash", 128)),
        "--threads", str(target_score.get("threads", 1)),
    ]
    targets = target_score.get("targets_csv")
    positions = target_score_positions_path(target_score, run_dir)
    if targets and positions:
        raise SystemExit("target_score accepts targets_csv or positions_jsonl/lichess_eval_input, not both")
    if targets:
        command.extend(["--targets", str(expand_path(str(targets)))])
    elif positions:
        command.extend(["--positions-jsonl", str(positions)])
    else:
        raise SystemExit("target_score.targets_csv, positions_jsonl, or lichess_eval_input is required")

    optional_args = {
        "syzygy_path": "--syzygy-path",
        "max_targets": "--max-targets",
        "max_abs_cp": "--max-abs-cp",
        "min_ply": "--min-ply",
        "min_material_count": "--min-material-count",
        "max_material_count": "--max-material-count",
        "scan_limit": "--scan-limit",
        "scan_progress": "--scan-progress",
        "score_progress": "--score-progress",
        "seed": "--seed",
    }
    for key, flag in optional_args.items():
        value = target_score.get(key)
        if value in (None, ""):
            continue
        if key == "syzygy_path":
            value = expand_path(str(value))
        command.extend([flag, str(value)])
    return command


def target_score_build_command(target_score: dict) -> list[str]:
    scores = target_score.get("scores_csv")
    output = target_score.get("search_targets_jsonl")
    if not scores or not output:
        raise SystemExit("target_score.scores_csv and search_targets_jsonl are required")
    summary = target_score.get("summary") or str(Path(str(output)).with_suffix(".summary.txt"))
    source_name = target_score.get("source_name", "broad")
    command = [
        str(expand_user(target_score.get("python", DEFAULTS.python))),
        tool("validate/build_search_targets.py"),
        "--scores", f"{source_name}={expand_path(str(scores))}",
        "--output", str(expand_path(str(output))),
        "--summary", str(expand_path(str(summary))),
        "--max-moves", str(target_score.get("max_moves", 0)),
        "--policy-temperature-cp", str(target_score.get("policy_temperature_cp", 200.0)),
        "--mate-gap-cp", str(target_score.get("mate_gap_cp", 30000)),
    ]
    if target_score.get("dedupe_fen", True):
        command.append("--dedupe-fen")
    return command


def cmd_target_build(args: argparse.Namespace) -> int:
    config_path = expand_path(args.config or default_build_config_path())
    config = json.loads(config_path.read_text(encoding="utf-8"))
    target_build = config.get("target_build")
    if not isinstance(target_build, dict):
        raise SystemExit(f"{config_path}: target_build object is required")
    run_dir = target_build_run_dir(config)
    event_command = args.event_command or str(config.get("event_command", ""))
    if not event_command:
        hooks = config.get("hooks", {})
        if isinstance(hooks, dict):
            event_command = str(hooks.get("event_command", ""))
    if args.dry_run:
        for output_key, summary_key, dedupe in [
            ("output_full", "summary_full", False),
            ("output_deduped", "summary_deduped", True),
        ]:
            print(" ".join(target_build_command(
                target_build,
                output_key=output_key,
                summary_key=summary_key,
                run_dir=run_dir,
                dedupe=dedupe,
                event_command=event_command,
            )))
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8")
    emit_event(
        run_dir, "phase_start", stage="target_build", status="running",
        message="building search-aware targets", hook_command=event_command,
    )
    rc = 0
    for output_key, summary_key, dedupe in [
        ("output_full", "summary_full", False),
        ("output_deduped", "summary_deduped", True),
    ]:
        command = target_build_command(
            target_build,
            output_key=output_key,
            summary_key=summary_key,
            run_dir=run_dir,
            dedupe=dedupe,
            event_command=event_command,
        )
        rc = run(command, echo=True)
        if rc != 0:
            emit_event(
                run_dir, "fail", stage="target_build", status="failed",
                rc=rc, command=command, hook_command=event_command,
            )
            return rc

    parts = []
    for label, output_key, summary_key in [
        ("full", "output_full", "summary_full"),
        ("deduped", "output_deduped", "summary_deduped"),
    ]:
        output = target_build.get(output_key)
        summary = target_build.get(summary_key) or str(Path(str(output)).with_suffix(".summary.txt"))
        summary_path = expand_path(str(summary))
        if summary_path.exists():
            parts.append(label)
            parts.append(summary_path.read_text(encoding="utf-8").strip())
            parts.append("")
    summary_path = run_dir / "summary.txt"
    summary_path.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
    emit_event(
        run_dir, "done", stage="target_build", status="ok", rc=0,
        log=summary_path, message="target build complete",
        hook_command=event_command,
    )
    print(summary_path.read_text(encoding="utf-8"), end="")
    return 0


def cmd_target_score(args: argparse.Namespace) -> int:
    config_path = expand_path(args.config or default_build_config_path())
    config = json.loads(config_path.read_text(encoding="utf-8"))
    target_score = config.get("target_score")
    if not isinstance(target_score, dict):
        raise SystemExit(f"{config_path}: target_score object is required")
    run_dir = target_score_run_dir(config)
    event_command = args.event_command or str(config.get("event_command", ""))
    if not event_command:
        hooks = config.get("hooks", {})
        if isinstance(hooks, dict):
            event_command = str(hooks.get("event_command", ""))
    import_command = target_score_import_command(target_score, run_dir)
    if args.dry_run:
        if import_command:
            print(" ".join(import_command))
        print(" ".join(target_score_command(target_score, run_dir)))
        print(" ".join(target_score_build_command(target_score)))
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8")
    stages = []
    if import_command:
        stages.append(("import_positions", import_command))
    stages.extend([
        ("score_legal_moves", target_score_command(target_score, run_dir)),
        ("build_search_targets", target_score_build_command(target_score)),
    ])
    for stage, command in stages:
        emit_event(
            run_dir, "phase_start", stage=stage, status="running",
            message=f"{stage} running", hook_command=event_command,
        )
        rc = run(command, echo=True)
        emit_event(
            run_dir, "phase_done" if rc == 0 else "fail",
            stage=stage, status="ok" if rc == 0 else "failed", rc=rc,
            command=command, hook_command=event_command,
        )
        if rc != 0:
            return rc

    summary = target_score.get("summary") or str(
        Path(str(target_score.get("search_targets_jsonl"))).with_suffix(".summary.txt"))
    summary_path = expand_path(str(summary))
    emit_event(
        run_dir, "done", stage="target_score", status="ok", rc=0,
        log=summary_path, message="target score complete",
        hook_command=event_command,
    )
    if summary_path.exists():
        print(summary_path.read_text(encoding="utf-8"), end="")
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
    parser.add_argument(
        "--labeled-jsonl",
        default=value("labeled_jsonl", d.labeled_jsonl),
        help="Existing labeled JSONL to pack/train from; skips posgen and score phases.",
    )
    parser.add_argument(
        "--backend",
        default=value("backend", d.backend),
        choices=["pytorch", "pairwise", "search-aware", "bullet", "material-head"],
        help=(
            "Training backend. 'material-head' trains bucketed float heads; "
            "'pairwise' fine-tunes Enyo .nn with child-position ranking pairs; "
            "'search-aware' blends broad scalar loss with multi-move "
            "search-target ranking/policy loss. 'bullet' is experimental; "
            "mode=reckless emits Bullet checkpoints, mode=enyo emits model.nn."),
    )
    parser.add_argument("--bucket-mode", default=value("bucket_mode", d.bucket_mode),
                        choices=["material", "king-pressure", "check-state"])

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
    parser.add_argument("--pack-skip", type=int, default=value("pack_skip", d.pack_skip))
    parser.add_argument("--pack-limit", type=int, default=value("pack_limit", d.pack_limit))
    parser.add_argument("--pack-progress", type=int, default=value("pack_progress", d.pack_progress))
    parser.add_argument("--blend-extra-jsonl", default=value("blend_extra_jsonl", d.blend_extra_jsonl))
    parser.add_argument("--blend-extra-repeat", type=int, default=value("blend_extra_repeat", d.blend_extra_repeat))
    parser.add_argument("--blend-base-limit", type=int, default=value("blend_base_limit", d.blend_base_limit))
    parser.add_argument("--blend-shuffle-seed", type=int, default=value("blend_shuffle_seed", d.blend_shuffle_seed))

    parser.add_argument(
        "--init-net", default=value("init_net", d.init_net),
        help="Initial .nn file. Use an empty string or null in build JSON to train from scratch.",
    )
    parser.add_argument("--init", default=value("init", d.init),
                        choices=["kaiming", "quantized-kaiming", "berserk-ish"])
    parser.add_argument("--objective", default=value("objective", d.objective),
                        choices=["mse", "huber", "mpe25"])
    parser.add_argument("--forward", default=value("forward", d.forward),
                        choices=["float", "quantized"])
    parser.add_argument("--target-clamp", type=int, default=value("target_clamp", d.target_clamp))
    parser.add_argument("--huber-beta", type=int, default=value("huber_beta", d.huber_beta))
    parser.add_argument("--wdl-lambda", type=float, default=value("wdl_lambda", d.wdl_lambda))
    parser.add_argument("--sign-loss-weight", type=float, default=value("sign_loss_weight", d.sign_loss_weight))
    parser.add_argument("--sign-loss-scale", type=float, default=value("sign_loss_scale", d.sign_loss_scale))
    parser.add_argument("--lr", type=float, default=value("lr", d.lr))
    parser.add_argument("--input-lr-mult", type=float, default=value("input_lr_mult", d.input_lr_mult))
    parser.add_argument("--l1-lr-mult", type=float, default=value("l1_lr_mult", d.l1_lr_mult))
    parser.add_argument("--dense-lr-mult", type=float, default=value("dense_lr_mult", d.dense_lr_mult))
    parser.add_argument("--epochs", type=int, default=value("epochs", d.epochs))
    parser.add_argument("--batch-size", type=int, default=value("batch_size", d.batch_size))
    parser.add_argument("--device", default=value("device", d.device))
    parser.add_argument("--workers", type=int, default=value("workers", d.workers))
    parser.add_argument("--val-rows", type=int, default=value("val_rows", d.val_rows))
    parser.add_argument("--max-rows", type=int, default=value("max_rows", d.max_rows))
    parser.add_argument("--skip-rows", type=int, default=value("skip_rows", d.skip_rows))
    parser.add_argument("--grad-norm-every", type=int, default=value("grad_norm_every", d.grad_norm_every))
    parser.add_argument("--patience", type=int, default=value("patience", d.patience))
    parser.add_argument("--select-metric", default=value("select_metric", d.select_metric),
                        choices=["loss", "mse", "mae", "sign"])
    parser.add_argument("--weight-decay", type=float, default=value("weight_decay", d.weight_decay))
    parser.add_argument("--trainable", default=value("trainable", d.trainable),
                        choices=["all", "input", "float-head", "output"])

    parser.add_argument("--pairwise-scores-csv", default=value("pairwise_scores_csv", d.pairwise_scores_csv))
    parser.add_argument("--pairwise-pairs-jsonl", default=value("pairwise_pairs_jsonl", d.pairwise_pairs_jsonl))
    parser.add_argument("--pairwise-candidate-moves-csv", default=value("pairwise_candidate_moves_csv", d.pairwise_candidate_moves_csv))
    parser.add_argument("--pairwise-pair-batch-size", type=int, default=value("pairwise_pair_batch_size", d.pairwise_pair_batch_size))
    parser.add_argument("--pairwise-broad-weight", type=float, default=value("pairwise_broad_weight", d.pairwise_broad_weight))
    parser.add_argument("--pairwise-pair-weight", type=float, default=value("pairwise_pair_weight", d.pairwise_pair_weight))
    parser.add_argument("--pairwise-pair-beta", type=float, default=value("pairwise_pair_beta", d.pairwise_pair_beta))
    parser.add_argument("--pairwise-min-target-margin", type=float, default=value("pairwise_min_target_margin", d.pairwise_min_target_margin))
    parser.add_argument("--pairwise-max-target-margin", type=float, default=value("pairwise_max_target_margin", d.pairwise_max_target_margin))
    parser.add_argument("--pairwise-loss-weight-by-cp", action="store_true", default=value("pairwise_loss_weight_by_cp", d.pairwise_loss_weight_by_cp))
    parser.add_argument("--pairwise-broad-target", default=value("pairwise_broad_target", d.pairwise_broad_target),
                        choices=["teacher", "init"])

    parser.add_argument("--search-targets-jsonl", default=value("search_targets_jsonl", d.search_targets_jsonl))
    parser.add_argument("--search-target-batch-size", type=int, default=value("search_target_batch_size", d.search_target_batch_size))
    parser.add_argument("--search-broad-weight", type=float, default=value("search_broad_weight", d.search_broad_weight))
    parser.add_argument("--search-broad-target", default=value("search_broad_target", d.search_broad_target),
                        choices=["teacher", "init"])
    parser.add_argument("--search-broad-target-net", default=value("search_broad_target_net", d.search_broad_target_net))
    parser.add_argument("--search-target-warmup-epochs", type=int, default=value("search_target_warmup_epochs", d.search_target_warmup_epochs))
    parser.add_argument("--search-warmup-broad-weight", type=float, default=value("search_warmup_broad_weight", d.search_warmup_broad_weight))
    parser.add_argument("--search-broad-ramp-epochs", type=int, default=value("search_broad_ramp_epochs", d.search_broad_ramp_epochs))
    parser.add_argument("--search-margin-weight", type=float, default=value("search_margin_weight", d.search_margin_weight))
    parser.add_argument("--search-policy-weight", type=float, default=value("search_policy_weight", d.search_policy_weight))
    parser.add_argument("--search-margin-beta", type=float, default=value("search_margin_beta", d.search_margin_beta))
    parser.add_argument("--search-policy-temperature-cp", type=float, default=value("search_policy_temperature_cp", d.search_policy_temperature_cp))
    parser.add_argument("--search-score-mode", default=value("search_score_mode", d.search_score_mode),
                        choices=["child-low", "root-high"])
    parser.add_argument("--search-max-gap-cp", type=float, default=value("search_max_gap_cp", d.search_max_gap_cp))
    parser.add_argument("--search-max-moves", type=int, default=value("search_max_moves", d.search_max_moves))
    parser.add_argument("--search-target-limit", type=int, default=value("search_target_limit", d.search_target_limit))
    parser.add_argument("--search-target-shuffle", action="store_true", default=value("search_target_shuffle", d.search_target_shuffle))
    parser.add_argument("--search-select-best-target", action="store_true", default=value("search_select_best_target", d.search_select_best_target))
    parser.add_argument("--search-required-tags", default=value("search_required_tags", d.search_required_tags))
    parser.add_argument("--search-tag-weights", default=value("search_tag_weights", d.search_tag_weights))
    parser.add_argument("--search-model-gate-device", default=value("search_model_gate_device", d.search_model_gate_device),
                        choices=["cpu", "cuda"])
    parser.add_argument("--search-model-gate-min-top1", type=int, default=value("search_model_gate_min_top1", d.search_model_gate_min_top1))

    parser.add_argument("--bullet-rows", type=int, default=value("bullet_rows", d.bullet_rows))
    parser.add_argument("--bullet-lichess-eval-input", default=value("bullet_lichess_eval_input", d.bullet_lichess_eval_input))
    parser.add_argument("--bullet-lichess-eval-buckets", default=value("bullet_lichess_eval_buckets", d.bullet_lichess_eval_buckets))
    parser.add_argument("--bullet-lichess-eval-min-depth", type=int, default=value("bullet_lichess_eval_min_depth", d.bullet_lichess_eval_min_depth))
    parser.add_argument("--bullet-lichess-eval-min-knodes", type=int, default=value("bullet_lichess_eval_min_knodes", d.bullet_lichess_eval_min_knodes))
    parser.add_argument("--bullet-lichess-eval-max-input-rows", type=int, default=value("bullet_lichess_eval_max_input_rows", d.bullet_lichess_eval_max_input_rows))
    parser.add_argument("--bullet-lichess-eval-unique-fen", action=argparse.BooleanOptionalAction,
                        default=value("bullet_lichess_eval_unique_fen", d.bullet_lichess_eval_unique_fen))
    parser.add_argument("--bullet-lichess-eval-stop-when-full", action=argparse.BooleanOptionalAction,
                        default=value("bullet_lichess_eval_stop_when_full", d.bullet_lichess_eval_stop_when_full))
    parser.add_argument("--bullet-lichess-eval-seed", type=int, default=value("bullet_lichess_eval_seed", d.bullet_lichess_eval_seed))
    parser.add_argument(
        "--bullet-data",
        default=value("bullet_data", d.bullet_data),
        help=(
            "Existing Bullet training data path. With --bullet-loader=direct this is a "
            "BulletFormat .data file; with --bullet-loader=sfbinpack this is one or more "
            "Stockfish binpack paths separated by ';'. Skips JSONL->Bullet conversion."
        ),
    )
    parser.add_argument("--bullet-loader", default=value("bullet_loader", d.bullet_loader),
                        choices=["direct", "sfbinpack"])
    parser.add_argument("--bullet-sfbinpack-buffer-mb", type=int,
                        default=value("bullet_sfbinpack_buffer_mb", d.bullet_sfbinpack_buffer_mb))
    parser.add_argument("--bullet-sfbinpack-min-ply", type=int,
                        default=value("bullet_sfbinpack_min_ply", d.bullet_sfbinpack_min_ply))
    parser.add_argument("--bullet-sfbinpack-max-abs-cp", type=int,
                        default=value("bullet_sfbinpack_max_abs_cp", d.bullet_sfbinpack_max_abs_cp))
    parser.add_argument("--bullet-sfbinpack-quiet-only", action=argparse.BooleanOptionalAction,
                        default=value("bullet_sfbinpack_quiet_only", d.bullet_sfbinpack_quiet_only))
    parser.add_argument("--bullet-mode", default=value("bullet_mode", d.bullet_mode),
                        choices=["reckless", "enyo"])
    parser.add_argument("--bullet-max-abs-cp", type=int, default=value("bullet_max_abs_cp", d.bullet_max_abs_cp))
    parser.add_argument("--bullet-manifest", default=value("bullet_manifest", d.bullet_manifest))
    parser.add_argument("--bullet-cuda-path", default=value("bullet_cuda_path", d.bullet_cuda_path))
    parser.add_argument("--bullet-cuda-arch", default=value("bullet_cuda_arch", d.bullet_cuda_arch))
    parser.add_argument("--bullet-cargo-target-dir", default=value("bullet_cargo_target_dir", d.bullet_cargo_target_dir))
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
    parser.add_argument("--bullet-enyo-input-buckets", type=int, choices=[16, 32],
                        default=value("bullet_enyo_input_buckets", d.bullet_enyo_input_buckets))
    parser.add_argument("--bullet-eval-scale", type=float, default=value("bullet_eval_scale", d.bullet_eval_scale))
    parser.add_argument("--bullet-save-rate", type=int, default=value("bullet_save_rate", d.bullet_save_rate))
    parser.add_argument("--bullet-export-init-only", action=argparse.BooleanOptionalAction,
                        default=value("bullet_export_init_only", d.bullet_export_init_only))


def build_parser(create_defaults: dict[str, object] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create and inspect Enyo NNUE candidate runs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Create/train a candidate.")
    add_create_args(create, create_defaults)
    create.set_defaults(func=cmd_create)

    status = subparsers.add_parser(
        "status",
        help="Show candidate run status. Defaults to build.json's run.",
    )
    status.add_argument("run", nargs="?")
    status.add_argument("--tail", type=int, default=0)
    status.set_defaults(func=cmd_status)

    report = subparsers.add_parser(
        "report",
        help="Print candidate run report. Defaults to build.json's run.",
    )
    report.add_argument("run", nargs="?")
    report.add_argument("--tail", type=int, default=20)
    report.set_defaults(func=cmd_report)

    target_build = subparsers.add_parser(
        "target-build",
        help="Build search-aware target files from build.json target_build.",
    )
    target_build.add_argument("-c", "--config", default=None)
    target_build.add_argument("--dry-run", action="store_true")
    target_build.add_argument(
        "--event-command",
        default=None,
        help="Optional event hook command.",
    )
    target_build.set_defaults(func=cmd_target_build)

    target_score = subparsers.add_parser(
        "target-score",
        help="Score legal moves and build search-aware targets from build.json target_score.",
    )
    target_score.add_argument("-c", "--config", default=None)
    target_score.add_argument("--dry-run", action="store_true")
    target_score.add_argument(
        "--event-command",
        default=None,
        help="Optional event hook command.",
    )
    target_score.set_defaults(func=cmd_target_score)

    return parser


def main() -> int:
    argv = normalize_argv(sys.argv)
    create_defaults = load_create_arg_defaults(create_config_path(argv))
    parser = build_parser(create_defaults)
    args = parser.parse_args(argv[1:])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
