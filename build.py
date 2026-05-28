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


def expand_path_list(raw: str) -> list[str]:
    return [
        str(expand_path(item.strip()))
        for item in raw.split(",")
        if item.strip()
    ]


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


def recorded_create_arg_keys(args: argparse.Namespace) -> set[str]:
    common = {
        "backend",
        "config",
        "event_command",
        "force",
        "name",
        "python",
        "run_dir",
    }
    data = {
        "pack_dir",
        "engine",
        "nnue_file",
        "book",
        "runner",
        "selfplay_games",
        "selfplay_shard_games",
        "selfplay_concurrency",
        "selfplay_threads",
        "selfplay_hash",
        "selfplay_depth",
        "selfplay_seed",
        "skip_plies",
        "source_max_abs_cp",
        "sample_preset",
        "score_engine",
        "score_depth",
        "score_shards",
        "score_threads",
        "score_hash",
        "score_max_abs_cp",
        "score_progress",
        "max_features",
        "pack_progress",
    }
    pytorch = data | {
        "init_net",
        "objective",
        "target_clamp",
        "huber_beta",
        "wdl_lambda",
        "lr",
        "epochs",
        "batch_size",
        "device",
        "workers",
        "prefetch_factor",
        "amp",
        "torch_compile",
        "dataset_in_memory",
        "val_rows",
        "patience",
        "select_metric",
        "weight_decay",
        "trainable",
    }
    child = data | {
        "init_net",
        "child_targets",
        "epochs",
        "batch_size",
        "child_batch_size",
        "child_loss",
        "lr",
        "weight_decay",
        "target_clamp",
        "ranking_weight",
        "broad_preserve_weight",
        "broad_anchor",
        "broad_deadzone_cp",
        "broad_beta",
        "rank_margin_cp",
        "rank_temperature_cp",
        "min_groups",
        "min_pairs",
        "device",
        "workers",
        "prefetch_factor",
        "amp",
        "torch_compile",
        "dataset_in_memory",
        "export_quantize_forward",
        "trainable",
        "child_broad_rows",
        "child_model_gate_min_top1",
        "child_engine_gate_min_top1",
    }
    policy = {
        "init_net",
        "policy_targets",
        "child_targets",
        "policy_hidden",
        "policy_feature_set",
        "policy_dropout",
        "policy_include_tags",
        "policy_exclude_tags",
        "policy_preserve_include_tags",
        "policy_preserve_exclude_tags",
        "policy_preserve_weight",
        "policy_preserve_margin",
        "policy_preserve_max_groups",
        "policy_preserve_val_fraction",
        "policy_base_best_preserve_weight",
        "policy_no_harm_weight",
        "policy_no_harm_gap_cp",
        "epochs",
        "lr",
        "weight_decay",
        "rank_temperature_cp",
        "policy_target_temperature_cp",
        "policy_val_fraction",
        "selfplay_seed",
        "device",
        "policy_gate_include_tags",
        "policy_gate_exclude_tags",
        "policy_broad_gate_include_tags",
        "policy_broad_gate_exclude_tags",
        "policy_broad_gate_min_groups",
        "policy_broad_gate_max_bad",
        "policy_broad_gate_max_overrides",
        "policy_breakdown_tags",
        "policy_thresholds",
        "policy_gate_min_top1",
        "policy_gate_max_bad",
        "policy_gate_min_val_top1",
        "policy_gate_max_val_bad",
        "policy_gate_min_val_good",
        "policy_gate_min_val_overrides",
        "policy_bad_tolerance_cp",
        "policy_export_threshold",
        "policy_export_max_abs_diff",
        "min_groups",
    }
    replay = {
        "replay",
        "score_engine",
        "replay_logs",
        "replay_candidate",
        "replay_reference",
        "replay_oracle_nodes",
        "replay_jobs",
        "replay_move",
        "replay_top_root_moves",
        "replay_include_checks",
        "replay_include_captures",
        "replay_include_promotions",
        "replay_include_history_sensitive",
        "replay_max_moves_per_position",
        "replay_min_score_gap",
        "replay_output",
        "replay_stderr",
        "replay_min_rows",
        "replay_child_targets",
        "replay_child_summary",
        "replay_child_min_groups",
    }
    lc0 = {
        "lc0_input",
        "lc0_output",
        "lc0_summary",
        "lc0_max_records",
        "lc0_top_policy",
        "lc0_min_rows",
        "lc0_min_played_legal_pct",
        "lc0_min_best_legal_pct",
        "lc0_child_targets",
        "lc0_child_summary",
        "lc0_child_min_groups",
        "lc0_child_max_groups",
        "lc0_child_unique_fen",
        "lc0_best_source",
        "lc0_policy_score_scale_cp",
        "lc0_policy_floor",
        "lc0_child_max_gap_cp",
        "lc0_min_best_policy",
        "lc0_min_policy_gap_cp",
    }
    backend_keys = {
        "pytorch": pytorch,
        "child-ranking": child,
        "policy-ranking": policy,
        "replay-jsonl": replay,
        "lc0-jsonl": lc0,
    }
    return common | backend_keys.get(args.backend, set())


def create_config(args: argparse.Namespace) -> dict:
    name = args.name or default_name()
    run_dir = run_dir_for(name, args.run_dir)
    candidate_dir = f"{{train}}/{name}"
    python = str(expand_user(args.python))
    steps = []
    data_dir = ""
    if args.backend in {"pytorch", "child-ranking"} and args.pack_dir:
        data_dir = str(expand_path(args.pack_dir))
    elif args.backend in {"pytorch", "child-ranking"}:
        data_dir = "{pack}/train"
        steps.extend([
            {
                "name": "posgen_selfplay",
                "command": [
                    python, tool("posgen/posgen.py"), "selfplay",
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
                    python, tool("posgen/posgen.py"), "extract",
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
                    python, tool("posgen/posgen.py"), "sample",
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
                    python, tool("score/score.py"), "uci",
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
                    python, tool("pack/pack.py"), "build",
                    "--input", "{score}/labeled.jsonl",
                    "--out-dir", "{pack}/train",
                    "--max-features", str(args.max_features),
                    "--progress", str(args.pack_progress),
                    "--python", python,
                ],
            },
        ])

    if args.backend == "pytorch":
        steps.append({
            "name": "train",
            "command": [
                python, tool("train/train.py"), "run",
                "--data", data_dir,
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
        })
    elif args.backend == "child-ranking":
        if not args.child_targets:
            raise SystemExit("backend=child-ranking requires child_targets")
        steps.extend([
            {
                "name": "train_child_ranking",
                "command": [
                    python, tool("train/train_child_ranking.py"),
                    "--data", data_dir,
                    "--child-targets", str(expand_path(args.child_targets)),
                    "--init-from-nn", str(expand_path(args.init_net)),
                    "--epochs", str(args.epochs),
                    "--batch-size", str(args.batch_size),
                    "--child-batch-size", str(args.child_batch_size),
                    "--child-loss", args.child_loss,
                    "--lr", str(args.lr),
                    "--weight-decay", str(args.weight_decay),
                    "--target-clamp", str(args.target_clamp),
                    "--ranking-weight", str(args.ranking_weight),
                    "--broad-preserve-weight", str(args.broad_preserve_weight),
                    "--broad-anchor", args.broad_anchor,
                    "--broad-deadzone-cp", str(args.broad_deadzone_cp),
                    "--broad-beta", str(args.broad_beta),
                    "--rank-margin-cp", str(args.rank_margin_cp),
                    "--rank-temperature-cp", str(args.rank_temperature_cp),
                    "--min-groups", str(args.min_groups),
                    "--min-pairs", str(args.min_pairs),
                    "--device", args.device,
                    "--workers", str(args.workers),
                    "--prefetch-factor", str(args.prefetch_factor),
                    "--amp", args.amp,
                    "--torch-compile" if args.torch_compile else "--no-torch-compile",
                    "--dataset-in-memory" if args.dataset_in_memory else "--no-dataset-in-memory",
                    "--export-quantize-forward" if args.export_quantize_forward else "--no-export-quantize-forward",
                    "--trainable", args.trainable,
                    "--max-rows", str(args.child_broad_rows),
                    "--out", f"{candidate_dir}/model.pt",
                    "--out-nn", f"{candidate_dir}/model.nn",
                ],
            },
            {
                "name": "validate_child_ranking_model",
                "command": [
                    python, tool("validate/child_rank_model_gate.py"),
                    "--targets", str(expand_path(args.child_targets)),
                    "--net", f"{candidate_dir}/model.nn",
                    "--pt", f"{candidate_dir}/model.pt",
                    "--pt-export-quantize-forward" if args.export_quantize_forward else "--no-pt-export-quantize-forward",
                    "--device", "cpu",
                    "--min-groups", str(args.min_groups),
                    "--fail-if-net-top1-below", str(args.child_model_gate_min_top1),
                    "--fail-if-pt-top1-below", str(args.child_model_gate_min_top1),
                ],
            },
            {
                "name": "validate_child_ranking_engine",
                "command": [
                    python, tool("validate/child_rank_engine_gate.py"),
                    "--targets", str(expand_path(args.child_targets)),
                    "--engine", str(expand_path(args.engine)),
                    "--net", f"{candidate_dir}/model.nn",
                    "--threads", "1",
                    "--hash", "64",
                    "--jobs", str(args.child_engine_jobs),
                    "--min-groups", str(args.min_groups),
                    "--fail-if-top1-below", str(args.child_engine_gate_min_top1),
                ],
            },
        ])
    elif args.backend == "policy-ranking":
        policy_targets = args.policy_targets or args.child_targets
        if not policy_targets:
            raise SystemExit("backend=policy-ranking requires policy_targets")
        policy_target_paths = expand_path_list(str(policy_targets))
        policy_gate_include_tags = (
            args.policy_gate_include_tags or args.policy_include_tags)
        policy_gate_exclude_tags = (
            args.policy_gate_exclude_tags or args.policy_exclude_tags)
        steps.extend([
            {
                "name": "train_policy_ranker",
                "command": [
                    python, tool("train/train_policy_ranker.py"),
                    "--targets", *policy_target_paths,
                    "--base-net", str(expand_path(args.init_net)),
                    "--out", f"{candidate_dir}/model.pt",
                    "--hidden", str(args.policy_hidden),
                    "--feature-set", args.policy_feature_set,
                    "--dropout", str(args.policy_dropout),
                    "--include-tags", args.policy_include_tags,
                    "--exclude-tags", args.policy_exclude_tags,
                    "--preserve-include-tags", args.policy_preserve_include_tags,
                    "--preserve-exclude-tags", args.policy_preserve_exclude_tags,
                    "--preserve-weight", str(args.policy_preserve_weight),
                    "--preserve-margin", str(args.policy_preserve_margin),
                    "--preserve-max-groups", str(args.policy_preserve_max_groups),
                    "--preserve-val-fraction", str(args.policy_preserve_val_fraction),
                    "--base-best-preserve-weight",
                    str(args.policy_base_best_preserve_weight),
                    "--no-harm-weight", str(args.policy_no_harm_weight),
                    "--no-harm-gap-cp", str(args.policy_no_harm_gap_cp),
                    "--epochs", str(args.epochs),
                    "--lr", str(args.lr),
                    "--weight-decay", str(args.weight_decay),
                    "--rank-temperature-cp", str(args.rank_temperature_cp),
                    "--target-temperature-cp", str(args.policy_target_temperature_cp),
                    "--val-fraction", str(args.policy_val_fraction),
                    "--seed", str(args.selfplay_seed),
                    "--device", args.device,
                ],
            },
            {
                "name": "validate_policy_ranker",
                "command": [
                    python, tool("validate/policy_ranker_gate.py"),
                    "--targets", *policy_target_paths,
                    "--model", f"{candidate_dir}/model.pt",
                    "--base-net", str(expand_path(args.init_net)),
                    "--device", args.device,
                    "--feature-set", args.policy_feature_set,
                    "--include-tags", policy_gate_include_tags,
                    "--exclude-tags", policy_gate_exclude_tags,
                    "--breakdown-tags", args.policy_breakdown_tags,
                    "--min-groups", str(args.min_groups),
                    "--thresholds", args.policy_thresholds,
                    "--split-seed", str(args.selfplay_seed),
                    "--split-val-fraction", str(args.policy_val_fraction),
                    "--fail-if-top1-below", str(args.policy_gate_min_top1),
                    "--fail-if-bad-above", str(args.policy_gate_max_bad),
                    "--fail-if-val-top1-below", str(args.policy_gate_min_val_top1),
                    "--fail-if-val-bad-above", str(args.policy_gate_max_val_bad),
                    "--fail-if-val-good-below", str(args.policy_gate_min_val_good),
                    "--fail-if-val-overrides-below", str(args.policy_gate_min_val_overrides),
                    "--bad-tolerance-cp", str(args.policy_bad_tolerance_cp),
                ],
            },
        ])
        steps.append({
            "name": "validate_policy_ranker_deploy",
            "command": [
                python, tool("validate/policy_ranker_gate.py"),
                "--targets", *policy_target_paths,
                "--model", f"{candidate_dir}/model.pt",
                "--base-net", str(expand_path(args.init_net)),
                "--device", args.device,
                "--feature-set", args.policy_feature_set,
                "--include-tags", policy_gate_include_tags,
                "--exclude-tags", policy_gate_exclude_tags,
                "--breakdown-tags", args.policy_breakdown_tags,
                "--min-groups", str(args.min_groups),
                "--thresholds", str(args.policy_export_threshold),
                "--split-seed", str(args.selfplay_seed),
                "--split-val-fraction", str(args.policy_val_fraction),
                "--fail-if-top1-below", str(args.policy_gate_min_top1),
                "--fail-if-bad-above", str(args.policy_gate_max_bad),
                "--fail-if-val-top1-below", str(args.policy_gate_min_val_top1),
                "--fail-if-val-bad-above", str(args.policy_gate_max_val_bad),
                "--fail-if-val-good-below", str(args.policy_gate_min_val_good),
                "--fail-if-val-overrides-below",
                str(args.policy_gate_min_val_overrides),
                "--bad-tolerance-cp", str(args.policy_bad_tolerance_cp),
            ],
        })
        if (args.policy_broad_gate_include_tags
                or args.policy_broad_gate_exclude_tags):
            steps.append({
                "name": "validate_policy_ranker_broad",
                "command": [
                    python, tool("validate/policy_ranker_gate.py"),
                    "--targets", *policy_target_paths,
                    "--model", f"{candidate_dir}/model.pt",
                    "--base-net", str(expand_path(args.init_net)),
                    "--device", args.device,
                    "--feature-set", args.policy_feature_set,
                    "--include-tags", args.policy_broad_gate_include_tags,
                    "--exclude-tags", args.policy_broad_gate_exclude_tags,
                    "--breakdown-tags", args.policy_breakdown_tags,
                    "--min-groups", str(args.policy_broad_gate_min_groups),
                    "--thresholds", str(args.policy_export_threshold),
                    "--split-seed", str(args.selfplay_seed),
                    "--split-val-fraction", "0",
                    "--fail-if-bad-above", str(args.policy_broad_gate_max_bad),
                    "--fail-if-overrides-above",
                    str(args.policy_broad_gate_max_overrides),
                    "--bad-tolerance-cp", str(args.policy_bad_tolerance_cp),
                ],
            })
        steps.extend([
            {
                "name": "export_policy_ranker",
                "command": [
                    python, tool("train/export_policy_ranker.py"),
                    "--model", f"{candidate_dir}/model.pt",
                    "--out", f"{candidate_dir}/policy_ranker.json",
                    "--threshold", str(args.policy_export_threshold),
                ],
            },
            {
                "name": "validate_policy_ranker_export",
                "command": [
                    python, tool("validate/policy_ranker_export_gate.py"),
                    "--targets", *policy_target_paths,
                    "--model", f"{candidate_dir}/model.pt",
                    "--export", f"{candidate_dir}/policy_ranker.json",
                    "--base-net", str(expand_path(args.init_net)),
                    "--device", args.device,
                    "--feature-set", args.policy_feature_set,
                    "--include-tags", policy_gate_include_tags,
                    "--exclude-tags", policy_gate_exclude_tags,
                    "--min-groups", str(args.min_groups),
                    "--max-abs-diff", str(args.policy_export_max_abs_diff),
                ],
            },
        ])
    elif args.backend == "replay-jsonl":
        if args.replay_reference:
            replay_script = r"""
set -euo pipefail
logs=$1
replay_bin=$2
candidate=$3
reference=$4
out=$5
err=$6
oracle_nodes=$7
jobs=$8
move_no=$9
top_root_moves=${{10}}
include_checks=${{11}}
include_captures=${{12}}
include_promotions=${{13}}
max_moves=${{14}}
min_score_gap=${{15}}
include_history_sensitive=${{16}}
oracle=${{17}}

mkdir -p "$(dirname "$out")"
cmd=("$replay_bin" --jsonl --candidate "$candidate")
cmd+=(--reference "$reference")
cmd+=(--oracle "$oracle")
cmd+=(--oracle-nodes "$oracle_nodes" --jobs "$jobs" --move "$move_no")
cmd+=(--top-root-moves "$top_root_moves")
if [[ "$include_checks" == "1" ]]; then
  cmd+=(--include-checks)
else
  cmd+=(--no-checks)
fi
if [[ "$include_captures" == "1" ]]; then
  cmd+=(--include-captures)
else
  cmd+=(--no-captures)
fi
if [[ "$include_promotions" == "1" ]]; then
  cmd+=(--include-promotions)
else
  cmd+=(--no-promotions)
fi
if [[ "$include_history_sensitive" == "1" ]]; then
  cmd+=(--include-history-sensitive)
fi
cmd+=(--max-moves-per-position "$max_moves" --min-score-gap "$min_score_gap" -)

find "$logs" -name '*.log' ! -iname '*conflicted*copy*.log' |
  sort |
  while IFS= read -r log_file; do
    if grep -Eq '(^|[[:space:]])go([[:space:]]|$)' "$log_file" &&
       grep -Eq '(^|[[:space:]])bestmove([[:space:]]|$)' "$log_file"; then
      printf '%s\n' "$log_file"
    else
      printf 'skip unreplayable log: %s\n' "$log_file" >&2
    fi
  done | "${{cmd[@]}}" > "$out" 2> "$err"
test -s "$out"
wc -l "$out" > "$out.wc"
"""
            replay_command = [
                "bash", "-lc", replay_script, "extract-replay-jsonl",
                str(expand_user(args.replay_logs)),
                str(expand_user(args.replay)),
                str(expand_user(args.replay_candidate)),
                str(expand_user(args.replay_reference)),
                str(expand_user(args.replay_output)),
                str(expand_user(args.replay_stderr)),
                str(args.replay_oracle_nodes),
                str(args.replay_jobs),
                str(args.replay_move),
                str(args.replay_top_root_moves),
                "1" if args.replay_include_checks else "0",
                "1" if args.replay_include_captures else "0",
                "1" if args.replay_include_promotions else "0",
                str(args.replay_max_moves_per_position),
                str(args.replay_min_score_gap),
                "1" if args.replay_include_history_sensitive else "0",
                str(expand_user(args.score_engine)),
            ]
        else:
            replay_script = r"""
set -euo pipefail
logs=$1
replay_bin=$2
candidate=$3
out=$4
err=$5
oracle_nodes=$6
jobs=$7
move_no=$8
top_root_moves=$9
include_checks=${{10}}
include_captures=${{11}}
include_promotions=${{12}}
max_moves=${{13}}
min_score_gap=${{14}}
include_history_sensitive=${{15}}
oracle=${{16}}

mkdir -p "$(dirname "$out")"
cmd=("$replay_bin" --jsonl --candidate "$candidate")
cmd+=(--oracle "$oracle")
cmd+=(--oracle-nodes "$oracle_nodes" --jobs "$jobs" --move "$move_no")
cmd+=(--top-root-moves "$top_root_moves")
if [[ "$include_checks" == "1" ]]; then
  cmd+=(--include-checks)
else
  cmd+=(--no-checks)
fi
if [[ "$include_captures" == "1" ]]; then
  cmd+=(--include-captures)
else
  cmd+=(--no-captures)
fi
if [[ "$include_promotions" == "1" ]]; then
  cmd+=(--include-promotions)
else
  cmd+=(--no-promotions)
fi
if [[ "$include_history_sensitive" == "1" ]]; then
  cmd+=(--include-history-sensitive)
fi
cmd+=(--max-moves-per-position "$max_moves" --min-score-gap "$min_score_gap" -)

find "$logs" -name '*.log' ! -iname '*conflicted*copy*.log' |
  sort |
  while IFS= read -r log_file; do
    if grep -Eq '(^|[[:space:]])go([[:space:]]|$)' "$log_file" &&
       grep -Eq '(^|[[:space:]])bestmove([[:space:]]|$)' "$log_file"; then
      printf '%s\n' "$log_file"
    else
      printf 'skip unreplayable log: %s\n' "$log_file" >&2
    fi
  done | "${{cmd[@]}}" > "$out" 2> "$err"
test -s "$out"
wc -l "$out" > "$out.wc"
"""
            replay_command = [
                "bash", "-lc", replay_script, "extract-replay-jsonl",
                str(expand_user(args.replay_logs)),
                str(expand_user(args.replay)),
                str(expand_user(args.replay_candidate)),
                str(expand_user(args.replay_output)),
                str(expand_user(args.replay_stderr)),
                str(args.replay_oracle_nodes),
                str(args.replay_jobs),
                str(args.replay_move),
                str(args.replay_top_root_moves),
                "1" if args.replay_include_checks else "0",
                "1" if args.replay_include_captures else "0",
                "1" if args.replay_include_promotions else "0",
                str(args.replay_max_moves_per_position),
                str(args.replay_min_score_gap),
                "1" if args.replay_include_history_sensitive else "0",
                str(expand_user(args.score_engine)),
            ]
        steps.extend([
            {
                "name": "extract_replay_jsonl",
                "command": replay_command,
            },
            {
                "name": "validate_replay_jsonl",
                "command": [
                    python, tool("validate/validate_replay_jsonl.py"),
                    "--input", str(expand_user(args.replay_output)),
                    "--min-rows", str(args.replay_min_rows),
                    "--fail-if-dirty-candidate",
                    *([] if args.replay_include_history_sensitive
                      else ["--fail-if-history-sensitive"]),
                ],
            },
            {
                "name": "convert_replay_child_targets",
                "command": [
                    python, tool("validate/replay_jsonl_to_child_targets.py"),
                    "--input", str(expand_user(args.replay_output)),
                    "--out", str(expand_user(args.replay_child_targets)),
                    "--summary", str(expand_user(args.replay_child_summary)),
                    "--min-groups", str(args.replay_child_min_groups),
                ],
            },
        ])
    elif args.backend == "lc0-jsonl":
        steps.extend([
            {
                "name": "extract_lc0_jsonl",
                "command": [
                    python, tool("posgen/lc0_to_jsonl.py"),
                    "--input", str(expand_user(args.lc0_input)),
                    "--out", str(expand_user(args.lc0_output)),
                    "--summary", str(expand_user(args.lc0_summary)),
                    "--max-records", str(args.lc0_max_records),
                    "--top-policy", str(args.lc0_top_policy),
                    "--min-rows", str(args.lc0_min_rows),
                    "--min-played-legal-pct", str(args.lc0_min_played_legal_pct),
                    "--min-best-legal-pct", str(args.lc0_min_best_legal_pct),
                ],
            },
            {
                "name": "convert_lc0_child_targets",
                "command": [
                    python, tool("validate/lc0_jsonl_to_child_targets.py"),
                    "--input", str(expand_user(args.lc0_output)),
                    "--out", str(expand_user(args.lc0_child_targets)),
                    "--summary", str(expand_user(args.lc0_child_summary)),
                    "--min-groups", str(args.lc0_child_min_groups),
                    "--max-groups", str(args.lc0_child_max_groups),
                    "--unique-fen" if args.lc0_child_unique_fen else "--no-unique-fen",
                    "--best-source", args.lc0_best_source,
                    "--policy-score-scale-cp", str(args.lc0_policy_score_scale_cp),
                    "--policy-floor", str(args.lc0_policy_floor),
                    "--max-gap-cp", str(args.lc0_child_max_gap_cp),
                    "--min-best-policy", str(args.lc0_min_best_policy),
                    "--min-policy-gap-cp", str(args.lc0_min_policy_gap_cp),
                ],
            },
        ])
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
            if key in recorded_create_arg_keys(args)
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
    command = [str(expand_user(args.python)), tool("pipeline/pipeline.py"), "launch", str(config_path)]
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
    for model in sorted(run_dir.glob("train/*/model.pt")):
        print(f"model={model}")
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
    parser.add_argument(
        "--pack-dir",
        default=value("pack_dir", d.pack_dir),
        help="Reuse an existing packed dataset and skip self-play, scoring, and packing.",
    )

    parser.add_argument("--init-net", default=value("init_net", d.init_net))
    parser.add_argument("--backend", default=value("backend", d.backend),
                        choices=["pytorch", "child-ranking", "policy-ranking",
                                 "replay-jsonl", "lc0-jsonl"])
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
    parser.add_argument("--export-quantize-forward", action=argparse.BooleanOptionalAction,
                        default=value("export_quantize_forward", d.export_quantize_forward))
    parser.add_argument("--val-rows", type=int, default=value("val_rows", d.val_rows))
    parser.add_argument("--patience", type=int, default=value("patience", d.patience))
    parser.add_argument("--select-metric", default=value("select_metric", d.select_metric),
                        choices=["loss", "mse", "mae", "sign"])
    parser.add_argument("--weight-decay", type=float, default=value("weight_decay", d.weight_decay))
    parser.add_argument("--trainable", default=value("trainable", d.trainable),
                        choices=["all", "float-head", "output"])
    parser.add_argument("--child-targets", default=value("child_targets", d.child_targets))
    parser.add_argument("--child-broad-rows", type=int, default=value("child_broad_rows", d.child_broad_rows))
    parser.add_argument("--child-batch-size", type=int, default=value("child_batch_size", d.child_batch_size))
    parser.add_argument("--child-loss", default=value("child_loss", d.child_loss),
                        choices=["pairwise", "listwise"])
    parser.add_argument("--ranking-weight", type=float, default=value("ranking_weight", d.ranking_weight))
    parser.add_argument("--broad-preserve-weight", type=float, default=value("broad_preserve_weight", d.broad_preserve_weight))
    parser.add_argument("--broad-anchor", default=value("broad_anchor", d.broad_anchor),
                        choices=["label", "reference"])
    parser.add_argument("--broad-deadzone-cp", type=int, default=value("broad_deadzone_cp", d.broad_deadzone_cp))
    parser.add_argument("--broad-beta", type=int, default=value("broad_beta", d.broad_beta))
    parser.add_argument("--rank-margin-cp", type=int, default=value("rank_margin_cp", d.rank_margin_cp))
    parser.add_argument("--rank-temperature-cp", type=int, default=value("rank_temperature_cp", d.rank_temperature_cp))
    parser.add_argument("--min-groups", type=int, default=value("min_groups", d.min_groups))
    parser.add_argument("--min-pairs", type=int, default=value("min_pairs", d.min_pairs))
    parser.add_argument("--child-model-gate-min-top1", type=int, default=value("child_model_gate_min_top1", d.child_model_gate_min_top1))
    parser.add_argument("--child-engine-gate-min-top1", type=int, default=value("child_engine_gate_min_top1", d.child_engine_gate_min_top1))
    parser.add_argument("--child-engine-jobs", type=int, default=value("child_engine_jobs", d.child_engine_jobs))
    parser.add_argument("--policy-targets", default=value("policy_targets", d.policy_targets))
    parser.add_argument("--policy-hidden", type=int, default=value("policy_hidden", d.policy_hidden))
    parser.add_argument("--policy-feature-set", default=value("policy_feature_set", d.policy_feature_set),
                        choices=["compact", "board"])
    parser.add_argument("--policy-dropout", type=float, default=value("policy_dropout", d.policy_dropout))
    parser.add_argument("--policy-include-tags", default=value("policy_include_tags", d.policy_include_tags))
    parser.add_argument("--policy-exclude-tags", default=value("policy_exclude_tags", d.policy_exclude_tags))
    parser.add_argument("--policy-preserve-include-tags", default=value("policy_preserve_include_tags", d.policy_preserve_include_tags))
    parser.add_argument("--policy-preserve-exclude-tags", default=value("policy_preserve_exclude_tags", d.policy_preserve_exclude_tags))
    parser.add_argument("--policy-preserve-weight", type=float, default=value("policy_preserve_weight", d.policy_preserve_weight))
    parser.add_argument("--policy-preserve-margin", type=float, default=value("policy_preserve_margin", d.policy_preserve_margin))
    parser.add_argument("--policy-preserve-max-groups", type=int, default=value("policy_preserve_max_groups", d.policy_preserve_max_groups))
    parser.add_argument("--policy-preserve-val-fraction", type=float, default=value("policy_preserve_val_fraction", d.policy_preserve_val_fraction))
    parser.add_argument("--policy-base-best-preserve-weight", type=float, default=value("policy_base_best_preserve_weight", d.policy_base_best_preserve_weight))
    parser.add_argument("--policy-no-harm-weight", type=float, default=value("policy_no_harm_weight", d.policy_no_harm_weight))
    parser.add_argument("--policy-no-harm-gap-cp", type=float, default=value("policy_no_harm_gap_cp", d.policy_no_harm_gap_cp))
    parser.add_argument("--policy-gate-include-tags", default=value("policy_gate_include_tags", d.policy_gate_include_tags))
    parser.add_argument("--policy-gate-exclude-tags", default=value("policy_gate_exclude_tags", d.policy_gate_exclude_tags))
    parser.add_argument("--policy-broad-gate-include-tags", default=value("policy_broad_gate_include_tags", d.policy_broad_gate_include_tags))
    parser.add_argument("--policy-broad-gate-exclude-tags", default=value("policy_broad_gate_exclude_tags", d.policy_broad_gate_exclude_tags))
    parser.add_argument("--policy-broad-gate-min-groups", type=int, default=value("policy_broad_gate_min_groups", d.policy_broad_gate_min_groups))
    parser.add_argument("--policy-broad-gate-max-bad", type=int, default=value("policy_broad_gate_max_bad", d.policy_broad_gate_max_bad))
    parser.add_argument("--policy-broad-gate-max-overrides", type=int, default=value("policy_broad_gate_max_overrides", d.policy_broad_gate_max_overrides))
    parser.add_argument("--policy-breakdown-tags", default=value("policy_breakdown_tags", d.policy_breakdown_tags))
    parser.add_argument("--policy-val-fraction", type=float, default=value("policy_val_fraction", d.policy_val_fraction))
    parser.add_argument("--policy-target-temperature-cp", type=int, default=value("policy_target_temperature_cp", d.policy_target_temperature_cp))
    parser.add_argument("--policy-thresholds", default=value("policy_thresholds", d.policy_thresholds))
    parser.add_argument("--policy-gate-min-top1", type=int, default=value("policy_gate_min_top1", d.policy_gate_min_top1))
    parser.add_argument("--policy-gate-max-bad", type=int, default=value("policy_gate_max_bad", d.policy_gate_max_bad))
    parser.add_argument("--policy-gate-min-val-top1", type=int, default=value("policy_gate_min_val_top1", d.policy_gate_min_val_top1))
    parser.add_argument("--policy-gate-max-val-bad", type=int, default=value("policy_gate_max_val_bad", d.policy_gate_max_val_bad))
    parser.add_argument("--policy-gate-min-val-good", type=int, default=value("policy_gate_min_val_good", d.policy_gate_min_val_good))
    parser.add_argument("--policy-gate-min-val-overrides", type=int, default=value("policy_gate_min_val_overrides", d.policy_gate_min_val_overrides))
    parser.add_argument("--policy-bad-tolerance-cp", type=float, default=value("policy_bad_tolerance_cp", d.policy_bad_tolerance_cp))
    parser.add_argument("--policy-export-threshold", type=float, default=value("policy_export_threshold", d.policy_export_threshold))
    parser.add_argument("--policy-export-max-abs-diff", type=float, default=value("policy_export_max_abs_diff", d.policy_export_max_abs_diff))
    parser.add_argument("--replay", default=value("replay", d.replay))
    parser.add_argument("--replay-logs", default=value("replay_logs", d.replay_logs))
    parser.add_argument("--replay-candidate", default=value("replay_candidate", d.replay_candidate))
    parser.add_argument("--replay-reference", default=value("replay_reference", d.replay_reference))
    parser.add_argument("--replay-oracle-nodes", type=int, default=value("replay_oracle_nodes", d.replay_oracle_nodes))
    parser.add_argument("--replay-jobs", type=int, default=value("replay_jobs", d.replay_jobs))
    parser.add_argument("--replay-move", type=int, default=value("replay_move", d.replay_move))
    parser.add_argument("--replay-top-root-moves", type=int, default=value("replay_top_root_moves", d.replay_top_root_moves))
    parser.add_argument("--replay-include-checks", action=argparse.BooleanOptionalAction,
                        default=value("replay_include_checks", d.replay_include_checks))
    parser.add_argument("--replay-include-captures", action=argparse.BooleanOptionalAction,
                        default=value("replay_include_captures", d.replay_include_captures))
    parser.add_argument("--replay-include-promotions", action=argparse.BooleanOptionalAction,
                        default=value("replay_include_promotions", d.replay_include_promotions))
    parser.add_argument("--replay-include-history-sensitive", action=argparse.BooleanOptionalAction,
                        default=value("replay_include_history_sensitive", d.replay_include_history_sensitive))
    parser.add_argument("--replay-max-moves-per-position", type=int, default=value("replay_max_moves_per_position", d.replay_max_moves_per_position))
    parser.add_argument("--replay-min-score-gap", type=int, default=value("replay_min_score_gap", d.replay_min_score_gap))
    parser.add_argument("--replay-output", default=value("replay_output", d.replay_output))
    parser.add_argument("--replay-stderr", default=value("replay_stderr", d.replay_stderr))
    parser.add_argument("--replay-min-rows", type=int, default=value("replay_min_rows", d.replay_min_rows))
    parser.add_argument("--replay-child-targets", default=value("replay_child_targets", d.replay_child_targets))
    parser.add_argument("--replay-child-summary", default=value("replay_child_summary", d.replay_child_summary))
    parser.add_argument("--replay-child-min-groups", type=int, default=value("replay_child_min_groups", d.replay_child_min_groups))
    parser.add_argument("--lc0-input", default=value("lc0_input", d.lc0_input))
    parser.add_argument("--lc0-output", default=value("lc0_output", d.lc0_output))
    parser.add_argument("--lc0-summary", default=value("lc0_summary", d.lc0_summary))
    parser.add_argument("--lc0-max-records", type=int, default=value("lc0_max_records", d.lc0_max_records))
    parser.add_argument("--lc0-top-policy", type=int, default=value("lc0_top_policy", d.lc0_top_policy))
    parser.add_argument("--lc0-min-rows", type=int, default=value("lc0_min_rows", d.lc0_min_rows))
    parser.add_argument("--lc0-min-played-legal-pct", type=float, default=value("lc0_min_played_legal_pct", d.lc0_min_played_legal_pct))
    parser.add_argument("--lc0-min-best-legal-pct", type=float, default=value("lc0_min_best_legal_pct", d.lc0_min_best_legal_pct))
    parser.add_argument("--lc0-child-targets", default=value("lc0_child_targets", d.lc0_child_targets))
    parser.add_argument("--lc0-child-summary", default=value("lc0_child_summary", d.lc0_child_summary))
    parser.add_argument("--lc0-child-min-groups", type=int, default=value("lc0_child_min_groups", d.lc0_child_min_groups))
    parser.add_argument("--lc0-child-max-groups", type=int, default=value("lc0_child_max_groups", d.lc0_child_max_groups))
    parser.add_argument("--lc0-child-unique-fen", action=argparse.BooleanOptionalAction,
                        default=value("lc0_child_unique_fen", d.lc0_child_unique_fen))
    parser.add_argument("--lc0-best-source", default=value("lc0_best_source", d.lc0_best_source),
                        choices=["top-policy", "best", "played"])
    parser.add_argument("--lc0-policy-score-scale-cp", type=float, default=value("lc0_policy_score_scale_cp", d.lc0_policy_score_scale_cp))
    parser.add_argument("--lc0-policy-floor", type=float, default=value("lc0_policy_floor", d.lc0_policy_floor))
    parser.add_argument("--lc0-child-max-gap-cp", type=float, default=value("lc0_child_max_gap_cp", d.lc0_child_max_gap_cp))
    parser.add_argument("--lc0-min-best-policy", type=float, default=value("lc0_min_best_policy", d.lc0_min_best_policy))
    parser.add_argument("--lc0-min-policy-gap-cp", type=float, default=value("lc0_min_policy_gap_cp", d.lc0_min_policy_gap_cp))


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
