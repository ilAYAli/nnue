#!/usr/bin/env python3
"""High-level Enyo NNUE candidate workflow command."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
import os
import shlex
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
    allowed.update({
        "name",
        "run_dir",
        "dry_run",
        "force",
        "event_command",
        "disabled",
        "disabled_reason",
    })
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


def templated_path_arg(value: str | Path) -> str:
    text = str(value)
    if "{" in text and "}" in text:
        return text
    return str(expand_path(text))


def templated_source_spec_arg(value: str | Path) -> str:
    text = str(value)
    try:
        path, rows, max_abs_cp = text.rsplit(":", 2)
    except ValueError as exc:
        try:
            path, rows = text.rsplit(":", 1)
        except ValueError as inner_exc:
            raise SystemExit(f"invalid source mix spec: {value}") from inner_exc
        max_abs_cp = ""
    if not path or not rows:
        raise SystemExit(f"invalid source mix spec: {value}")
    if max_abs_cp:
        return f"{templated_path_arg(path)}:{rows}:{max_abs_cp}"
    return f"{templated_path_arg(path)}:{rows}"


def append_source_mix_step(
    steps: list[dict],
    args: argparse.Namespace,
) -> str:
    python = str(expand_user(args.python))
    mix_output = "{score}/mixed.jsonl"
    command = [
        python, tool("posgen/mix_jsonl.py"),
        "--output", mix_output,
        "--seed", str(args.source_mix_seed),
        "--progress", str(args.source_mix_progress),
    ]
    for spec in args.source_mix_jsonl:
        command.extend(["--source", templated_source_spec_arg(spec)])
    steps.append({
        "name": "source_mix",
        "command": command,
    })
    return mix_output


def sibling_rows_file_arg(value: str | Path) -> str:
    text = str(value)
    if "{" in text and "}" in text:
        return ""
    path = expand_path(text)
    rows_file = path.with_suffix(".wc")
    return str(rows_file) if rows_file.exists() else ""


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def borrowed_selfplay_nnue_reason(value: str | Path | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return "empty nnue_file would use the embedded default evaluator"
    name = Path(os.path.expandvars(text)).expanduser().name.lower()
    if name == "default.net":
        return "default.net is a borrowed-weight source"
    if "berserk" in name:
        return "Berserk is a borrowed-weight source"
    return None


def validate_create_args(args: argparse.Namespace) -> None:
    if args.selfplay_crucible:
        if args.selfplay_games <= 0:
            raise SystemExit("selfplay_crucible requires selfplay_games > 0")
        if args.selfplay_shard_games <= 0:
            raise SystemExit("selfplay_crucible requires selfplay_shard_games > 0")
        if args.selfplay_crucible_local_slots < 0:
            raise SystemExit("selfplay_crucible_local_slots must be >= 0")

    if args.score_crucible:
        if args.score_shards <= 0:
            raise SystemExit("score_crucible requires score_shards > 0")
        if args.score_crucible_local_slots < 0:
            raise SystemExit("score_crucible_local_slots must be >= 0")
    if args.score_backend == "stockfish-static":
        if args.score_output_format != "bullet-data":
            raise SystemExit("score_backend=stockfish-static requires score_output_format=bullet-data")
        if not args.score_stockfish_net:
            raise SystemExit("score_backend=stockfish-static requires score_stockfish_net")
        if args.score_min_delta_cp and not args.score_enyo_net:
            raise SystemExit("score_min_delta_cp requires score_enyo_net")
    if args.score_output_format == "packed" and args.backend != "pytorch":
        raise SystemExit("score_output_format=packed currently requires backend=pytorch")
    if args.score_output_format in {"bullet-text", "bullet-data"}:
        if args.backend != "bullet":
            raise SystemExit(
                f"score_output_format={args.score_output_format} requires backend=bullet"
            )
        if args.bullet_loader != "direct":
            raise SystemExit(
                f"score_output_format={args.score_output_format} requires bullet_loader=direct"
            )

    if args.bullet_init_net and args.bullet_init_weights:
        raise SystemExit("bullet_init_net conflicts with bullet_init_weights")
    if args.bullet_enyo_feature_channels == 11 and (
        args.bullet_enyo_input_buckets != 32
        or args.bullet_enyo_runtime_input_buckets != 32
    ):
        raise SystemExit(
            "bullet_enyo_feature_channels=11 requires "
            "bullet_enyo_input_buckets=32 and "
            "bullet_enyo_runtime_input_buckets=32")

    if args.source_mix_jsonl:
        if args.bullet_generate_source:
            raise SystemExit("source_mix_jsonl conflicts with bullet_generate_source")
        if args.score_source_jsonl:
            raise SystemExit("source_mix_jsonl conflicts with score_source_jsonl")
        if args.labeled_jsonl:
            raise SystemExit("source_mix_jsonl conflicts with labeled_jsonl")
        if args.backend == "bullet":
            if args.bullet_data:
                raise SystemExit("source_mix_jsonl conflicts with bullet_data")
            if args.bullet_source_jsonl:
                raise SystemExit("source_mix_jsonl conflicts with bullet_source_jsonl")

    if args.labeled_jsonl:
        if args.backend not in {"pytorch", "pairwise"}:
            raise SystemExit("labeled_jsonl currently requires backend=pytorch or pairwise")
        if args.score_source_jsonl:
            raise SystemExit("labeled_jsonl conflicts with score_source_jsonl")

    if args.backend == "pairwise":
        if not args.pairwise_pairs_jsonl:
            raise SystemExit("backend=pairwise requires pairwise_pairs_jsonl")
        if not (args.pairwise_data or args.labeled_jsonl):
            raise SystemExit("backend=pairwise requires pairwise_data or labeled_jsonl")
        if not (args.pairwise_init_from_nn or args.init_net):
            raise SystemExit("backend=pairwise requires pairwise_init_from_nn or init_net")

    if not (args.require_clean_enyo_owned and args.bullet_generate_source):
        return
    if not args.selfplay_use_nnue:
        return
    reason = borrowed_selfplay_nnue_reason(args.nnue_file)
    if reason:
        raise SystemExit(
            "clean Enyo-owned source generation requires an explicit clean "
            f"self-play evaluator; {reason}"
        )


def append_score_steps(
    steps: list[dict],
    args: argparse.Namespace,
    *,
    input_jsonl: str,
) -> None:
    if args.score_crucible:
        append_crucible_score_steps(steps, args, input_jsonl=input_jsonl)
    else:
        append_local_score_steps(steps, args, input_jsonl=input_jsonl)


def score_output_extension(args: argparse.Namespace) -> str:
    if args.score_output_format == "packed":
        return "npz"
    if args.score_output_format == "bullet-text":
        return "txt"
    if args.score_output_format == "bullet-data":
        return "data"
    return "jsonl"


def score_command(
    args: argparse.Namespace,
    *,
    input_jsonl: str,
    output: str,
    shard_count: str,
    shard_index: str,
) -> list[str]:
    if args.score_backend == "stockfish-static":
        command = [
            str(expand_user(args.score_static_datagen_tool)),
            "--stockfish-net", str(expand_user(args.score_stockfish_net)),
            "--input", input_jsonl,
            "--output", output,
            "--shard-count", shard_count,
            "--shard-index", shard_index,
            "--limit", str(args.score_limit),
            "--max-abs-cp", str(args.score_max_abs_cp),
            "--progress", str(args.score_progress),
        ]
        if args.score_enyo_net:
            command.extend(["--enyo-net", str(expand_user(args.score_enyo_net))])
        if args.score_min_delta_cp:
            command.extend(["--min-delta-cp", str(args.score_min_delta_cp)])
        return command

    python = str(expand_user(args.python))
    command = [
        python, tool("score/score.py"), "uci",
        "--input", input_jsonl,
        "--output", output,
        "--engine", str(expand_user(args.score_engine)),
        "--depth", str(args.score_depth),
        "--threads", str(args.score_threads),
        "--hash", str(args.score_hash),
        "--shard-count", shard_count,
        "--shard-index", shard_index,
        "--limit", str(args.score_limit),
        "--max-abs-cp", str(args.score_max_abs_cp),
        "--progress", str(args.score_progress),
        "--output-format", args.score_output_format,
        "--max-features", str(args.max_features),
    ]
    if (
        args.score_output_format in {"bullet-text", "bullet-data"}
        and args.bullet_enyo_runtime_target
    ):
        command.append("--enyo-runtime-target")
    return command


def append_packed_score_pack_step(
    steps: list[dict],
    args: argparse.Namespace,
    *,
    python: str,
    prefix_command: list[str] | None = None,
) -> None:
    command = [*(prefix_command if prefix_command is not None else [python])]
    command.extend([
        tool("pack/pack.py"), "merge-shards",
        "--input-dir", "{score}/shards",
        "--out-dir", "{pack}/train",
        "--pattern", "label.*.npz",
        "--max-features", str(args.max_features),
        "--progress", str(args.pack_progress),
        "--python", python,
    ])
    steps.append({
        "name": "pack",
        "command": command,
    })


def append_local_score_steps(
    steps: list[dict],
    args: argparse.Namespace,
    *,
    input_jsonl: str,
) -> None:
    extension = score_output_extension(args)
    for shard in range(args.score_shards):
        steps.append({
            "name": f"score_{shard:02d}",
            "parallel_group": "score",
            "command": score_command(
                args,
                input_jsonl=input_jsonl,
                output=f"{{score}}/shards/label.{shard}.{extension}",
                shard_count=str(args.score_shards),
                shard_index=str(shard),
            ),
        })

    if args.score_output_format == "packed":
        append_packed_score_pack_step(
            steps,
            args,
            python=str(expand_user(args.python)),
        )
        return

    if args.score_output_format == "bullet-text":
        merged = "{score}/labeled.txt"
    elif args.score_output_format == "bullet-data":
        merged = "{score}/labeled.data"
    else:
        merged = "{score}/labeled.jsonl"
    steps.append({
        "name": "score_merge",
        "command": [
            "bash", "-lc",
            (
                "cat \"$1\"/shards/label.*.\"$2\" > \"$3\" && "
                "if [ \"$2\" = data ]; then "
                "\"$5\" \"$6\" \"$3\" > \"$4\"; "
                "else wc -l \"$3\" > \"$4\"; fi"
            ),
            "merge-score", "{score}", extension, merged, "{score}/labeled.wc",
            str(expand_user(args.python)), tool("bullet/count_bullet_data.py"),
        ],
    })


def append_crucible_deploy_step(
    steps: list[dict],
    *,
    name: str,
    crucible_python: str,
    crucible: str,
    workers: str,
    manifest: str,
    jobs: int,
    coordinator_host: str,
    remote_timeout_seconds: int,
    verbose: bool,
) -> None:
    command = [
        "bash", "-lc",
        (
            "python=\"$1\" crucible=\"$2\" workers=\"$3\" manifest=\"$4\" "
            "jobs=\"$5\" coordinator_host=\"$6\" timeout=\"$7\" verbose=\"$8\"; "
            "tmp=$(mktemp); "
            "cmd=(\"$python\" \"$crucible\" deploy \"$workers\" \"$manifest\" "
            "--jobs \"$jobs\" --remote-timeout-seconds \"$timeout\"); "
            "if [ -n \"$coordinator_host\" ]; then "
            "cmd+=(--coordinator-host \"$coordinator_host\"); fi; "
            "if [ \"$verbose\" = 1 ]; then cmd+=(--verbose); fi; "
            "\"${{cmd[@]}}\" 2>&1 | tee \"$tmp\"; "
            "rc=${{PIPESTATUS[0]}}; "
            "if [ \"$rc\" -eq 0 ]; then rm -f \"$tmp\"; exit 0; fi; "
            "if grep -q 'run already exists; pass --resume or --replace' \"$tmp\"; then "
            "rm -f \"$tmp\"; \"${{cmd[@]}}\" --resume; exit $?; fi; "
            "rm -f \"$tmp\"; exit \"$rc\""
        ),
        "crucible-deploy",
        crucible_python,
        crucible,
        str(expand_path(workers)),
        manifest,
        str(jobs),
        str(coordinator_host),
        str(remote_timeout_seconds),
        "1" if verbose else "0",
    ]
    steps.append({
        "name": name,
        "command": command,
    })


def append_crucible_score_steps(
    steps: list[dict],
    args: argparse.Namespace,
    *,
    input_jsonl: str,
) -> None:
    crucible_python = str(expand_user(args.score_crucible_python or sys.executable))
    crucible = str(expand_user(args.score_crucible_tool))
    manifest = "{score}/crucible/manifest.json"
    merge_run = "score-{candidate}" if args.score_crucible_workers else "{score}/crucible"
    score_command_template = shell_join(score_command(
        args,
        input_jsonl="{{source}}",
        output="{{output}}",
        shard_count="{{shards}}",
        shard_index="{{index}}",
    ))
    extension = score_output_extension(args)
    task_label = "score:stockfish-static" if args.score_backend == "stockfish-static" else "score:uci"
    plan_command = [
        crucible_python, crucible, "plan",
        "--name", "score-{candidate}",
        "--kind", "label",
        "--description", "NNUE source scoring",
        "--task-label", task_label,
        "--shards", str(args.score_shards),
        "--work-dir", "{repo}",
        "--out", manifest,
        "--state-dir", "{score}/crucible/state",
        "--log-dir", "{score}/crucible/logs",
        "--command-template", score_command_template,
        "--var", f"source={input_jsonl}",
        "--output-template", f"{{score}}/shards/label.{{{{index}}}}.{extension}",
        "--progress-unit", "rows",
        "--progress-total-lines", input_jsonl,
        "--progress-log-regex", r"selected=(?P<done>\d+)\s+written=(?P<output>\d+)\s+rate=(?P<rate>[0-9.]+)/s",
        "--progress-output-unit", "output",
    ]
    for mapping in args.score_crucible_path_map or []:
        plan_command.extend(["--path-map", str(mapping)])

    steps.append({
        "name": "score_crucible_plan",
        "command": plan_command,
    })

    steps.append({
        "name": "score_crucible_add_input",
        "command": [
            crucible_python, crucible, "add-input",
            "--manifest", manifest,
            "--path", input_jsonl,
        ],
    })

    doctor_command = [
        crucible_python, crucible, "doctor",
        "--manifest", manifest,
        "--role", "coordinator",
    ]
    if args.score_crucible_require_notify:
        doctor_command.extend([
            "--notify-command",
            str(expand_user(args.score_crucible_notify_command)),
        ])
    steps.append({
        "name": "score_crucible_doctor",
        "command": doctor_command,
    })

    if args.score_crucible_workers:
        append_crucible_deploy_step(
            steps,
            name="score_crucible_deploy",
            crucible_python=crucible_python,
            crucible=crucible,
            workers=args.score_crucible_workers,
            manifest=manifest,
            jobs=args.score_crucible_jobs,
            coordinator_host=args.score_crucible_coordinator_host,
            remote_timeout_seconds=args.score_crucible_remote_timeout_seconds,
            verbose=args.score_crucible_verbose,
        )
    else:
        for slot in range(args.score_crucible_local_slots):
            steps.append({
                "name": f"score_crucible_work_{slot:02d}",
                "parallel_group": "score_crucible_work",
                "command": [
                    crucible_python, crucible, "work",
                    "--manifest", manifest,
                    "--coordinator", manifest,
                    "--worker", f"coordinator-{slot:02d}",
                    "--lease-seconds", str(args.score_crucible_lease_seconds),
                ],
            })

        steps.append({
            "name": "score_crucible_wait",
            "command": [
                crucible_python, crucible, "wait",
                "--manifest", manifest,
                "--lease-seconds", str(args.score_crucible_lease_seconds),
            ],
        })

    if args.score_output_format == "packed":
        append_packed_score_pack_step(
            steps,
            args,
            python=crucible_python,
            prefix_command=[
                "bash", "-lc",
                (
                    "python=\"$1\" crucible=\"$2\" run=\"$3\" pack_py=\"$4\"; "
                    "\"$python\" \"$crucible\" verify \"$run\" && "
                    "shift 4 && \"$python\" \"$pack_py\" \"$@\""
                ),
                "pack-crucible-score",
                crucible_python,
                crucible,
                merge_run,
            ],
        )
    else:
        merged = (
            "{score}/labeled.txt"
            if args.score_output_format == "bullet-text"
            else "{score}/labeled.data"
            if args.score_output_format == "bullet-data"
            else "{score}/labeled.jsonl"
        )
        steps.append({
            "name": "score_crucible_merge",
            "command": [
                "bash", "-lc",
                (
                    "python=\"$1\" crucible=\"$2\" run=\"$3\" output=\"$4\" rows=\"$5\" "
                    "format=\"$6\"; "
                    "\"$python\" \"$crucible\" verify \"$run\" && "
                    "\"$python\" \"$crucible\" merge \"$run\" --output \"$output\" --force && "
                    "if [ \"$format\" = data ]; then "
                    "\"$python\" \"$7\" \"$output\" > \"$rows\"; "
                    "else wc -l \"$output\" > \"$rows\"; fi"
                ),
                "merge-crucible-score",
                crucible_python,
                crucible,
                merge_run,
                merged,
                "{score}/labeled.wc",
                extension,
                tool("bullet/count_bullet_data.py"),
            ],
        })


def selfplay_shard_count(args: argparse.Namespace) -> int:
    return max(1, (args.selfplay_games + args.selfplay_shard_games - 1) // args.selfplay_shard_games)


def selfplay_engine_options(args: argparse.Namespace) -> list[str]:
    options = [f"Hash={args.selfplay_hash}"]
    if not args.selfplay_use_nnue:
        options.append("use_nnue=false")
    return options


def selfplay_generate_command(
    args: argparse.Namespace,
    *,
    output_pgn: str,
    metadata: str,
    total_games: str,
    shards: str,
    shard_index: str,
) -> list[str]:
    python = str(expand_user(args.python))
    command = [
        python, tool("posgen/selfplay_shards.py"), "generate",
        "--runner", str(expand_user(args.runner)),
        "--engine", str(expand_user(args.engine)),
        "--book", str(expand_path(args.book)),
        "--output-pgn", output_pgn,
        "--total-games", total_games,
        "--shards", shards,
        "--shard-index", shard_index,
        "--concurrency", str(args.selfplay_concurrency),
        "--threads", str(args.selfplay_threads),
        "--depth", str(args.selfplay_depth),
        "--base-seed", str(args.selfplay_seed),
        "--restart", "off",
    ]
    if metadata:
        command.extend(["--metadata", metadata])
    if args.nnue_file:
        command.extend(["--nnue-file", str(expand_path(args.nnue_file))])
    for option in selfplay_engine_options(args):
        command.extend(["--engine-option", option])
    return command


def append_local_selfplay_steps(steps: list[dict], args: argparse.Namespace) -> None:
    python = str(expand_user(args.python))
    selfplay_command = [
        python, tool("posgen/posgen.py"), "selfplay",
        "--runner", str(expand_user(args.runner)),
        "--engine", str(expand_user(args.engine)),
        "--book", str(expand_path(args.book)),
        "--output", "{posgen}/selfplay.pgn",
        "--games", str(args.selfplay_games),
        "--shard-games", str(args.selfplay_shard_games),
        "--concurrency", str(args.selfplay_concurrency),
        "--threads", str(args.selfplay_threads),
        "--depth", str(args.selfplay_depth),
        "--srand", str(args.selfplay_seed),
        "--restart", "off",
    ]
    if args.nnue_file:
        selfplay_command.extend(["--nnue-file", str(expand_path(args.nnue_file))])
    for option in selfplay_engine_options(args):
        selfplay_command.extend(["--engine-option", option])
    steps.append({
        "name": "posgen_selfplay",
        "command": selfplay_command,
    })


def append_crucible_selfplay_steps(steps: list[dict], args: argparse.Namespace) -> None:
    crucible_python = str(expand_user(args.selfplay_crucible_python or sys.executable))
    crucible = str(expand_user(args.selfplay_crucible_tool))
    manifest = "{posgen}/selfplay_crucible/manifest.json"
    merge_run = "selfplay-{candidate}" if args.selfplay_crucible_workers else "{posgen}/selfplay_crucible"
    shards = selfplay_shard_count(args)
    shard_pgn = "{posgen}/selfplay_shards/shard.{{index}}.pgn"
    command_template = shell_join(selfplay_generate_command(
        args,
        output_pgn="{{output}}",
        metadata="",
        total_games="{{total_games}}",
        shards="{{shards}}",
        shard_index="{{index}}",
    ))
    plan_command = [
        crucible_python, crucible, "plan",
        "--name", "selfplay-{candidate}",
        "--kind", "selfplay",
        "--description", "NNUE selfplay generation",
        "--task-label", "selfplay",
        "--shards", str(shards),
        "--work-dir", "{repo}",
        "--out", manifest,
        "--state-dir", "{posgen}/selfplay_crucible/state",
        "--log-dir", "{posgen}/selfplay_crucible/logs",
        "--command-template", command_template,
        "--var", f"total_games={args.selfplay_games}",
        "--output-template", shard_pgn,
        "--progress-unit", "games",
        "--progress-total", str(args.selfplay_shard_games),
        "--progress-log-regex", r"\[\s*\d+/\d+\s+(?P<done>\d+)\s*/\s*(?P<total>\d+)\]",
    ]
    for mapping in args.selfplay_crucible_path_map or []:
        plan_command.extend(["--path-map", str(mapping)])
    steps.append({
        "name": "selfplay_crucible_plan",
        "command": plan_command,
    })

    input_command = [
        crucible_python, crucible, "add-input",
        "--manifest", manifest,
        "--path", str(expand_path(args.book)),
    ]
    if args.nnue_file:
        input_command.extend(["--path", str(expand_path(args.nnue_file))])
    steps.append({
        "name": "selfplay_crucible_add_input",
        "command": input_command,
    })

    doctor_command = [
        crucible_python, crucible, "doctor",
        "--manifest", manifest,
        "--role", "coordinator",
    ]
    if args.selfplay_crucible_require_notify:
        doctor_command.extend([
            "--notify-command",
            str(expand_user(args.selfplay_crucible_notify_command)),
        ])
    steps.append({
        "name": "selfplay_crucible_doctor",
        "command": doctor_command,
    })

    if args.selfplay_crucible_workers:
        append_crucible_deploy_step(
            steps,
            name="selfplay_crucible_deploy",
            crucible_python=crucible_python,
            crucible=crucible,
            workers=args.selfplay_crucible_workers,
            manifest=manifest,
            jobs=args.selfplay_crucible_jobs,
            coordinator_host=args.selfplay_crucible_coordinator_host,
            remote_timeout_seconds=args.selfplay_crucible_remote_timeout_seconds,
            verbose=args.selfplay_crucible_verbose,
        )
    else:
        for slot in range(args.selfplay_crucible_local_slots):
            steps.append({
                "name": f"selfplay_crucible_work_{slot:02d}",
                "parallel_group": "selfplay_crucible_work",
                "command": [
                    crucible_python, crucible, "work",
                    "--manifest", manifest,
                    "--coordinator", manifest,
                    "--worker", f"coordinator-selfplay-{slot:02d}",
                    "--lease-seconds", str(args.selfplay_crucible_lease_seconds),
                ],
            })

        steps.append({
            "name": "selfplay_crucible_wait",
            "command": [
                crucible_python, crucible, "wait",
                "--manifest", manifest,
                "--lease-seconds", str(args.selfplay_crucible_lease_seconds),
            ],
        })

    steps.append({
        "name": "selfplay_crucible_merge",
        "command": [
            "bash", "-lc",
            (
                "python=\"$1\" crucible=\"$2\" run=\"$3\" posgen=\"$4\" shard_tool=\"$5\" "
                "expected_shards=\"$6\" expected_games=\"$7\"; "
                "\"$python\" \"$crucible\" verify \"$run\" && "
                "shopt -s nullglob && pgns=(\"$posgen\"/selfplay_shards/shard.*.pgn) && "
                "if [ \"${{#pgns[@]}}\" -ne \"$expected_shards\" ]; then "
                "echo \"self-play PGN shard count ${{#pgns[@]}} != expected $expected_shards\"; exit 1; fi && "
                "\"$python\" \"$shard_tool\" merge-pgns --output-pgn \"$posgen/selfplay.pgn\" "
                "--manifest \"$posgen/selfplay_manifest.json\" --expected-games \"$expected_games\" "
                "--force \"${{pgns[@]}}\""
            ),
            "merge-crucible-selfplay",
            crucible_python,
            crucible,
            merge_run,
            "{posgen}",
            tool("posgen/selfplay_shards.py"),
            str(shards),
            str(args.selfplay_games),
        ],
    })


def append_position_source_steps(steps: list[dict], args: argparse.Namespace) -> None:
    if args.selfplay_crucible:
        append_crucible_selfplay_steps(steps, args)
    else:
        append_local_selfplay_steps(steps, args)

    python = str(expand_user(args.python))
    steps.extend([
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


def append_source_generation_steps(
    steps: list[dict],
    args: argparse.Namespace,
) -> None:
    append_position_source_steps(steps, args)
    append_score_steps(steps, args, input_jsonl="{posgen}/source.jsonl")


def append_provenance_step(
    steps: list[dict],
    *,
    args: argparse.Namespace,
    python: str,
    net: str,
) -> None:
    if not (args.validate_provenance or args.require_clean_enyo_owned):
        return
    command = [
        python, tool("validate/net_provenance.py"),
        "--net", net,
    ]
    if args.require_clean_enyo_owned:
        command.append("--require-clean-enyo-owned")
    steps.append({
        "name": "validate_provenance",
        "command": command,
    })


def append_engine_static_step(
    steps: list[dict],
    args: argparse.Namespace,
    *,
    name: str,
    net: str,
    jsonl: str,
) -> None:
    python = str(expand_user(args.python))
    steps.append({
        "name": name,
        "command": [
            python, tool("validate/eval_jsonl_engine.py"),
            "--engine", str(expand_path(args.engine_static_engine)),
            "--net", net,
            "--jsonl", templated_path_arg(jsonl),
            "--rows", str(args.engine_static_rows),
            "--buckets",
            "--sources",
        ],
    })


def append_bullet_format_step(
    steps: list[dict],
    args: argparse.Namespace,
    *,
    python: str,
    input_text: str,
    output_data: str,
) -> None:
    steps.append({
        "name": "bullet_format",
        "command": [
            python, tool("bullet/bullet.py"), "format",
            "--input", input_text,
            "--output", output_data,
            "--bullet-manifest", str(expand_path(args.bullet_manifest)),
            "--validate",
        ],
    })


def create_config(args: argparse.Namespace) -> dict:
    if args.disabled:
        reason = str(args.disabled_reason or "no candidate build is selected")
        raise SystemExit(f"build config is disabled: {reason}")

    validate_create_args(args)
    name = args.name or default_name()
    run_dir = run_dir_for(name, args.run_dir)
    candidate_dir = f"{{train}}/{name}"
    python = str(expand_user(args.python))
    steps: list[dict] = []
    engine_static_done = False

    if args.backend == "pytorch":
        packed_score_data = False
        if args.source_mix_jsonl:
            labeled_jsonl = append_source_mix_step(steps, args)
            rows_file = ""
            if not args.engine_static_jsonl:
                args.engine_static_jsonl = labeled_jsonl
        elif args.labeled_jsonl:
            labeled_jsonl = templated_path_arg(args.labeled_jsonl)
            rows_file = sibling_rows_file_arg(args.labeled_jsonl)
            if not args.engine_static_jsonl:
                args.engine_static_jsonl = labeled_jsonl
        else:
            append_source_generation_steps(steps, args)
            if args.score_output_format == "packed":
                packed_score_data = True
                labeled_jsonl = ""
                rows_file = ""
            else:
                labeled_jsonl = "{score}/labeled.jsonl"
                rows_file = "{score}/labeled.wc"

        if not packed_score_data:
            pack_command = [
                python, tool("pack/pack.py"), "build",
                "--input", labeled_jsonl,
                "--out-dir", "{pack}/train",
                "--max-features", str(args.max_features),
                "--progress", str(args.pack_progress),
                "--python", python,
            ]
            if rows_file:
                pack_command.extend(["--rows-file", rows_file])
            steps.append({
                "name": "pack",
                "command": pack_command,
            })

        steps.extend([
            {
                "name": "train",
                "command": [
                    python, tool("train/train.py"), "run",
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
                    "--output-head-features", args.output_head_features,
                    "--python", python,
                    "--out", f"{candidate_dir}/model.pt",
                    "--out-nn", f"{candidate_dir}/model.nn",
                ],
            },
        ])
        append_provenance_step(
            steps,
            args=args,
            python=python,
            net=f"{candidate_dir}/model.nn",
        )
    elif args.backend == "pairwise":
        if args.pairwise_data:
            pairwise_data = templated_path_arg(args.pairwise_data)
        else:
            labeled_jsonl = templated_path_arg(args.labeled_jsonl)
            rows_file = sibling_rows_file_arg(args.labeled_jsonl)
            pack_command = [
                python, tool("pack/pack.py"), "build",
                "--input", labeled_jsonl,
                "--out-dir", "{pack}/train",
                "--max-features", str(args.max_features),
                "--progress", str(args.pack_progress),
                "--python", python,
            ]
            if rows_file:
                pack_command.extend(["--rows-file", rows_file])
            steps.append({
                "name": "pack",
                "command": pack_command,
            })
            pairwise_data = "{pack}/train"
            if not args.engine_static_jsonl:
                args.engine_static_jsonl = labeled_jsonl

        init_from_nn = templated_path_arg(args.pairwise_init_from_nn or args.init_net)
        steps.append({
            "name": "train_pairwise",
            "command": [
                python, tool("train/train_pairwise.py"),
                "--data", pairwise_data,
                "--pairs", templated_path_arg(args.pairwise_pairs_jsonl),
                "--out", f"{candidate_dir}/model.pt",
                "--out-nn", f"{candidate_dir}/model.nn",
                "--checkpoint-dir", f"{candidate_dir}/checkpoints",
                "--checkpoint-every", str(args.pairwise_checkpoint_every),
                "--init-from-nn", init_from_nn,
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--pair-batch-size", str(args.pairwise_pair_batch_size),
                "--lr", str(args.lr),
                "--weight-decay", str(args.weight_decay),
                "--huber-beta", str(args.huber_beta),
                "--pair-beta", str(args.pairwise_pair_beta),
                "--pair-weight", str(args.pairwise_pair_weight),
                "--broad-weight", str(args.pairwise_broad_weight),
                "--steps-per-epoch", str(args.pairwise_steps_per_epoch),
                "--target-clamp", str(args.target_clamp),
                "--max-target-margin", str(args.pairwise_max_target_margin),
                "--min-target-margin", str(args.pairwise_min_target_margin),
                *(["--loss-weight-by-cp"] if args.pairwise_loss_weight_by_cp else []),
                "--device", args.device,
                "--workers", str(args.workers),
                "--max-rows", str(args.pairwise_max_rows),
                "--skip-rows", str(args.pairwise_skip_rows),
            ],
        })
        append_provenance_step(
            steps,
            args=args,
            python=python,
            net=f"{candidate_dir}/model.nn",
        )
        if args.engine_static_jsonl:
            baseline_net = templated_path_arg(
                args.pairwise_move_gate_baseline_net
                or args.pairwise_init_from_nn
                or args.init_net
            )
            append_engine_static_step(
                steps,
                args,
                name="validate_engine_static_baseline",
                net=baseline_net,
                jsonl=args.engine_static_jsonl,
            )
            append_engine_static_step(
                steps,
                args,
                name="validate_engine_static",
                net=f"{candidate_dir}/model.nn",
                jsonl=args.engine_static_jsonl,
            )
            engine_static_done = True
        if args.pairwise_move_gate_cases:
            move_gate_command = [
                python, tool("validate/eval_move_gate.py"),
                "--cases", templated_path_arg(args.pairwise_move_gate_cases),
                "--engine", str(expand_path(args.engine_static_engine)),
                "--baseline-net", templated_path_arg(
                    args.pairwise_move_gate_baseline_net
                    or args.pairwise_init_from_nn
                    or args.init_net
                ),
                "--candidate-net", f"{candidate_dir}/model.nn",
                "--limit", str(args.pairwise_move_gate_limit),
                "--output", "{validate}/move_gate.jsonl",
                "--summary-json", "{validate}/move_gate.summary.json",
            ]
            if args.pairwise_move_gate_fail_candidate_below_baseline:
                move_gate_command.append("--fail-if-candidate-below-baseline")
            if args.pairwise_move_gate_fail_regressed_above >= 0:
                move_gate_command.extend([
                    "--fail-if-regressed-above",
                    str(args.pairwise_move_gate_fail_regressed_above),
                ])
            if args.pairwise_move_gate_fail_fixed_below >= 0:
                move_gate_command.extend([
                    "--fail-if-fixed-below",
                    str(args.pairwise_move_gate_fail_fixed_below),
                ])
            move_gate_command.extend([
                "--fail-if-delta-below",
                str(args.pairwise_move_gate_fail_delta_below),
                "--fail-if-loss-weighted-delta-below",
                str(args.pairwise_move_gate_fail_loss_weighted_delta_below),
            ])
            steps.append({
                "name": "validate_move_gate",
                "command": move_gate_command,
            })
    elif args.backend == "bullet":
        bullet_source_text = ""
        bullet_source_data = ""
        if args.source_mix_jsonl:
            mix_output = append_source_mix_step(steps, args)
            args.bullet_source_jsonl = mix_output
            if not args.engine_static_jsonl:
                args.engine_static_jsonl = mix_output
        elif args.score_source_jsonl:
            append_score_steps(
                steps,
                args,
                input_jsonl=templated_path_arg(args.score_source_jsonl),
            )
            if args.score_output_format == "bullet-text":
                bullet_source_text = "{score}/labeled.txt"
            elif args.score_output_format == "bullet-data":
                bullet_source_data = "{score}/labeled.data"
            else:
                args.bullet_source_jsonl = "{score}/labeled.jsonl"
                if not args.engine_static_jsonl:
                    args.engine_static_jsonl = "{score}/labeled.jsonl"
        elif args.bullet_generate_source:
            append_source_generation_steps(steps, args)
            if args.score_output_format == "bullet-text":
                bullet_source_text = "{score}/labeled.txt"
            elif args.score_output_format == "bullet-data":
                bullet_source_data = "{score}/labeled.data"
            else:
                args.bullet_source_jsonl = "{score}/labeled.jsonl"
            if (
                not args.engine_static_jsonl
                and args.score_output_format not in {"bullet-text", "bullet-data"}
            ):
                args.engine_static_jsonl = "{score}/labeled.jsonl"

        if args.bullet_loader == "direct":
            if args.bullet_data:
                bullet_data = str(expand_user(args.bullet_data))
            else:
                if not (bullet_source_text or bullet_source_data or args.bullet_source_jsonl):
                    raise SystemExit(
                        "backend=bullet direct loader requires scored source data or bullet_data"
                    )
                bullet_text = "{assets}/bullet.txt"
                bullet_data = "{assets}/bullet.data"
                if bullet_source_data:
                    bullet_data = bullet_source_data
                elif bullet_source_text:
                    bullet_text = bullet_source_text
                    append_bullet_format_step(
                        steps,
                        args,
                        python=python,
                        input_text=bullet_text,
                        output_data=bullet_data,
                    )
                else:
                    steps.extend([
                        {
                            "name": "bullet_text",
                            "command": [
                                python, tool("bullet/jsonl_to_bullet_text.py"),
                                "--input", templated_path_arg(args.bullet_source_jsonl),
                                "--output", bullet_text,
                                "--limit", str(args.bullet_limit),
                                "--max-abs-cp", str(args.bullet_max_abs_cp),
                                *(
                                    ["--enyo-runtime-target"]
                                    if args.bullet_enyo_runtime_target else []
                                ),
                            ],
                        },
                    ])
                    append_bullet_format_step(
                        steps,
                        args,
                        python=python,
                        input_text=bullet_text,
                        output_data=bullet_data,
                    )
        elif args.bullet_loader == "sfbinpack":
            if not args.bullet_data:
                raise SystemExit("backend=bullet sfbinpack loader requires bullet_data")
            converted_data = "{assets}/bullet.data"
            steps.append({
                "name": "bullet_convert",
                "command": [
                    python, tool("bullet/bullet.py"), "convert",
                    "--data", args.bullet_data,
                    "--output", converted_data,
                    "--cargo-target-dir", "{run}/cargo-target",
                    "--buffer-mb", str(args.bullet_sfbinpack_buffer_mb),
                    "--threads", str(args.bullet_threads),
                    "--limit", str(args.bullet_limit),
                    "--min-ply", str(args.bullet_sfbinpack_min_ply),
                    "--max-abs-cp", str(args.bullet_sfbinpack_max_abs_cp),
                    (
                        "--quiet-only"
                        if args.bullet_sfbinpack_quiet_only
                        else "--no-quiet-only"
                    ),
                ],
            })
            bullet_data = converted_data
            bullet_loader = "direct"
        else:
            raise SystemExit(f"unsupported bullet_loader={args.bullet_loader}")

        if args.bullet_loader == "direct":
            bullet_loader = "direct"

        if args.bullet_init_net:
            args.bullet_init_weights = "{assets}/init/optimiser_state/weights.bin"
            steps.append({
                "name": "bullet_init_weights",
                "command": [
                    python, tool("bullet/enyo_nn_to_bullet_weights.py"),
                    "--input", str(expand_path(args.bullet_init_net)),
                    "--output", args.bullet_init_weights,
                    "--eval-scale", str(args.bullet_eval_scale),
                    "--l1-export-scale", str(args.bullet_enyo_l1_export_scale),
                    "--input-buckets", str(args.bullet_enyo_input_buckets),
                    "--feature-channels", str(args.bullet_enyo_feature_channels),
                    "--output-buckets", str(args.bullet_enyo_output_buckets),
                ],
            })

        steps.append({
            "name": "bullet_train",
            "command": [
                python, tool("bullet/bullet.py"), "train",
                "--data", bullet_data,
                "--loader", bullet_loader,
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
                "--enyo-feature-channels", str(args.bullet_enyo_feature_channels),
                "--enyo-runtime-input-buckets",
                str(args.bullet_enyo_runtime_input_buckets),
                "--enyo-output-buckets", str(args.bullet_enyo_output_buckets),
                "--eval-scale", str(args.bullet_eval_scale),
                "--save-rate", str(args.bullet_save_rate),
                *(
                    ["--init-weights", templated_path_arg(args.bullet_init_weights)]
                    if args.bullet_init_weights else []
                ),
                "--trainable", args.bullet_trainable,
                "--weight-decay", str(args.bullet_weight_decay),
                *(["--export-init-only"] if args.bullet_export_init_only else []),
            ],
        })
        append_provenance_step(
            steps,
            args=args,
            python=python,
            net=f"{candidate_dir}/model.nn",
        )
        if args.bullet_static_data:
            steps.append({
                "name": "validate_bullet_static",
                "command": [
                    python, tool("validate/eval_dataset.py"),
                    "--net", f"{candidate_dir}/model.nn",
                    "--data", templated_path_arg(args.bullet_static_data),
                    "--rows", str(args.bullet_static_rows),
                    "--device", args.device,
                    "--buckets",
                    "--sources",
                ],
            })
    else:
        raise SystemExit(f"unsupported backend={args.backend}")

    if args.engine_static_jsonl and not engine_static_done:
        append_engine_static_step(
            steps,
            args,
            name="validate_engine_static",
            net=f"{candidate_dir}/model.nn",
            jsonl=args.engine_static_jsonl,
        )

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

    def append_default(key: str, fallback: object) -> list[object]:
        current = value(key, fallback)
        if current is None:
            return []
        if isinstance(current, str):
            return [current]
        return list(current)

    parser.add_argument("-c", "--config", default=value("config", None),
                        help="JSON create-argument config. CLI args override it.")
    parser.add_argument("--name", default=value("name", None))
    parser.add_argument("--run-dir", default=value("run_dir", None))
    parser.add_argument("--dry-run", action="store_true",
                        default=value("dry_run", False))
    parser.add_argument("--force", action="store_true",
                        default=value("force", False))
    parser.add_argument("--disabled", action=argparse.BooleanOptionalAction,
                        default=value("disabled", False),
                        help="Make this create config fail safely instead of launching a run.")
    parser.add_argument("--disabled-reason",
                        default=value("disabled_reason", "no candidate build is selected"))
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
                        choices=["pytorch", "bullet", "pairwise"])

    parser.add_argument("--selfplay-games", type=int, default=value("selfplay_games", d.selfplay_games))
    parser.add_argument("--selfplay-shard-games", type=int, default=value("selfplay_shard_games", d.selfplay_shard_games))
    parser.add_argument("--selfplay-concurrency", type=int, default=value("selfplay_concurrency", d.selfplay_concurrency))
    parser.add_argument("--selfplay-threads", type=int, default=value("selfplay_threads", d.selfplay_threads))
    parser.add_argument("--selfplay-hash", type=int, default=value("selfplay_hash", d.selfplay_hash))
    parser.add_argument("--selfplay-use-nnue", action=argparse.BooleanOptionalAction,
                        default=value("selfplay_use_nnue", d.selfplay_use_nnue))
    parser.add_argument("--selfplay-depth", type=int, default=value("selfplay_depth", d.selfplay_depth))
    parser.add_argument("--selfplay-seed", type=int, default=value("selfplay_seed", d.selfplay_seed))
    parser.add_argument("--selfplay-crucible", action=argparse.BooleanOptionalAction,
                        default=value("selfplay_crucible", d.selfplay_crucible))
    parser.add_argument("--selfplay-crucible-tool",
                        default=value("selfplay_crucible_tool", d.selfplay_crucible_tool))
    parser.add_argument("--selfplay-crucible-python",
                        default=value("selfplay_crucible_python", d.selfplay_crucible_python))
    parser.add_argument("--selfplay-crucible-local-slots", type=int,
                        default=value("selfplay_crucible_local_slots", d.selfplay_crucible_local_slots))
    parser.add_argument("--selfplay-crucible-lease-seconds", type=int,
                        default=value("selfplay_crucible_lease_seconds", d.selfplay_crucible_lease_seconds))
    parser.add_argument("--selfplay-crucible-path-map", action="append",
                        default=append_default("selfplay_crucible_path_map", d.selfplay_crucible_path_map))
    parser.add_argument("--selfplay-crucible-require-notify",
                        action=argparse.BooleanOptionalAction,
                        default=value("selfplay_crucible_require_notify", d.selfplay_crucible_require_notify))
    parser.add_argument("--selfplay-crucible-notify-command",
                        default=value("selfplay_crucible_notify_command", d.selfplay_crucible_notify_command))
    parser.add_argument("--selfplay-crucible-workers",
                        default=value("selfplay_crucible_workers", d.selfplay_crucible_workers))
    parser.add_argument("--selfplay-crucible-jobs", type=int,
                        default=value("selfplay_crucible_jobs", d.selfplay_crucible_jobs))
    parser.add_argument("--selfplay-crucible-coordinator-host",
                        default=value("selfplay_crucible_coordinator_host", d.selfplay_crucible_coordinator_host))
    parser.add_argument("--selfplay-crucible-remote-timeout-seconds", type=int,
                        default=value("selfplay_crucible_remote_timeout_seconds", d.selfplay_crucible_remote_timeout_seconds))
    parser.add_argument("--selfplay-crucible-verbose",
                        action=argparse.BooleanOptionalAction,
                        default=value("selfplay_crucible_verbose", d.selfplay_crucible_verbose))

    parser.add_argument("--skip-plies", type=int, default=value("skip_plies", d.skip_plies))
    parser.add_argument("--source-max-abs-cp", type=int, default=value("source_max_abs_cp", d.source_max_abs_cp))
    parser.add_argument("--sample-preset", default=value("sample_preset", d.sample_preset))

    parser.add_argument("--score-engine", default=value("score_engine", d.score_engine))
    parser.add_argument("--score-depth", type=int, default=value("score_depth", d.score_depth))
    parser.add_argument("--score-shards", type=int, default=value("score_shards", d.score_shards))
    parser.add_argument("--score-threads", type=int, default=value("score_threads", d.score_threads))
    parser.add_argument("--score-hash", type=int, default=value("score_hash", d.score_hash))
    parser.add_argument("--score-limit", type=int, default=value("score_limit", d.score_limit))
    parser.add_argument("--score-source-jsonl", default=value("score_source_jsonl", d.score_source_jsonl))
    parser.add_argument("--labeled-jsonl", default=value("labeled_jsonl", d.labeled_jsonl),
                        help="Pre-scored JSONL for backend=pytorch; skips source generation and scoring.")
    parser.add_argument("--score-backend",
                        choices=["uci", "stockfish-static"],
                        default=value("score_backend", d.score_backend))
    parser.add_argument("--score-static-datagen-tool",
                        default=value("score_static_datagen_tool", d.score_static_datagen_tool))
    parser.add_argument("--score-stockfish-net",
                        default=value("score_stockfish_net", d.score_stockfish_net))
    parser.add_argument("--score-enyo-net",
                        default=value("score_enyo_net", d.score_enyo_net))
    parser.add_argument("--score-min-delta-cp", type=int,
                        default=value("score_min_delta_cp", d.score_min_delta_cp))
    parser.add_argument("--score-max-abs-cp", type=int, default=value("score_max_abs_cp", d.score_max_abs_cp))
    parser.add_argument("--score-progress", type=int, default=value("score_progress", d.score_progress))
    parser.add_argument("--score-output-format",
                        choices=["jsonl", "packed", "bullet-text", "bullet-data"],
                        default=value("score_output_format", d.score_output_format))
    parser.add_argument("--score-crucible", action=argparse.BooleanOptionalAction,
                        default=value("score_crucible", d.score_crucible))
    parser.add_argument("--score-crucible-tool",
                        default=value("score_crucible_tool", d.score_crucible_tool))
    parser.add_argument("--score-crucible-python",
                        default=value("score_crucible_python", d.score_crucible_python))
    parser.add_argument("--score-crucible-local-slots", type=int,
                        default=value("score_crucible_local_slots", d.score_crucible_local_slots))
    parser.add_argument("--score-crucible-lease-seconds", type=int,
                        default=value("score_crucible_lease_seconds", d.score_crucible_lease_seconds))
    parser.add_argument("--score-crucible-path-map", action="append",
                        default=append_default("score_crucible_path_map", d.score_crucible_path_map))
    parser.add_argument("--score-crucible-require-notify",
                        action=argparse.BooleanOptionalAction,
                        default=value("score_crucible_require_notify", d.score_crucible_require_notify))
    parser.add_argument("--score-crucible-notify-command",
                        default=value("score_crucible_notify_command", d.score_crucible_notify_command))
    parser.add_argument("--score-crucible-workers",
                        default=value("score_crucible_workers", d.score_crucible_workers))
    parser.add_argument("--score-crucible-jobs", type=int,
                        default=value("score_crucible_jobs", d.score_crucible_jobs))
    parser.add_argument("--score-crucible-coordinator-host",
                        default=value("score_crucible_coordinator_host", d.score_crucible_coordinator_host))
    parser.add_argument("--score-crucible-remote-timeout-seconds", type=int,
                        default=value("score_crucible_remote_timeout_seconds", d.score_crucible_remote_timeout_seconds))
    parser.add_argument("--score-crucible-verbose",
                        action=argparse.BooleanOptionalAction,
                        default=value("score_crucible_verbose", d.score_crucible_verbose))

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
    parser.add_argument("--output-head-features",
                        default=value("output_head_features", d.output_head_features),
                        choices=["none", "material-phase"])

    parser.add_argument("--source-mix-jsonl", action="append",
                        default=append_default("source_mix_jsonl", d.source_mix_jsonl),
                        help="JSONL source mix spec PATH:ROWS. Repeatable.")
    parser.add_argument("--source-mix-seed", type=int,
                        default=value("source_mix_seed", d.source_mix_seed))
    parser.add_argument("--source-mix-progress", type=int,
                        default=value("source_mix_progress", d.source_mix_progress))
    parser.add_argument("--bullet-source-jsonl", default=value("bullet_source_jsonl", d.bullet_source_jsonl))
    parser.add_argument("--bullet-generate-source", action=argparse.BooleanOptionalAction,
                        default=value("bullet_generate_source", d.bullet_generate_source))
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
    parser.add_argument("--bullet-enyo-feature-channels", type=int,
                        default=value("bullet_enyo_feature_channels", d.bullet_enyo_feature_channels),
                        choices=[11, 12])
    parser.add_argument("--bullet-enyo-runtime-input-buckets", type=int,
                        default=value("bullet_enyo_runtime_input_buckets", d.bullet_enyo_runtime_input_buckets),
                        choices=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--bullet-enyo-output-buckets", type=int,
                        default=value("bullet_enyo_output_buckets", d.bullet_enyo_output_buckets),
                        choices=[1, 2, 4, 8])
    parser.add_argument("--bullet-eval-scale", type=float, default=value("bullet_eval_scale", d.bullet_eval_scale))
    parser.add_argument("--bullet-save-rate", type=int, default=value("bullet_save_rate", d.bullet_save_rate))
    parser.add_argument("--bullet-init-weights", default=value("bullet_init_weights", d.bullet_init_weights))
    parser.add_argument("--bullet-init-net", default=value("bullet_init_net", d.bullet_init_net))
    parser.add_argument("--bullet-trainable", default=value("bullet_trainable", d.bullet_trainable),
                        choices=["all", "input", "float-head", "output"])
    parser.add_argument("--bullet-weight-decay", type=float, default=value("bullet_weight_decay", d.bullet_weight_decay))
    parser.add_argument("--bullet-export-init-only", action="store_true",
                        default=value("bullet_export_init_only", d.bullet_export_init_only))
    parser.add_argument("--bullet-static-data", default=value("bullet_static_data", d.bullet_static_data))
    parser.add_argument("--bullet-static-rows", type=int, default=value("bullet_static_rows", d.bullet_static_rows))
    parser.add_argument("--engine-static-jsonl", default=value("engine_static_jsonl", d.engine_static_jsonl))
    parser.add_argument("--engine-static-rows", type=int, default=value("engine_static_rows", d.engine_static_rows))
    parser.add_argument("--engine-static-engine", default=value("engine_static_engine", d.engine_static_engine))
    parser.add_argument("--validate-provenance", action=argparse.BooleanOptionalAction,
                        default=value("validate_provenance", d.validate_provenance),
                        help="Run net_provenance.py without requiring Enyo-only position sources.")
    parser.add_argument("--require-clean-enyo-owned", action=argparse.BooleanOptionalAction,
                        default=value("require_clean_enyo_owned", d.require_clean_enyo_owned))
    parser.add_argument("--pairwise-data", default=value("pairwise_data", d.pairwise_data))
    parser.add_argument("--pairwise-pairs-jsonl", default=value("pairwise_pairs_jsonl", d.pairwise_pairs_jsonl))
    parser.add_argument("--pairwise-init-from-nn", default=value("pairwise_init_from_nn", d.pairwise_init_from_nn))
    parser.add_argument("--pairwise-pair-batch-size", type=int,
                        default=value("pairwise_pair_batch_size", d.pairwise_pair_batch_size))
    parser.add_argument("--pairwise-pair-weight", type=float,
                        default=value("pairwise_pair_weight", d.pairwise_pair_weight))
    parser.add_argument("--pairwise-broad-weight", type=float,
                        default=value("pairwise_broad_weight", d.pairwise_broad_weight))
    parser.add_argument("--pairwise-pair-beta", type=float,
                        default=value("pairwise_pair_beta", d.pairwise_pair_beta))
    parser.add_argument("--pairwise-max-target-margin", type=float,
                        default=value("pairwise_max_target_margin", d.pairwise_max_target_margin))
    parser.add_argument("--pairwise-min-target-margin", type=float,
                        default=value("pairwise_min_target_margin", d.pairwise_min_target_margin))
    parser.add_argument("--pairwise-loss-weight-by-cp",
                        action=argparse.BooleanOptionalAction,
                        default=value("pairwise_loss_weight_by_cp", d.pairwise_loss_weight_by_cp))
    parser.add_argument("--pairwise-steps-per-epoch", type=int,
                        default=value("pairwise_steps_per_epoch", d.pairwise_steps_per_epoch))
    parser.add_argument("--pairwise-max-rows", type=int,
                        default=value("pairwise_max_rows", d.pairwise_max_rows))
    parser.add_argument("--pairwise-skip-rows", type=int,
                        default=value("pairwise_skip_rows", d.pairwise_skip_rows))
    parser.add_argument("--pairwise-checkpoint-every", type=int,
                        default=value("pairwise_checkpoint_every", d.pairwise_checkpoint_every))
    parser.add_argument("--pairwise-move-gate-cases",
                        default=value("pairwise_move_gate_cases", d.pairwise_move_gate_cases))
    parser.add_argument("--pairwise-move-gate-baseline-net",
                        default=value("pairwise_move_gate_baseline_net", d.pairwise_move_gate_baseline_net))
    parser.add_argument("--pairwise-move-gate-limit", type=int,
                        default=value("pairwise_move_gate_limit", d.pairwise_move_gate_limit))
    parser.add_argument("--pairwise-move-gate-fail-candidate-below-baseline",
                        action=argparse.BooleanOptionalAction,
                        default=value(
                            "pairwise_move_gate_fail_candidate_below_baseline",
                            d.pairwise_move_gate_fail_candidate_below_baseline))
    parser.add_argument("--pairwise-move-gate-fail-regressed-above", type=int,
                        default=value(
                            "pairwise_move_gate_fail_regressed_above",
                            d.pairwise_move_gate_fail_regressed_above))
    parser.add_argument("--pairwise-move-gate-fail-fixed-below", type=int,
                        default=value(
                            "pairwise_move_gate_fail_fixed_below",
                            d.pairwise_move_gate_fail_fixed_below))
    parser.add_argument("--pairwise-move-gate-fail-delta-below", type=float,
                        default=value(
                            "pairwise_move_gate_fail_delta_below",
                            d.pairwise_move_gate_fail_delta_below))
    parser.add_argument("--pairwise-move-gate-fail-loss-weighted-delta-below", type=float,
                        default=value(
                            "pairwise_move_gate_fail_loss_weighted_delta_below",
                            d.pairwise_move_gate_fail_loss_weighted_delta_below))


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
