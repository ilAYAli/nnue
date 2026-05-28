"""Shared defaults for Enyo NNUE tooling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from pathlib import Path
import shutil


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def chess_root() -> Path:
    return repo_root().parent


def default_executable(preferred: str, fallback: str) -> str:
    path = Path(preferred).expanduser()
    if path.exists():
        return preferred
    resolved = shutil.which(fallback)
    if resolved:
        return resolved
    return preferred


@dataclass(frozen=True)
class CandidateDefaults:
    engine: str = "~/code/cpp/chess/assets/engines/reference"
    nnue_file: str = "~/code/cpp/chess/enyo/nnue/berserk-d43206fe90e4.nn"
    book: str = "~/code/cpp/chess/assets/books/UHO_Lichess_4852_v1.epd"
    runner: str = "~/local/bin/fastchess"
    score_engine: str = default_executable("~/local/bin/stockfish", "stockfish")
    python: str = default_executable("~/.venv/bin/python", "python3")
    init_net: str = "~/code/cpp/chess/enyo/nnue/berserk-d43206fe90e4.nn"
    reference_net: str = "~/code/cpp/chess/enyo/nnue/berserk-d43206fe90e4.nn"
    reference_engine: str = "~/code/cpp/chess/assets/engines/reference"
    sprt: str = "~/code/cpp/chess/sprt/sprt"
    run_base: str = "~/code/cpp/chess/nnue/runs"

    selfplay_games: int = 160000
    selfplay_shard_games: int = 1000
    selfplay_concurrency: int = 12
    selfplay_threads: int = 1
    selfplay_hash: int = 128
    selfplay_depth: int = 8
    selfplay_seed: int = 2026051501

    skip_plies: int = 8
    source_max_abs_cp: int = 1600
    sample_preset: str = "signed-balanced-v1"

    score_depth: int = 16
    score_shards: int = 24
    score_threads: int = 1
    score_hash: int = 128
    score_max_abs_cp: int = 1600
    score_progress: int = 10000

    max_features: int = 32
    pack_progress: int = 250000
    pack_dir: str = ""

    backend: str = "pytorch"
    objective: str = "huber"
    target_clamp: int = 800
    huber_beta: int = 200
    wdl_lambda: float = 0.75
    lr: float = 7e-7
    epochs: int = 8
    batch_size: int = 8192
    device: str = "cuda"
    workers: int = 4
    prefetch_factor: int = 2
    amp: Literal["off", "bf16"] = "bf16"
    torch_compile: bool = True
    dataset_in_memory: bool = True
    export_quantize_forward: bool = False
    val_rows: int = 100000
    patience: int = 2
    select_metric: str = "mae"
    weight_decay: float = 1e-6
    trainable: str = "all"

    child_targets: str = ""
    child_broad_rows: int = 100000
    child_batch_size: int = 64
    child_loss: str = "pairwise"
    ranking_weight: float = 1.0
    broad_preserve_weight: float = 0.1
    broad_anchor: str = "label"
    broad_deadzone_cp: int = 40
    broad_beta: int = 100
    rank_margin_cp: int = 100
    rank_temperature_cp: int = 50
    min_groups: int = 1
    min_pairs: int = 1
    child_model_gate_min_top1: int = 1
    child_engine_gate_min_top1: int = 1
    child_engine_jobs: int = 4
    policy_targets: str = ""
    policy_hidden: int = 128
    policy_feature_set: str = "compact"
    policy_dropout: float = 0.0
    policy_include_tags: str = ""
    policy_exclude_tags: str = ""
    policy_preserve_include_tags: str = ""
    policy_preserve_exclude_tags: str = ""
    policy_preserve_weight: float = 0.0
    policy_preserve_margin: float = 4.0
    policy_preserve_max_groups: int = 0
    policy_preserve_val_fraction: float = -1.0
    policy_gate_include_tags: str = ""
    policy_gate_exclude_tags: str = ""
    policy_broad_gate_include_tags: str = ""
    policy_broad_gate_exclude_tags: str = ""
    policy_broad_gate_min_groups: int = 1
    policy_broad_gate_max_bad: int = -1
    policy_broad_gate_max_overrides: int = -1
    policy_breakdown_tags: str = ""
    policy_val_fraction: float = 0.2
    policy_target_temperature_cp: int = 80
    policy_thresholds: str = "0,1,2,4,8"
    policy_gate_min_top1: int = 1
    policy_gate_max_bad: int = -1
    policy_gate_min_val_top1: int = -1
    policy_gate_max_val_bad: int = -1
    policy_gate_min_val_good: int = -1
    policy_gate_min_val_overrides: int = -1
    policy_bad_tolerance_cp: float = 0.0
    policy_export_threshold: float = 4.0
    policy_export_max_abs_diff: float = 1e-3

    replay: str = default_executable("~/local/bin/replay", "replay")
    replay_logs: str = "~/code/cpp/chess/lichess/logs/loss"
    replay_candidate: str = "~/code/cpp/chess/assets/engines/reference"
    replay_reference: str = ""
    replay_oracle_nodes: int = 200000
    replay_jobs: int = 8
    replay_move: int = 8
    replay_top_root_moves: int = 8
    replay_include_checks: bool = True
    replay_include_captures: bool = True
    replay_include_promotions: bool = True
    replay_include_history_sensitive: bool = False
    replay_max_moves_per_position: int = 16
    replay_min_score_gap: int = 0
    replay_output: str = "targets/replay-loss-20260528/loss_replay.jsonl"
    replay_stderr: str = "targets/replay-loss-20260528/loss_replay.stderr"
    replay_min_rows: int = 1
    replay_child_targets: str = "targets/replay-loss-20260528/loss_replay_child_targets.jsonl"
    replay_child_summary: str = "targets/replay-loss-20260528/loss_replay_child_targets.summary.txt"
    replay_child_min_groups: int = 1

    sprt_games: int = 1000
    sprt_tc: str = "2+0.02"
    sprt_concurrency: int = 10
    sprt_threads: int = 2
    sprt_hash: int = 512
    sprt_elo0: int = 0
    sprt_elo1: int = 8


DEFAULTS = CandidateDefaults()
