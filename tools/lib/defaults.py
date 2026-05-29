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
    nnue_file: str = "~/code/cpp/chess/enyo/net/berserk-d43206fe90e4.nn"
    book: str = "~/code/cpp/chess/assets/books/UHO_Lichess_4852_v1.epd"
    runner: str = "~/local/bin/fastchess"
    score_engine: str = default_executable("~/local/bin/stockfish", "stockfish")
    python: str = default_executable("~/.venv/bin/python", "python3")
    init_net: str = "~/code/cpp/chess/enyo/net/berserk-d43206fe90e4.nn"
    reference_net: str = "~/code/cpp/chess/enyo/net/berserk-d43206fe90e4.nn"
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
    child_static_gate_rows: int = 100000
    child_static_gate_max_mae_regression_cp: float = 5.0
    child_static_gate_max_sign_drop_pct: float = 1.0
    child_static_gate_max_near_zero_sign_drop_pct: float = 2.0
    child_batch_size: int = 64
    child_loss: str = "pairwise"
    ranking_weight: float = 1.0
    broad_preserve_weight: float = 0.1
    broad_anchor: str = "label"
    broad_deadzone_cp: int = 40
    broad_beta: int = 100
    child_preserve_targets: str = ""
    child_preserve_weight: float = 0.0
    child_preserve_deadzone_cp: int = 5
    child_preserve_beta: int = 100
    child_preserve_batch_size: int = 64
    child_preserve_min_groups: int = 1
    rank_margin_cp: int = 100
    rank_temperature_cp: int = 50
    min_groups: int = 1
    min_pairs: int = 1
    child_model_gate_min_top1: int = 1
    child_engine_gate_min_top1: int = 1
    child_engine_jobs: int = 4
    child_search_gate_targets: str = ""
    child_search_gate_reference_net: str = ""
    child_search_gate_nodes: int = 20000
    child_search_gate_depth: int = 0
    child_search_gate_jobs: int = 8
    child_search_gate_min_groups: int = 1
    child_search_gate_min_top1: int = -1
    child_search_gate_max_missing: int = 0
    child_search_gate_max_reference_better: int = -1
    child_search_gate_min_sum_diff: float = -1.0e18
    child_augment_input: str = ""
    child_augment_output: str = "targets/child-ranking/augmented.jsonl"
    child_augment_summary: str = "targets/child-ranking/augmented.summary.txt"
    child_augment_search_nets: str = ""
    child_augment_search_nodes: int = 20000
    child_augment_search_depth: int = 0
    child_augment_jobs: int = 8
    child_augment_min_groups: int = 1
    child_augment_max_missing_after: int = 0
    child_augment_oracle_nodes: int = 200000
    child_augment_oracle_depth: int = 0
    child_augment_oracle_threads: int = 1
    child_augment_oracle_hash: int = 128
    child_base_input: str = ""
    child_base_output: str = "targets/child-ranking/base_move.jsonl"
    child_base_summary: str = "targets/child-ranking/base_move.summary.txt"
    child_base_origin: str = ""
    child_base_min_groups: int = 1
    child_base_keep_missing: bool = False
    search_descendant_input: str = ""
    search_descendant_output: str = "targets/search-descendant/child_targets.jsonl"
    search_descendant_summary: str = "targets/search-descendant/summary.txt"
    search_descendant_candidate_net: str = ""
    search_descendant_reference_net: str = ""
    search_descendant_root_nodes: int = 20000
    search_descendant_root_depth: int = 0
    search_descendant_nodes: int = 10000
    search_descendant_depth: int = 0
    search_descendant_jobs: int = 4
    search_descendant_min_ply: int = 1
    search_descendant_max_ply: int = 4
    search_descendant_max_gap_cp: float = 800.0
    search_descendant_min_oracle_gap_cp: float = 20.0
    search_descendant_min_input_groups: int = 1
    search_descendant_min_output_groups: int = 1
    search_descendant_oracle_nodes: int = 200000
    search_descendant_oracle_depth: int = 0
    search_descendant_oracle_threads: int = 1
    search_descendant_oracle_hash: int = 128
    mix_sources: str = ""
    mix_output: str = "targets/mixed/mixed.jsonl"
    mix_summary: str = "targets/mixed/mix.summary.json"
    mix_seed: int = 1
    mix_progress: int = 250000
    mix_min_rows: int = 1
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
    policy_base_best_preserve_weight: float = 0.0
    policy_no_harm_weight: float = 0.0
    policy_no_harm_gap_cp: float = 10.0
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

    lc0_input: str = "~/code/cpp/chess/assets/lc0/raw/training-run1--20210605-0516.tar"
    lc0_output: str = "targets/lc0-smoke-20260528/lc0_positions.jsonl"
    lc0_summary: str = "targets/lc0-smoke-20260528/lc0_positions.summary.txt"
    lc0_max_records: int = 1000
    lc0_top_policy: int = 8
    lc0_min_rows: int = 1
    lc0_min_played_legal_pct: float = 99.0
    lc0_min_best_legal_pct: float = 99.0
    lc0_child_targets: str = "targets/lc0-smoke-20260528/lc0_child_targets.jsonl"
    lc0_child_summary: str = "targets/lc0-smoke-20260528/lc0_child_targets.summary.txt"
    lc0_child_min_groups: int = 1
    lc0_child_max_groups: int = 0
    lc0_child_unique_fen: bool = True
    lc0_best_source: str = "top-policy"
    lc0_policy_score_scale_cp: float = 50.0
    lc0_policy_floor: float = 1e-4
    lc0_child_max_gap_cp: float = 300.0
    lc0_min_best_policy: float = 0.0
    lc0_min_policy_gap_cp: float = 0.0
    lc0_oracle_child_targets: str = "targets/lc0-oracle-smoke-20260528/lc0_oracle_child_targets.jsonl"
    lc0_oracle_child_summary: str = "targets/lc0-oracle-smoke-20260528/lc0_oracle_child_targets.summary.txt"
    lc0_oracle_min_groups: int = 1
    lc0_oracle_max_groups: int = 0
    lc0_oracle_nodes: int = 200000
    lc0_oracle_depth: int = 12
    lc0_oracle_jobs: int = 8
    lc0_oracle_threads: int = 1
    lc0_oracle_hash: int = 128
    lc0_oracle_max_moves_per_position: int = 8
    lc0_oracle_max_gap_cp: float = 800.0
    lc0_oracle_min_gap_cp: float = 0.0
    lc0_oracle_preselect_multiplier: int = 3

    smoke_pgn: str = ""
    smoke_child_targets: str = "targets/smoke-pgn-20260528/smoke_child_targets.jsonl"
    smoke_child_summary: str = "targets/smoke-pgn-20260528/smoke_child_targets.summary.txt"
    smoke_candidate_net: str = ""
    smoke_reference_net: str = "~/code/cpp/chess/enyo/net/berserk-d43206fe90e4.nn"
    smoke_search_nodes: int = 20000
    smoke_search_depth: int = 0
    smoke_oracle_nodes: int = 100000
    smoke_oracle_depth: int = 12
    smoke_jobs: int = 8
    smoke_threads: int = 1
    smoke_hash: int = 64
    smoke_oracle_threads: int = 1
    smoke_oracle_hash: int = 128
    smoke_min_ply: int = 8
    smoke_max_per_game: int = 8
    smoke_sample_mode: str = "first"
    smoke_max_positions: int = 1000
    smoke_max_gap_cp: float = 800.0
    smoke_min_oracle_gap_cp: float = 20.0
    smoke_min_groups: int = 1
    smoke_unique_fen: bool = True
    smoke_only_candidate_losses: bool = False
    smoke_only_candidate_worse: bool = False
    smoke_candidate_name: str = "candidate"
    smoke_reference_name: str = "reference"

    sprt_games: int = 1000
    sprt_tc: str = "2+0.02"
    sprt_concurrency: int = 10
    sprt_threads: int = 2
    sprt_hash: int = 512
    sprt_elo0: int = 0
    sprt_elo1: int = 8


DEFAULTS = CandidateDefaults()
