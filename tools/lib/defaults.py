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
    runner: str = "~/.local/bin/fastchess"
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
    selfplay_use_nnue: bool = True
    selfplay_depth: int = 8
    selfplay_seed: int = 2026051501
    selfplay_crucible: bool = False
    selfplay_crucible_tool: str = "~/.local/bin/crucible"
    selfplay_crucible_python: str = ""
    selfplay_crucible_local_slots: int = 1
    selfplay_crucible_lease_seconds: int = 600
    selfplay_crucible_path_map: tuple[str, ...] = ()
    selfplay_crucible_require_notify: bool = True
    selfplay_crucible_notify_command: str = "/home/petter/scripts/notifai.sh"
    selfplay_crucible_workers: str = ""
    selfplay_crucible_jobs: int = 4
    selfplay_crucible_remote_timeout_seconds: int = 1800
    selfplay_crucible_verbose: bool = False

    skip_plies: int = 8
    source_max_abs_cp: int = 1600
    sample_preset: str = "signed-balanced-v1"

    score_depth: int = 16
    score_shards: int = 24
    score_threads: int = 1
    score_hash: int = 128
    score_limit: int = 0
    score_source_jsonl: str = ""
    score_max_abs_cp: int = 1600
    score_progress: int = 10000
    score_crucible: bool = False
    score_crucible_python: str = ""
    score_crucible_tool: str = "~/.local/bin/crucible"
    score_crucible_local_slots: int = 1
    score_crucible_lease_seconds: int = 600
    score_crucible_path_map: tuple[str, ...] = ()
    score_crucible_require_notify: bool = True
    score_crucible_notify_command: str = "/home/petter/scripts/notifai.sh"
    score_crucible_workers: str = ""
    score_crucible_jobs: int = 4
    score_crucible_remote_timeout_seconds: int = 1800
    score_crucible_verbose: bool = False

    max_features: int = 32
    pack_progress: int = 250000

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
    val_rows: int = 100000
    patience: int = 2
    select_metric: str = "mae"
    weight_decay: float = 1e-6
    trainable: str = "all"
    backend: str = "pytorch"

    bullet_source_jsonl: str = "runs/imported/fresh_d12self18h64_d16_labels_20260519_113826/score/labeled.jsonl"
    source_mix_jsonl: tuple[str, ...] = ()
    source_mix_seed: int = 1
    source_mix_progress: int = 250000
    bullet_generate_source: bool = False
    bullet_data: str = ""
    bullet_manifest: str = "~/source/bullet/Cargo.toml"
    bullet_loader: str = "direct"
    bullet_limit: int = 100000
    bullet_max_abs_cp: int = 1600
    bullet_enyo_runtime_target: bool = True
    bullet_sfbinpack_buffer_mb: int = 1024
    bullet_sfbinpack_min_ply: int = 16
    bullet_sfbinpack_max_abs_cp: int = 1600
    bullet_sfbinpack_quiet_only: bool = True
    bullet_mode: str = "enyo"
    bullet_accelerator: str = "cuda"
    bullet_cuda_path: str = ""
    bullet_cuda_arch: str = "auto"
    bullet_hidden: int = 1024
    bullet_l2: int = 16
    bullet_batch_size: int = 4096
    bullet_batches: int = 64
    bullet_superbatches: int = 2048
    bullet_threads: int = 4
    bullet_wdl: float = 0.75
    bullet_lr: float = 1e-3
    bullet_final_lr: float = 3e-4
    bullet_enyo_l0_std: float = 8.0
    bullet_enyo_l1_std: float = 1.0
    bullet_enyo_l1_export_scale: float = 1.0
    bullet_enyo_input_factorizer: bool = False
    bullet_enyo_input_buckets: int = 16
    bullet_enyo_runtime_input_buckets: int = 16
    bullet_enyo_output_buckets: int = 1
    bullet_eval_scale: float = 400.0
    bullet_save_rate: int = 64
    bullet_init_weights: str = ""
    bullet_trainable: str = "all"
    bullet_weight_decay: float = 0.0
    bullet_export_init_only: bool = False
    bullet_static_data: str = "runs/imported/fresh_d12self18h64_d16_labels_20260519_113826/pack/train"
    bullet_static_rows: int = 100000
    engine_static_jsonl: str = ""
    engine_static_rows: int = 1000
    engine_static_engine: str = "~/code/cpp/chess/assets/engines/reference"
    require_clean_enyo_owned: bool = False

    sprt_games: int = 1000
    sprt_tc: str = "2+0.02"
    sprt_concurrency: int = 10
    sprt_threads: int = 2
    sprt_hash: int = 512
    sprt_elo0: int = 0
    sprt_elo1: int = 8


DEFAULTS = CandidateDefaults()
