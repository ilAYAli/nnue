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

    sprt_games: int = 1000
    sprt_tc: str = "2+0.02"
    sprt_concurrency: int = 10
    sprt_threads: int = 2
    sprt_hash: int = 512
    sprt_elo0: int = 0
    sprt_elo1: int = 8


DEFAULTS = CandidateDefaults()
