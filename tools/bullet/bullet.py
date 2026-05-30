#!/usr/bin/env python3
"""Bullet helper commands for Enyo NNUE runs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ENYO_DEFAULT_INPUT_BUCKETS = 16
ENYO_SUPPORTED_INPUT_BUCKETS = (1, 2, 4, 8, 16, 32)
ENYO_PIECE_TYPES = 12
ENYO_SQUARES = 64
ENYO_HIDDEN = 1024
ENYO_L2 = 16
ENYO_L3 = 32


def enyo_network_size(
    input_buckets: int = ENYO_DEFAULT_INPUT_BUCKETS,
    *,
    hidden: int = ENYO_HIDDEN,
    l2: int = ENYO_L2,
    l3: int = ENYO_L3,
) -> int:
    features = input_buckets * ENYO_PIECE_TYPES * ENYO_SQUARES
    l1 = 2 * hidden
    return (
        features * hidden * 2
        + hidden * 2
        + l1 * l2
        + l2 * 4
        + l2 * l3 * 4
        + l3 * 4
        + l3 * 4
        + 4
    )


ENYO_NETWORK_SIZE = enyo_network_size()

ENYO_FEATURE_STRIDE = ENYO_PIECE_TYPES * ENYO_SQUARES
ENYO_LEGACY_BUCKET_FOR_32 = (
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 8, 9, 10, 11,
    12, 12, 13, 13, 12, 12, 13, 13,
    14, 14, 15, 15, 14, 14, 15, 15,
)


def trim_bullet_checkpoint(
    raw: bytes,
    input_buckets: int,
    *,
    hidden: int = ENYO_HIDDEN,
    l2: int = ENYO_L2,
    l3: int = ENYO_L3,
) -> bytes:
    network_size = enyo_network_size(input_buckets, hidden=hidden, l2=l2, l3=l3)
    if len(raw) < network_size:
        raise SystemExit(f"checkpoint is {len(raw)} bytes, expected at least {network_size}")
    if len(raw) == network_size:
        return raw
    trailer = raw[network_size:]
    expected = (b"bullet" * ((len(trailer) + 5) // 6))[:len(trailer)]
    if trailer != expected:
        raise SystemExit(f"checkpoint has unexpected {len(trailer)} byte trailer")
    return raw[:network_size]


def enyo_parent_bucket(
    target_bucket: int,
    input_buckets: int,
    *,
    runtime_input_buckets: int = ENYO_DEFAULT_INPUT_BUCKETS,
) -> int:
    if input_buckets == runtime_input_buckets:
        return target_bucket
    if (
        input_buckets not in ENYO_SUPPORTED_INPUT_BUCKETS
        or runtime_input_buckets not in ENYO_SUPPORTED_INPUT_BUCKETS
    ):
        raise SystemExit(
            "unsupported Enyo input bucket count: "
            f"train={input_buckets} runtime={runtime_input_buckets}"
        )
    if input_buckets > runtime_input_buckets:
        raise SystemExit(
            "cannot export fewer runtime buckets than trained buckets: "
            f"train={input_buckets} runtime={runtime_input_buckets}"
        )
    if runtime_input_buckets == 32:
        legacy_bucket = ENYO_LEGACY_BUCKET_FOR_32[target_bucket]
        return legacy_bucket * input_buckets // 16
    return target_bucket * input_buckets // runtime_input_buckets


def expand_enyo_input_buckets(
    raw: bytes,
    input_buckets: int,
    *,
    runtime_input_buckets: int = ENYO_DEFAULT_INPUT_BUCKETS,
    hidden: int = ENYO_HIDDEN,
    l2: int = ENYO_L2,
    l3: int = ENYO_L3,
) -> bytes:
    raw = trim_bullet_checkpoint(raw, input_buckets, hidden=hidden, l2=l2, l3=l3)
    if input_buckets == runtime_input_buckets:
        return raw

    source_features = input_buckets * ENYO_FEATURE_STRIDE
    target_features = runtime_input_buckets * ENYO_FEATURE_STRIDE
    feature_bytes = hidden * 2
    source_l0_bytes = source_features * feature_bytes
    target_l0_bytes = target_features * feature_bytes
    rest = raw[source_l0_bytes:]
    expanded = bytearray(target_l0_bytes + len(rest))

    for target_bucket in range(runtime_input_buckets):
        source_bucket = enyo_parent_bucket(
            target_bucket,
            input_buckets,
            runtime_input_buckets=runtime_input_buckets,
        )
        for offset in range(ENYO_FEATURE_STRIDE):
            source_feature = source_bucket * ENYO_FEATURE_STRIDE + offset
            target_feature = target_bucket * ENYO_FEATURE_STRIDE + offset
            source_start = source_feature * feature_bytes
            target_start = target_feature * feature_bytes
            expanded[target_start:target_start + feature_bytes] = raw[
                source_start:source_start + feature_bytes
            ]

    expanded[target_l0_bytes:] = rest
    return bytes(expanded)


def expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser().resolve()


def expand_data_paths(value: str) -> str:
    return ";".join(
        str(expand_path(path))
        for path in str(value).split(";")
        if path.strip()
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cargo_bin() -> str:
    cargo = shutil.which("cargo")
    if cargo:
        return cargo
    fallback = Path.home() / ".cargo" / "bin" / "cargo"
    if fallback.exists():
        return str(fallback)
    raise SystemExit("cargo not found in PATH or ~/.cargo/bin")


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=repo_root(), env=env)


def validate_init_weights(path: str) -> Path:
    resolved = expand_path(path)
    if resolved.name == "raw.bin":
        raise SystemExit(
            "--init-weights must point to Bullet optimiser_state/weights.bin, "
            "not raw.bin"
        )
    if resolved.name != "weights.bin":
        raise SystemExit(
            "--init-weights must point to Bullet optimiser_state/weights.bin"
        )
    if resolved.parent.name != "optimiser_state":
        raise SystemExit(
            "--init-weights must point to Bullet optimiser_state/weights.bin"
        )
    if not resolved.exists():
        raise SystemExit(f"--init-weights file not found: {resolved}")
    return resolved


def usable_cuda_path(path: Path) -> bool:
    return (
        (path / "include" / "cuda.h").exists()
        and (path / "lib64" / "libcudart.so").exists()
        and (path / "lib64" / "libnvrtc.so").exists()
        and (path / "lib64" / "libcublas.so").exists()
    )


def symlink_dir(link: Path, target: Path) -> None:
    if link.is_symlink() or link.exists():
        if link.resolve() == target.resolve():
            return
        raise SystemExit(f"{link} already exists and does not point to {target}")
    link.symlink_to(target, target_is_directory=True)


def auto_cuda_path(target_dir: Path) -> Path:
    for raw in (os.environ.get("CUDA_PATH"), "/usr/local/cuda", "/usr/lib/cuda", "/usr"):
        if not raw:
            continue
        path = expand_path(raw)
        if usable_cuda_path(path):
            return path

    debian_include = Path("/usr/include")
    debian_lib = Path("/usr/lib/x86_64-linux-gnu")
    if (
        (debian_include / "cuda.h").exists()
        and (debian_lib / "libcudart.so").exists()
        and (debian_lib / "libnvrtc.so").exists()
        and (debian_lib / "libcublas.so").exists()
    ):
        compat = target_dir / "cuda-path"
        compat.mkdir(parents=True, exist_ok=True)
        symlink_dir(compat / "include", debian_include)
        symlink_dir(compat / "lib64", debian_lib)
        return compat

    raise SystemExit(
        "CUDA libraries not found. Set --cuda-path to a directory with include/cuda.h "
        "and lib64/{libcudart,libnvrtc,libcublas}.so."
    )


def nvidia_compute_cap() -> tuple[int, int] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    first = out.strip().splitlines()[0] if out.strip() else ""
    try:
        major, minor = first.split(".", 1)
        return int(major), int(minor)
    except (ValueError, IndexError):
        return None


def cuda_arch_override(value: str) -> str:
    if value in {"", "native", "none"}:
        return ""
    if value != "auto":
        return value
    cap = nvidia_compute_cap()
    if cap and cap >= (12, 0):
        # CUDA 12.4 does not accept sm_120 in NVRTC. PTX compute_90 still
        # lets the installed driver JIT kernels for Blackwell GPUs.
        return "compute_90"
    return ""


def patch_bullet_cuda_arch(manifest: Path, arch: str) -> None:
    if not arch:
        return
    run([cargo_bin(), "fetch", "--manifest-path", str(manifest)])
    checkouts = Path.home() / ".cargo" / "git" / "checkouts"
    candidates = sorted(checkouts.glob("bullet-*/**/crates/gpu/src/runtime/cuda.rs"))
    if not candidates:
        raise SystemExit("could not find Bullet CUDA runtime source in cargo git checkout")
    for path in candidates:
        text = path.read_text(encoding="utf-8")
        old_native = 'arch: Some(format!("sm_{mjr}{mnr}")),'
        old_forced_prefix = 'arch: Some("'
        old_forced_suffix = '".to_string()),'
        new = f'arch: Some("{arch}".to_string()),'
        if new in text:
            continue
        if old_native in text:
            if not (path.with_suffix(path.suffix + ".enyo-bak")).exists():
                path.with_suffix(path.suffix + ".enyo-bak").write_text(text, encoding="utf-8")
            path.write_text(text.replace(old_native, new), encoding="utf-8")
            continue
        start = text.find(old_forced_prefix)
        if start >= 0:
            end = text.find(old_forced_suffix, start)
            if end >= 0:
                end += len(old_forced_suffix)
                if not (path.with_suffix(path.suffix + ".enyo-bak")).exists():
                    path.with_suffix(path.suffix + ".enyo-bak").write_text(text, encoding="utf-8")
                path.write_text(text[:start] + new + text[end:], encoding="utf-8")
                continue
        raise SystemExit(f"could not patch Bullet CUDA arch in {path}")


def cmd_format(args: argparse.Namespace) -> int:
    input_path = expand_path(args.input)
    output_path = expand_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = expand_path(args.bullet_manifest)

    base = [
        cargo_bin(),
        "run",
        "-q",
        "-p",
        "bullet-utils",
        "--manifest-path",
        str(manifest),
        "--",
    ]
    run([
        *base,
        "convert",
        "--from",
        "text",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ])
    if args.validate:
        run([*base, "validate", "--input", str(output_path)])
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    output_path = expand_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_dir = expand_path(args.cargo_target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = repo_root() / "tools" / "bullet" / "spike_trainer" / "Cargo.toml"

    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target_dir)

    run([
        cargo_bin(), "run", "-q", "--release",
        "--bin", "convert_sfbinpack",
        "--manifest-path", str(manifest),
        "--",
        "--data", expand_data_paths(args.data),
        "--output", str(output_path),
        "--buffer-mb", str(args.buffer_mb),
        "--threads", str(args.threads),
        "--limit", str(args.limit),
        "--min-ply", str(args.min_ply),
        "--max-abs-cp", str(args.max_abs_cp),
        "--quiet-only", "1" if args.quiet_only else "0",
    ], env=env)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    data = expand_data_paths(args.data)
    out_dir = expand_path(args.out_dir)
    target_dir = expand_path(args.cargo_target_dir)
    manifest = repo_root() / "tools" / "bullet" / "spike_trainer" / "Cargo.toml"
    out_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "CARGO_TARGET_DIR": str(target_dir),
        "ENYO_BULLET_DATA": data,
        "ENYO_BULLET_LOADER": str(args.loader),
        "ENYO_BULLET_SFBINPACK_BUFFER_MB": str(args.sfbinpack_buffer_mb),
        "ENYO_BULLET_SFBINPACK_MIN_PLY": str(args.sfbinpack_min_ply),
        "ENYO_BULLET_SFBINPACK_MAX_ABS_CP": str(args.sfbinpack_max_abs_cp),
        "ENYO_BULLET_SFBINPACK_QUIET_ONLY": "1" if args.sfbinpack_quiet_only else "0",
        "ENYO_BULLET_OUT": str(out_dir),
        "ENYO_BULLET_NET_ID": args.net_id,
        "ENYO_BULLET_MODE": args.mode,
        "ENYO_BULLET_HIDDEN": str(args.hidden),
        "ENYO_BULLET_L2": str(args.l2),
        "ENYO_BULLET_BATCH_SIZE": str(args.batch_size),
        "ENYO_BULLET_BATCHES": str(args.batches),
        "ENYO_BULLET_SUPERBATCHES": str(args.superbatches),
        "ENYO_BULLET_THREADS": str(args.threads),
        "ENYO_BULLET_WDL": str(args.wdl),
        "ENYO_BULLET_LR": str(args.lr),
        "ENYO_BULLET_FINAL_LR": str(args.final_lr),
        "ENYO_BULLET_ENYO_L0_STD": str(args.enyo_l0_std),
        "ENYO_BULLET_ENYO_L1_STD": str(args.enyo_l1_std),
        "ENYO_BULLET_ENYO_L1_EXPORT_SCALE": str(args.enyo_l1_export_scale),
        "ENYO_BULLET_ENYO_INPUT_FACTORISER": "1" if args.enyo_input_factorizer else "0",
        "ENYO_BULLET_ENYO_INPUT_BUCKETS": str(args.enyo_input_buckets),
        "ENYO_BULLET_ENYO_RUNTIME_INPUT_BUCKETS": str(args.enyo_runtime_input_buckets),
        "ENYO_BULLET_EVAL_SCALE": str(args.eval_scale),
        "ENYO_BULLET_SAVE_RATE": str(args.save_rate),
        "ENYO_BULLET_EXPORT_INIT_ONLY": "1" if args.export_init_only else "0",
        "ENYO_BULLET_TRAINABLE": args.trainable,
        "ENYO_BULLET_WEIGHT_DECAY": str(args.weight_decay),
    })
    if args.init_weights:
        env["ENYO_BULLET_INIT_WEIGHTS"] = str(
            validate_init_weights(args.init_weights))
    if args.cuda_path:
        env["CUDA_PATH"] = str(expand_path(args.cuda_path))
    elif args.accelerator == "cuda":
        env["CUDA_PATH"] = str(auto_cuda_path(target_dir))
        patch_bullet_cuda_arch(manifest, cuda_arch_override(args.cuda_arch))

    command = [
        cargo_bin(),
        "run",
        "-q",
        "--release",
        "--features",
        args.accelerator,
        "--bin",
        "enyo-bullet-spike",
        "--manifest-path",
        str(manifest),
    ]
    run(command, env=env)
    if args.mode == "enyo":
        checkpoints = sorted(out_dir.glob(f"{args.net_id}-*/quantised.bin"))
        if not checkpoints:
            raise SystemExit(f"no Bullet quantised.bin checkpoints found under {out_dir}")
        checkpoints.sort(key=lambda path: path.stat().st_mtime)
        model_path = out_dir.parent / "model.nn"
        raw = expand_enyo_input_buckets(
            checkpoints[-1].read_bytes(),
            args.enyo_input_buckets,
            runtime_input_buckets=args.enyo_runtime_input_buckets,
            hidden=args.hidden,
            l2=args.l2,
        )
        model_path.write_bytes(raw)
        print(f"wrote {model_path}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Bullet conversion/training phases.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fmt = subparsers.add_parser("format", help="Convert Bullet text to BulletFormat.")
    fmt.add_argument("--input", required=True)
    fmt.add_argument("--output", required=True)
    fmt.add_argument("--bullet-manifest", default="~/source/bullet/Cargo.toml")
    fmt.add_argument("--validate", action="store_true", default=True)
    fmt.set_defaults(func=cmd_format)

    train = subparsers.add_parser("train", help="Train the experimental Bullet net.")
    train.add_argument("--data", required=True)
    train.add_argument(
        "--loader",
        choices=["direct", "sfbinpack"],
        default="direct",
        help="Training data loader. direct expects BulletFormat .data; sfbinpack reads Stockfish binpack directly.",
    )
    train.add_argument("--sfbinpack-buffer-mb", type=int, default=1024)
    train.add_argument("--sfbinpack-min-ply", type=int, default=16)
    train.add_argument("--sfbinpack-max-abs-cp", type=int, default=10000)
    train.add_argument("--sfbinpack-quiet-only", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--out-dir", required=True)
    train.add_argument("--net-id", required=True)
    train.add_argument("--cargo-target-dir", required=True)
    train.add_argument("--mode", choices=["reckless", "enyo"], default="reckless")
    train.add_argument("--accelerator", choices=["cuda", "rocm"], default="cuda")
    train.add_argument("--cuda-path", default="")
    train.add_argument("--cuda-arch", default="auto")
    train.add_argument("--hidden", type=int, default=1024)
    train.add_argument("--l2", type=int, default=16)
    train.add_argument("--batch-size", type=int, default=2048)
    train.add_argument("--batches", type=int, default=64)
    train.add_argument("--superbatches", type=int, default=2048)
    train.add_argument("--threads", type=int, default=4)
    train.add_argument("--wdl", type=float, default=0.75)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--final-lr", type=float, default=3e-4)
    train.add_argument("--enyo-l0-std", type=float, default=8.0)
    train.add_argument("--enyo-l1-std", type=float, default=1.0)
    train.add_argument("--enyo-l1-export-scale", type=float, default=1.0)
    train.add_argument(
        "--enyo-input-factorizer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train a shared (piece,square) input factorizer folded into exported Enyo l0w.",
    )
    train.add_argument(
        "--enyo-input-buckets",
        type=int,
        choices=list(ENYO_SUPPORTED_INPUT_BUCKETS),
        default=ENYO_DEFAULT_INPUT_BUCKETS,
        help="Enyo training input king buckets.",
    )
    train.add_argument(
        "--enyo-runtime-input-buckets",
        type=int,
        choices=list(ENYO_SUPPORTED_INPUT_BUCKETS),
        default=ENYO_DEFAULT_INPUT_BUCKETS,
        help="Enyo runtime king buckets to export into the .nn file.",
    )
    train.add_argument("--eval-scale", type=float, default=400.0)
    train.add_argument("--save-rate", type=int, default=64)
    train.add_argument("--init-weights", default="")
    train.add_argument(
        "--trainable",
        choices=["all", "input", "float-head", "output"],
        default="all",
    )
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument(
        "--export-init-only",
        action="store_true",
        help="Load --init-weights, export checkpoint 0, and exit before training.",
    )
    train.set_defaults(func=cmd_train)

    convert = subparsers.add_parser(
        "convert",
        help="Convert sfbinpack to BulletFormat .data (run once, train with direct loader).",
    )
    convert.add_argument("--data", required=True, help="Source sfbinpack path(s), semicolon-separated.")
    convert.add_argument("--output", required=True, help="Destination .data file.")
    convert.add_argument("--cargo-target-dir", required=True)
    convert.add_argument("--buffer-mb", type=int, default=1024)
    convert.add_argument("--threads", type=int, default=4)
    convert.add_argument("--limit", type=int, default=0,
                         help="Maximum filtered rows to write; 0 means unlimited.")
    convert.add_argument("--min-ply", type=int, default=16)
    convert.add_argument("--max-abs-cp", type=int, default=10000)
    convert.add_argument("--quiet-only", action=argparse.BooleanOptionalAction, default=True)
    convert.set_defaults(func=cmd_convert)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
