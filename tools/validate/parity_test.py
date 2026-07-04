#!/usr/bin/env python3
"""Train/export/engine parity test — catches square indexing, sign, and quantization bugs.

Trains a tiny net on a small data slice via Bullet spike_trainer, exports to .nn,
evaluates a known FEN in both Python (.pt model) and engine (evalnet), and asserts
the scores agree within quantization tolerance.

Usage:
    ./parity_test.py --data path/to/training.bullet --engine path/to/enyo --output /tmp/parity

Exit codes:
    0: parity passed
    1: parity failed (scores diverged beyond tolerance)
    2: setup/training/export failure
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Add tools/lib to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bullet.bullet import (
    ENYO_NETWORK_HEADER_SIZE,
    enyo_network_size,
    enyo_v2_container,
    expand_enyo_input_buckets,
    pad_enyo_hidden,
)
from lib import enyo_nnue as nn2
from lib.nnue_model import EnyoNNUE, load_model_from_nn

# Test FEN: a simple middlegame position with pieces scattered
# (not startpos — we want to test actual square mappings)
TEST_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"

# Quantization tolerance: int8 L1 weights + int16 accumulator → expect ~5-10 cp tolerance
TOLERANCE_CP = 15


def run_spike_trainer(
    data_path: Path,
    output_dir: Path,
    rows: int,
    superbatches: int,
    input_buckets: int = 16,
    feature_channels: int = 12,
    hidden: int = 1024,
    output_buckets: int = 1,
) -> Path:
    """Train a tiny net and return a normalized runtime .nn path."""
    if rows <= 0:
        raise ValueError(f"rows must be positive, got {rows}")

    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = min(rows, 2048)
    batches = (rows + batch_size - 1) // batch_size

    env = {
        "ENYO_BULLET_DATA": str(data_path),
        "ENYO_BULLET_OUT": str(output_dir),
        "ENYO_BULLET_NET_ID": "parity",
        "ENYO_BULLET_MODE": "enyo",
        "ENYO_BULLET_HIDDEN": str(hidden),
        "ENYO_BULLET_L2": "16",
        "ENYO_BULLET_BATCH_SIZE": str(batch_size),
        "ENYO_BULLET_BATCHES": str(batches),
        "ENYO_BULLET_SUPERBATCHES": str(superbatches),
        "ENYO_BULLET_THREADS": "4",
        "ENYO_BULLET_LR": "1e-3",
        "ENYO_BULLET_FINAL_LR": "1e-4",
        "ENYO_BULLET_WDL": "0.3",
        "ENYO_BULLET_ENYO_INPUT_BUCKETS": str(input_buckets),
        "ENYO_BULLET_ENYO_FEATURE_CHANNELS": str(feature_channels),
        "ENYO_BULLET_ENYO_OUTPUT_BUCKETS": str(output_buckets),
        "ENYO_BULLET_ENYO_INPUT_FACTORISER": "0",
        "ENYO_BULLET_EVAL_SCALE": "400",
        "ENYO_BULLET_SAVE_RATE": str(superbatches),
    }

    spike_trainer = Path(__file__).resolve().parents[1] / "bullet" / "spike_trainer" / "target" / "release" / "enyo-bullet-spike"
    if not spike_trainer.exists():
        spike_trainer = shutil.which("enyo-bullet-spike")
        if not spike_trainer:
            raise FileNotFoundError(
                "spike_trainer not found. Build it first:\n"
                "  cd tools/bullet/spike_trainer && cargo build --release"
            )
        spike_trainer = Path(spike_trainer)

    print(f"Training {rows} rows for {superbatches} superbatches via {spike_trainer}...", flush=True)
    result = subprocess.run(
        [str(spike_trainer)],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"spike_trainer failed with exit code {result.returncode}")

    # Select the Bullet checkpoint, then normalize it for the Enyo runtime.
    net_id = "parity"
    checkpoint_path = output_dir / f"{net_id}-{superbatches}" / "quantised.bin"
    if not checkpoint_path.exists():
        # Try checkpoint-0 if final doesn't exist
        checkpoint_path = output_dir / f"{net_id}-0" / "quantised.bin"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Expected checkpoint at {checkpoint_path} but not found. "
            f"spike_trainer output:\n{result.stdout}"
        )

    runtime_net = expand_enyo_input_buckets(
        checkpoint_path.read_bytes(),
        input_buckets,
        feature_channels=feature_channels,
        runtime_input_buckets=input_buckets,
        output_buckets=output_buckets,
        hidden=hidden,
        l2=16,
    )
    runtime_net = pad_enyo_hidden(
        runtime_net,
        input_buckets,
        feature_channels=feature_channels,
        output_buckets=output_buckets,
        hidden=hidden,
    )
    expected_payload_size = enyo_network_size(
        input_buckets,
        feature_channels=feature_channels,
        output_buckets=output_buckets,
        hidden=1024,
        l2=16,
    )
    if len(runtime_net) != expected_payload_size:
        raise RuntimeError(
            f"runtime net is {len(runtime_net)} bytes, expected {expected_payload_size}"
        )
    runtime_net = enyo_v2_container(
        runtime_net,
        input_buckets=input_buckets,
        feature_channels=feature_channels,
        hidden=hidden,
        output_buckets=output_buckets,
    )

    runtime_path = output_dir / "parity-runtime.nn"
    runtime_path.write_bytes(runtime_net)
    print(f"Exported runtime net to {runtime_path}", flush=True)
    return runtime_path


def eval_python(nn_path: Path, fen: str) -> int:
    """Evaluate FEN using Python .pt model. Returns score in cp."""
    model = load_model_from_nn(nn_path, device="cpu")
    model.eval()

    pieces, stm = nn2.parse_fen(fen)
    pieces.sort(key=lambda item: item[2])
    w_feats = nn2.features_from_pieces(
        pieces, nn2.WHITE, model.input_buckets, model.feature_channels)
    b_feats = nn2.features_from_pieces(
        pieces, nn2.BLACK, model.input_buckets, model.feature_channels)
    phase_scale = nn2.phase_scale_from_pieces(pieces)
    piece_count = len(pieces)

    import torch
    with torch.no_grad():
        score = model(
            torch.tensor(w_feats, dtype=torch.long),
            torch.tensor(b_feats, dtype=torch.long),
            torch.tensor([0], dtype=torch.long),  # offset 0
            torch.tensor([0], dtype=torch.long),  # offset 0
            torch.tensor([stm], dtype=torch.long),
            torch.tensor([phase_scale], dtype=torch.float32),
            piece_count=torch.tensor([piece_count], dtype=torch.long),
        )

    return int(round(score.item()))


def eval_engine(engine_path: Path, nn_path: Path, fen: str) -> int:
    """Evaluate FEN using engine evalnet. Returns score in cp."""
    proc = subprocess.Popen(
        [str(engine_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    commands = [
        "uci",
        f"setoption name nnue_file value {nn_path.resolve()}",
        "isready",
        f"position fen {fen}",
        "evalnet",
        "quit",
    ]

    stdout, stderr = proc.communicate("\n".join(commands) + "\n", timeout=5.0)

    if proc.returncode != 0:
        raise RuntimeError(f"Engine failed: {stderr}")

    # Parse "evalnet <score> cp" from output
    for line in stdout.splitlines():
        if line.startswith("evalnet "):
            parts = line.split()
            if len(parts) >= 3 and parts[2] == "cp":
                return int(parts[1])

    raise RuntimeError(f"Engine didn't output 'evalnet <score> cp'. stdout:\n{stdout}\nstderr:\n{stderr}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Path to .bullet training data")
    ap.add_argument("--engine", required=True, help="Path to Enyo engine binary")
    ap.add_argument("--output", default=None, help="Output directory (default: temp)")
    ap.add_argument("--rows", type=int, default=10000, help="Training rows (default: 10000)")
    ap.add_argument("--superbatches", type=int, default=8, help="Training superbatches (default: 8)")
    ap.add_argument("--input-buckets", type=int, default=16,
                    choices=(10, 16, 32))
    ap.add_argument("--feature-channels", type=int, default=12,
                    choices=(11, 12))
    ap.add_argument("--hidden", type=int, default=1024,
                    choices=(512, 768, 1024))
    ap.add_argument("--output-buckets", type=int, default=8,
                    choices=(1, 4, 8))
    ap.add_argument("--fen", default=TEST_FEN, help=f"Test FEN (default: {TEST_FEN})")
    ap.add_argument("--tolerance", type=int, default=TOLERANCE_CP, help=f"Tolerance in cp (default: {TOLERANCE_CP})")
    args = ap.parse_args()

    data_path = Path(args.data)
    engine_path = Path(args.engine)

    if not data_path.exists():
        print(f"Error: data file not found: {data_path}", file=sys.stderr)
        return 2

    if not engine_path.exists():
        print(f"Error: engine not found: {engine_path}", file=sys.stderr)
        return 2

    if args.output:
        output_dir = Path(args.output)
        cleanup = False
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="parity_test_"))
        cleanup = True

    try:
        # Train tiny net
        nn_path = run_spike_trainer(
            data_path,
            output_dir,
            args.rows,
            args.superbatches,
            args.input_buckets,
            args.feature_channels,
            args.hidden,
            args.output_buckets,
        )

        # Eval in Python
        print(f"Evaluating {args.fen} in Python...", flush=True)
        score_py = eval_python(nn_path, args.fen)
        print(f"  Python: {score_py:+d} cp", flush=True)

        # Eval in engine
        print(f"Evaluating {args.fen} in engine...", flush=True)
        score_engine = eval_engine(engine_path, nn_path, args.fen)
        print(f"  Engine: {score_engine:+d} cp", flush=True)

        # Compare
        diff = abs(score_py - score_engine)
        print(f"  Diff:   {diff} cp (tolerance: {args.tolerance} cp)", flush=True)

        if diff <= args.tolerance:
            print("PASS: parity within tolerance", flush=True)
            return 0
        else:
            print(f"FAIL: parity exceeded tolerance ({diff} > {args.tolerance})", file=sys.stderr, flush=True)
            return 1

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        return 2

    finally:
        if cleanup and output_dir.exists():
            shutil.rmtree(output_dir)


if __name__ == "__main__":
    sys.exit(main())
