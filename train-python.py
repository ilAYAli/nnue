#!/usr/bin/env python3
"""Python-based NNUE training (fallback while spike_trainer export is broken)
Usage: ./train-python.py <run_name> <rows> <epochs>
"""
import json
import os
import subprocess
import sys
from pathlib import Path

def main():
    if len(sys.argv) != 4:
        print("Usage: train-python.py <run_name> <rows> <epochs>", file=sys.stderr)
        sys.exit(1)

    run_name = sys.argv[1]
    rows = int(sys.argv[2])
    epochs = int(sys.argv[3])

    config_path = Path(os.getenv("ENYO_NNUE_CONFIG", "~/.config/enyo/nnue.json")).expanduser()
    config = json.loads(config_path.read_text())

    repo_root = Path(__file__).parent
    os.chdir(repo_root)

    assets_dir = Path(config["assets_dir"]).expanduser()
    data_path = repo_root / config["data"]

    run_dir = repo_root / "runs" / run_name
    output_dir = run_dir / "pytorch"
    net_path = assets_dir / f"{run_name}.nn"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Activate venv and train
    venv_python = repo_root / ".venv/bin/python"

    cmd = [
        str(venv_python),
        "tools/train/train.py", "run",
        "--data", str(data_path),
        "--max-rows", str(rows),
        "--epochs", str(epochs),
        "--batch-size", "8192",
        "--lr", str(config["training"]["lr_start"]),
        "--device", "cuda",
        "--out", str(output_dir),
        "--out-nn", str(net_path),
    ]

    print(f"=== Training {run_name}: {rows} rows × {epochs} epochs ===", flush=True)
    subprocess.run(cmd, check=True)

    print(f"\nTrained net: {net_path}")

if __name__ == "__main__":
    main()
