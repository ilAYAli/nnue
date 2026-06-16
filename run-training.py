#!/usr/bin/env python3
"""train.py - Train an NNUE net
Usage: ./train.py <run_name> <superbatches>
"""
import json
import os
import subprocess
import sys
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print("Usage: train.py <run_name> <superbatches>", file=sys.stderr)
        sys.exit(1)

    run_name = sys.argv[1]
    superbatches = int(sys.argv[2])

    config_path = Path(os.getenv("ENYO_NNUE_CONFIG", "~/.config/enyo/nnue.json")).expanduser()
    config = json.loads(config_path.read_text())

    repo_root = Path(__file__).parent
    os.chdir(repo_root)

    assets_dir = Path(config["assets_dir"]).expanduser()
    spike_trainer = repo_root / config["spike_trainer"]

    run_dir = repo_root / "runs" / run_name
    output_dir = run_dir / "checkpoints"
    net_path = assets_dir / f"{run_name}.nn"

    # Determine final LR based on superbatch count
    if superbatches <= 1000:
        final_lr = config["training"]["lr_end_smoke"]
    else:
        final_lr = config["training"]["lr_end_deep"]

    # Build env vars for spike_trainer (Enyo mode)
    env = os.environ.copy()
    env.update({
        "ENYO_BULLET_MODE": "enyo",
        "ENYO_BULLET_DATA": config["data"],
        "ENYO_BULLET_OUT": str(output_dir),
        "ENYO_BULLET_SUPERBATCHES": str(superbatches),
        "ENYO_BULLET_LR": str(config["training"]["lr_start"]),
        "ENYO_BULLET_FINAL_LR": str(final_lr),
        "ENYO_BULLET_HIDDEN": str(config["architecture"]["hidden"]),
        "ENYO_BULLET_L2": str(config["architecture"]["l2_size"]),
        "ENYO_BULLET_ENYO_INPUT_BUCKETS": str(config["architecture"]["input_buckets"]),
        "ENYO_BULLET_ENYO_FEATURE_CHANNELS": str(config["architecture"]["feature_channels"]),
        "ENYO_BULLET_ENYO_OUTPUT_BUCKETS": str(config["architecture"]["output_buckets"]),
        "ENYO_BULLET_THREADS": str(config["training"]["threads"]),
        "ENYO_BULLET_WDL": str(config["training"]["wdl_lambda"]),
        "ENYO_BULLET_EXPORT_INIT_ONLY": str(config["training"]["export_init_only"]),
    })

    print(f"=== Training {run_name}: {superbatches} superbatches ===", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(spike_trainer)], env=env, check=True)

    print(f"\n=== Exporting to {net_path} ===", flush=True)
    checkpoint = output_dir / f"enyo_bullet_spike-{superbatches}/quantised.bin"
    net_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", str(checkpoint), str(net_path)], check=True)

    print(f"\nTrained net: {net_path}")

if __name__ == "__main__":
    main()
