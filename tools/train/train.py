#!/usr/bin/env python3
"""Train and evaluate Enyo NNUE candidates."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def expand_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def expand_user(value: str | Path) -> Path:
    return Path(value).expanduser()


def tools_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run(command: list[str]) -> int:
    print(" ".join(command), flush=True)
    proc = subprocess.Popen(command)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        raise


def cmd_run(args: argparse.Namespace) -> int:
    script = tools_root() / "train" / "train_impl.py"
    command = [
        str(expand_user(args.python)),
        str(script),
        "--data", str(expand_path(args.data)),
        "--out", str(expand_path(args.out)),
        "--objective", args.objective,
        "--forward", args.forward,
        "--huber-beta", str(args.huber_beta),
        "--select-metric", args.select_metric,
        "--wdl-lambda", str(args.wdl_lambda),
        "--sign-loss-weight", str(args.sign_loss_weight),
        "--sign-loss-scale", str(args.sign_loss_scale),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--input-lr-mult", str(args.input_lr_mult),
        "--l1-lr-mult", str(args.l1_lr_mult),
        "--dense-lr-mult", str(args.dense_lr_mult),
        "--target-clamp", str(args.target_clamp),
        "--device", args.device,
        "--workers", str(args.workers),
        "--prefetch-factor", str(args.prefetch_factor),
        "--amp", args.amp,
        "--patience", str(args.patience),
        "--max-rows", str(args.max_rows),
        "--skip-rows", str(args.skip_rows),
        "--val-rows", str(args.val_rows),
        "--trainable", args.trainable,
        "--grad-norm-every", str(args.grad_norm_every),
    ]
    if args.val:
        command += ["--val", str(expand_path(args.val))]
    if args.out_nn:
        command += ["--out-nn", str(expand_path(args.out_nn))]
    if args.init_from_nn:
        command += ["--init-from-nn", str(expand_path(args.init_from_nn))]
    else:
        command += ["--init", args.init]
    for item in args.source_wdl_lambda or []:
        command += ["--source-wdl-lambda", item]
    for item in args.source_loss_weight or []:
        command += ["--source-loss-weight", item]
    return run(command)


def cmd_eval(args: argparse.Namespace) -> int:
    script = tools_root() / "validate" / "eval_dataset.py"
    command = [
        str(expand_user(args.python)),
        str(script),
        "--net", str(expand_path(args.net)),
        "--data", str(expand_path(args.data)),
        "--rows", str(args.rows),
        "--skip", str(args.skip),
        "--batch-size", str(args.batch_size),
        "--device", args.device,
        "--target-clamp", str(args.target_clamp),
    ]
    if args.buckets:
        command.append("--buckets")
    if args.sources:
        command.append("--sources")
    return run(command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Enyo NNUE candidates.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("run", help="Train/export a candidate net.")
    train.add_argument("--data", required=True)
    train.add_argument("--val")
    train.add_argument("--out", required=True)
    train.add_argument("--out-nn")
    train.add_argument("--init-from-nn")
    train.add_argument("--init", default="kaiming",
                       choices=["kaiming", "quantized-kaiming", "berserk-ish"])
    train.add_argument("--objective", default="mpe25",
                       choices=["mse", "huber", "mpe25"])
    train.add_argument("--forward", default="float",
                       choices=["float", "quantized"])
    train.add_argument("--huber-beta", type=float, default=200.0)
    train.add_argument("--select-metric", default="loss",
                       choices=["loss", "mse", "mae", "sign"])
    train.add_argument("--wdl-lambda", type=float, default=0.75)
    train.add_argument("--sign-loss-weight", type=float, default=0.0)
    train.add_argument("--sign-loss-scale", type=float, default=100.0)
    train.add_argument("--source-wdl-lambda", action="append", default=[])
    train.add_argument("--source-loss-weight", action="append", default=[])
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--batch-size", type=int, default=4096)
    train.add_argument("--lr", type=float, default=1e-5)
    train.add_argument("--weight-decay", type=float, default=1e-6)
    train.add_argument("--input-lr-mult", type=float, default=1.0)
    train.add_argument("--l1-lr-mult", type=float, default=1.0)
    train.add_argument("--dense-lr-mult", type=float, default=1.0)
    train.add_argument("--target-clamp", type=float, default=0.0)
    train.add_argument("--device", default="cpu")
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--prefetch-factor", type=int, default=2)
    train.add_argument("--amp", default="off", choices=["off", "bf16"])
    train.add_argument("--patience", type=int, default=0)
    train.add_argument("--max-rows", type=int, default=0)
    train.add_argument("--skip-rows", type=int, default=0)
    train.add_argument("--val-rows", type=int, default=0)
    train.add_argument("--trainable", default="all",
                       choices=["all", "input", "float-head", "output"])
    train.add_argument("--grad-norm-every", type=int, default=0)
    train.add_argument("--python", default=sys.executable)
    train.set_defaults(func=cmd_run)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a net on packed data.")
    eval_parser.add_argument("--net", required=True)
    eval_parser.add_argument("--data", required=True)
    eval_parser.add_argument("--rows", type=int, default=50000)
    eval_parser.add_argument("--skip", type=int, default=0)
    eval_parser.add_argument("--batch-size", type=int, default=4096)
    eval_parser.add_argument("--device", default="cpu")
    eval_parser.add_argument("--target-clamp", type=float, default=0.0)
    eval_parser.add_argument("--buckets", action="store_true")
    eval_parser.add_argument("--sources", action="store_true")
    eval_parser.add_argument("--python", default=sys.executable)
    eval_parser.set_defaults(func=cmd_eval)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
