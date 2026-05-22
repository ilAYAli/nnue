#!/usr/bin/env python3
"""Train Enyo material-bucket output heads from an existing .nn.

This is for the `nnue_reckless` lane: keep the existing sparse input and L1
path fixed, duplicate the current float head into material buckets, then train
only the small bucketed head. It exports Enyo's bucketed-head .nn format.
"""
from __future__ import annotations

import argparse
from functools import partial
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import enyo_nnue as nn2


MPE_SCALE = 2.5 / 400.0
MPE_EXPONENT = 2.5


class PackedMaterialDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        *,
        limit: int = 0,
        skip: int = 0,
        bucket_mode: str = "material",
    ):
        root = Path(path)
        self.bucket_mode = bucket_mode
        self.w_features = np.load(root / "white_features.npy", mmap_mode="r")
        self.b_features = np.load(root / "black_features.npy", mmap_mode="r")
        self.counts = np.load(root / "counts.npy", mmap_mode="r")
        self.stms = np.load(root / "stm.npy", mmap_mode="r")
        self.scores = np.load(root / "score.npy", mmap_mode="r")
        self.wdls = np.load(root / "wdl.npy", mmap_mode="r")
        self.phase_scales = np.load(root / "phase_scale.npy", mmap_mode="r")
        king_pressure_path = root / "king_pressure_bucket.npy"
        self.king_pressure_buckets = (
            np.load(king_pressure_path, mmap_mode="r")
            if king_pressure_path.exists() else None
        )
        check_state_path = root / "check_state_bucket.npy"
        self.check_state_buckets = (
            np.load(check_state_path, mmap_mode="r")
            if check_state_path.exists() else None
        )
        if bucket_mode == "king-pressure" and self.king_pressure_buckets is None:
            raise FileNotFoundError(
                f"{king_pressure_path} missing; rebuild pack data first")
        if bucket_mode == "check-state" and self.check_state_buckets is None:
            raise FileNotFoundError(
                f"{check_state_path} missing; rebuild pack data first")
        rows = len(self.counts)
        self.start = min(skip, rows)
        self.end = rows if limit <= 0 else min(rows, self.start + limit)

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, idx: int):
        real_idx = self.start + idx
        return (
            self.w_features[real_idx],
            self.b_features[real_idx],
            self.counts[real_idx],
            self.stms[real_idx],
            self.scores[real_idx],
            self.wdls[real_idx],
            self.phase_scales[real_idx],
            (
                self.king_pressure_buckets[real_idx]
                if self.bucket_mode == "king-pressure"
                and self.king_pressure_buckets is not None
                else self.check_state_buckets[real_idx]
                if self.bucket_mode == "check-state"
                and self.check_state_buckets is not None
                else 0
            ),
        )


def collate_material(batch, *, bucket_mode: str):
    w_rows, b_rows, counts, stms, scores, wdls, phase_scales, row_buckets = zip(*batch)
    w_padded = torch.as_tensor(np.stack(w_rows), dtype=torch.long)
    b_padded = torch.as_tensor(np.stack(b_rows), dtype=torch.long)
    counts_t = torch.as_tensor(np.asarray(counts), dtype=torch.long)
    max_features = w_padded.shape[1]
    mask = torch.arange(max_features).unsqueeze(0) < counts_t.unsqueeze(1)
    offsets = torch.cat((
        torch.zeros(1, dtype=torch.long),
        torch.cumsum(counts_t, dim=0)[:-1],
    ))
    if bucket_mode == "material":
        buckets = torch.clamp((counts_t - 2) // 4, 0, nn2.N_OUTPUT_BUCKETS - 1)
    elif bucket_mode == "king-pressure":
        buckets = torch.as_tensor(np.asarray(row_buckets), dtype=torch.long)
    elif bucket_mode == "check-state":
        buckets = torch.as_tensor(np.asarray(row_buckets), dtype=torch.long)
    else:
        raise ValueError(f"unknown bucket mode: {bucket_mode}")
    return (
        w_padded[mask],
        b_padded[mask],
        offsets,
        offsets.clone(),
        torch.as_tensor(np.asarray(stms), dtype=torch.long),
        torch.as_tensor(np.asarray(scores), dtype=torch.float32),
        torch.as_tensor(np.asarray(wdls), dtype=torch.float32),
        torch.as_tensor(np.asarray(phase_scales), dtype=torch.float32),
        buckets,
    )


class MaterialHeadNNUE(nn.Module):
    def __init__(self, net: nn2.Net, *, trainable: str):
        super().__init__()
        self.register_buffer(
            "input_weight",
            torch.from_numpy(net.input_weights.astype(np.float32)),
        )
        self.register_buffer(
            "input_bias",
            torch.from_numpy(net.input_biases.astype(np.float32)),
        )
        self.register_buffer(
            "l1_weight",
            torch.from_numpy(net.l1_weights.astype(np.float32)),
        )
        self.register_buffer(
            "l1_bias",
            torch.from_numpy(net.l1_biases.astype(np.float32)),
        )

        if net.l2_weights.ndim == 2:
            l2_weight = np.repeat(
                net.l2_weights[np.newaxis, :, :],
                nn2.N_OUTPUT_BUCKETS,
                axis=0,
            )
            l2_bias = np.repeat(
                net.l2_biases[np.newaxis, :],
                nn2.N_OUTPUT_BUCKETS,
                axis=0,
            )
            output_weight = np.repeat(
                net.output_weights[np.newaxis, :],
                nn2.N_OUTPUT_BUCKETS,
                axis=0,
            )
            output_bias = np.repeat(
                np.asarray([net.output_bias], dtype=np.float32),
                nn2.N_OUTPUT_BUCKETS,
                axis=0,
            )
        else:
            l2_weight = net.l2_weights
            l2_bias = net.l2_biases
            output_weight = net.output_weights
            output_bias = net.output_bias

        self.l2_weight = nn.Parameter(torch.from_numpy(l2_weight.astype(np.float32)))
        self.l2_bias = nn.Parameter(torch.from_numpy(l2_bias.astype(np.float32)))
        self.output_weight = nn.Parameter(
            torch.from_numpy(output_weight.astype(np.float32)))
        self.output_bias = nn.Parameter(
            torch.from_numpy(np.asarray(output_bias, dtype=np.float32)))

        if trainable == "output":
            self.l2_weight.requires_grad_(False)
            self.l2_bias.requires_grad_(False)
        elif trainable != "float-head":
            raise ValueError(f"unknown trainable mode: {trainable}")

    def forward(self, w_feats, b_feats, w_offsets, b_offsets, stm,
                phase_scale, buckets):
        w_acc = F.embedding_bag(
            w_feats, self.input_weight, w_offsets, mode="sum") + self.input_bias
        b_acc = F.embedding_bag(
            b_feats, self.input_weight, b_offsets, mode="sum") + self.input_bias
        stm_f = stm.unsqueeze(-1).float()
        us = (1.0 - stm_f) * w_acc + stm_f * b_acc
        them = stm_f * w_acc + (1.0 - stm_f) * b_acc
        acc = torch.cat([us, them], dim=-1)
        x0 = torch.clamp(acc, min=0.0, max=float(127 << nn2.QUANT1_BITS))
        x0 = x0 / float(1 << nn2.QUANT1_BITS)
        x1 = torch.relu(x0 @ self.l1_weight.t() + self.l1_bias)

        l2_weight = self.l2_weight[buckets]
        l2_bias = self.l2_bias[buckets]
        x2 = torch.relu(torch.bmm(l2_weight, x1.unsqueeze(-1)).squeeze(-1)
                        + l2_bias)
        output_weight = self.output_weight[buckets]
        raw = (x2 * output_weight).sum(dim=-1) + self.output_bias[buckets]
        pred = raw / nn2.EVAL_DIVISOR
        return torch.clamp(pred * phase_scale, min=-2045.0, max=2045.0)


def score_loss(pred, target, wdl, args):
    if args.objective == "huber":
        losses = F.smooth_l1_loss(
            pred, target, beta=args.huber_beta, reduction="none")
    elif args.objective == "mse":
        losses = (pred - target) ** 2
    elif args.objective == "mpe25":
        pred_p = torch.sigmoid(pred * MPE_SCALE)
        target_p = torch.sigmoid(target * MPE_SCALE)
        target_p = args.wdl_lambda * target_p + (1.0 - args.wdl_lambda) * wdl
        losses = (pred_p - target_p).abs() ** MPE_EXPONENT
    else:
        raise ValueError(args.objective)

    if args.sign_loss_weight > 0.0:
        sign_mask = target != 0
        sign_targets = (target > 0).to(pred.dtype)
        sign_losses = F.binary_cross_entropy_with_logits(
            pred / args.sign_loss_scale,
            sign_targets,
            reduction="none",
        )
        losses = losses + torch.where(
            sign_mask,
            sign_losses * args.sign_loss_weight,
            torch.zeros_like(sign_losses),
        )
    return losses.mean()


@torch.no_grad()
def eval_metrics(model, loader, args):
    model.eval()
    loss_sum = mae_sum = mse_sum = 0.0
    sign_sum = sign_n = n = 0
    bucket_rows = [0] * nn2.N_OUTPUT_BUCKETS
    for batch in loader:
        w, b, w_off, b_off, stm, y, wdl, phase_scale, buckets = [
            item.to(args.device) for item in batch
        ]
        if args.target_clamp > 0:
            y = torch.clamp(y, -args.target_clamp, args.target_clamp)
        pred = model(w, b, w_off, b_off, stm, phase_scale, buckets)
        loss = score_loss(pred, y, wdl, args)
        err = pred - y
        sign_mask = y != 0
        batch_n = len(y)
        loss_sum += float(loss) * batch_n
        mae_sum += float(err.abs().sum())
        mse_sum += float((err * err).sum())
        sign_sum += int(((pred[sign_mask] > 0) == (y[sign_mask] > 0)).sum())
        sign_n += int(sign_mask.sum())
        n += batch_n
        counts = torch.bincount(
            buckets.detach().cpu(), minlength=nn2.N_OUTPUT_BUCKETS)
        for idx, count in enumerate(counts.tolist()):
            bucket_rows[idx] += int(count)
    model.train()
    denom = max(1, n)
    return {
        "loss": loss_sum / denom,
        "mse": mse_sum / denom,
        "mae": mae_sum / denom,
        "sign": sign_sum / max(1, sign_n),
        "bucket_rows": bucket_rows,
    }


def select_value(metrics, args):
    value = metrics[args.select_metric]
    return -value if args.select_metric == "sign" else value


def export_bucketed(model: MaterialHeadNNUE, base: nn2.Net, path: str | Path):
    m = model.cpu()
    out = nn2.Net(
        input_weights=base.input_weights,
        input_biases=base.input_biases,
        l1_weights=base.l1_weights,
        l1_biases=base.l1_biases,
        l2_weights=m.l2_weight.detach().numpy().astype(np.float32),
        l2_biases=m.l2_bias.detach().numpy().astype(np.float32),
        output_weights=m.output_weight.detach().numpy().astype(np.float32),
        output_bias=m.output_bias.detach().numpy().astype(np.float32),
    )
    nn2.write_net(out, path)


def train(args):
    if not Path(args.data).is_dir():
        raise SystemExit("train_material_head.py currently expects packed data dir")

    total_rows = len(np.load(Path(args.data) / "counts.npy", mmap_mode="r"))
    train_limit = args.max_rows
    val_skip = args.skip_rows + (args.max_rows if args.max_rows > 0 else 0)
    if args.max_rows == 0 and args.val_rows > 0:
        train_limit = max(0, total_rows - args.skip_rows - args.val_rows)
        val_skip = args.skip_rows + train_limit
    print(
        f"rows total={total_rows} train_skip={args.skip_rows} "
        f"train_limit={train_limit} val_skip={val_skip} val_rows={args.val_rows}",
        flush=True,
    )

    train_set = PackedMaterialDataset(
        args.data, limit=train_limit, skip=args.skip_rows,
        bucket_mode=args.bucket_mode)
    val_set = (
        PackedMaterialDataset(
            args.data, limit=args.val_rows, skip=val_skip,
            bucket_mode=args.bucket_mode)
        if args.val_rows > 0 else None
    )
    collate = partial(collate_material, bucket_mode=args.bucket_mode)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=args.workers,
            pin_memory=args.device.startswith("cuda"),
        )
        if val_set is not None else None
    )

    base = nn2.load_net(args.init_from_nn)
    model = MaterialHeadNNUE(base, trainable=args.trainable).to(args.device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(
        f"bucket_mode={args.bucket_mode} trainable={args.trainable} "
        f"params={sum(p.numel() for p in params)}",
        flush=True,
    )
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    best_metric = float("inf")
    best_display = float("inf")
    best_state = None
    bad = 0
    for epoch in range(args.epochs):
        mae_sum = mse_sum = 0.0
        n = 0
        for batch in train_loader:
            w, b, w_off, b_off, stm, y, wdl, phase_scale, buckets = [
                item.to(args.device) for item in batch
            ]
            if args.target_clamp > 0:
                y = torch.clamp(y, -args.target_clamp, args.target_clamp)
            pred = model(w, b, w_off, b_off, stm, phase_scale, buckets)
            loss = score_loss(pred, y, wdl, args)
            opt.zero_grad()
            loss.backward()
            opt.step()
            err = pred.detach() - y
            mae_sum += float(err.abs().sum())
            mse_sum += float((err * err).sum())
            n += len(y)

        line = (
            f"epoch {epoch:4d} train mse={mse_sum / max(1, n):10.2f} "
            f"mae={mae_sum / max(1, n):7.2f}"
        )
        if val_loader is not None:
            metrics = eval_metrics(model, val_loader, args)
            line += (
                f" val loss={metrics['loss']:.6f}"
                f" mse={metrics['mse']:10.2f}"
                f" mae={metrics['mae']:7.2f}"
                f" sign={metrics['sign'] * 100:5.2f}%"
            )
            metric = select_value(metrics, args)
            if metric < best_metric:
                best_metric = metric
                best_display = metrics[args.select_metric]
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                bad = 0
            else:
                bad += 1
            if args.patience > 0:
                line += f" best_{args.select_metric}={best_display:.6f} bad={bad}"
        print(line, flush=True)
        if args.patience > 0 and val_loader is not None and bad >= args.patience:
            print(f"early stop at epoch {epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--init-from-nn", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--out-nn", required=True)
    parser.add_argument("--trainable", default="output",
                        choices=["output", "float-head"])
    parser.add_argument("--bucket-mode", default="material",
                        choices=["material", "king-pressure", "check-state"])
    parser.add_argument("--objective", default="huber",
                        choices=["huber", "mse", "mpe25"])
    parser.add_argument("--huber-beta", type=float, default=200.0)
    parser.add_argument("--wdl-lambda", type=float, default=0.75)
    parser.add_argument("--sign-loss-weight", type=float, default=0.0)
    parser.add_argument("--sign-loss-scale", type=float, default=100.0)
    parser.add_argument("--select-metric", default="sign",
                        choices=["loss", "mse", "mae", "sign"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--target-clamp", type=float, default=800.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--skip-rows", type=int, default=0)
    parser.add_argument("--val-rows", type=int, default=100000)
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()

    model, base = train(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    export_bucketed(model, base, args.out_nn)
    print(f"wrote {out}", flush=True)
    print(f"wrote {args.out_nn}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
