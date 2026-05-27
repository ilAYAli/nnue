#!/usr/bin/env python3
"""Train child-move ranking groups with broad deadzone preservation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import enyo_nnue as nn2
from lib.child_rank_targets import ChildRankGroup, child_fen, load_groups, target_gap_cp
from lib.nnue_dataset import load_score_dataset
from lib.nnue_model import EnyoNNUE, export_model, load_model_from_nn


def fen_features(fen: str) -> tuple[list[int], list[int], int, float]:
    pieces, stm = nn2.parse_fen(fen)
    pieces.sort(key=lambda item: item[2])
    return (
        nn2.features_from_pieces(pieces, nn2.WHITE),
        nn2.features_from_pieces(pieces, nn2.BLACK),
        stm,
        nn2.phase_scale_from_pieces(pieces),
    )


class ChildRankingDataset(Dataset):
    def __init__(self, groups: list[ChildRankGroup], *,
                 rank_margin_cp: float) -> None:
        self.items = []
        for group in groups:
            best = group.best
            best_features = fen_features(child_fen(group.fen, best.move))
            for move in group.moves:
                if move.move == best.move:
                    continue
                gap = target_gap_cp(group, move)
                if gap <= 0.0:
                    continue
                self.items.append((
                    group.group_id,
                    best.move,
                    move.move,
                    best_features,
                    fen_features(child_fen(group.fen, move.move)),
                    gap,
                    min(gap, rank_margin_cp),
                ))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def collate_feature_rows(rows):
    w_all, b_all = [], []
    w_offsets, b_offsets = [0], [0]
    stms, phase_scales = [], []
    for w, b, stm, phase_scale in rows:
        w_all.append(torch.tensor(w, dtype=torch.long))
        b_all.append(torch.tensor(b, dtype=torch.long))
        w_offsets.append(w_offsets[-1] + len(w))
        b_offsets.append(b_offsets[-1] + len(b))
        stms.append(torch.tensor(stm, dtype=torch.long))
        phase_scales.append(torch.tensor(phase_scale, dtype=torch.float32))
    return (
        torch.cat(w_all),
        torch.cat(b_all),
        torch.tensor(w_offsets[:-1], dtype=torch.long),
        torch.tensor(b_offsets[:-1], dtype=torch.long),
        torch.stack(stms),
        torch.stack(phase_scales),
    )


def collate_rank_pairs(batch):
    best_rows = [item[3] for item in batch]
    other_rows = [item[4] for item in batch]
    gaps = torch.tensor([item[5] for item in batch], dtype=torch.float32)
    margins = torch.tensor([item[6] for item in batch], dtype=torch.float32)
    return (*collate_feature_rows(best_rows),
            *collate_feature_rows(other_rows),
            gaps, margins)


def cycle(loader):
    while True:
        yield from loader


def move_batch(batch, device: str):
    return tuple(
        item.to(device, non_blocking=True) if torch.is_tensor(item) else item
        for item in batch
    )


def dataloader_kwargs(args: argparse.Namespace) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "num_workers": args.workers,
        "pin_memory": args.device.startswith("cuda"),
    }
    if args.workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def autocast_context(args: argparse.Namespace):
    if args.amp == "bf16" and args.device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    from contextlib import nullcontext
    return nullcontext()


def maybe_compile_model(model: EnyoNNUE, args: argparse.Namespace):
    if args.torch_compile:
        print("torch.compile enabled mode=reduce-overhead", flush=True)
        return torch.compile(model, mode="reduce-overhead")
    return model


def train(args: argparse.Namespace) -> EnyoNNUE:
    groups = load_groups(args.child_targets, min_groups=args.min_groups)
    rank_set = ChildRankingDataset(groups, rank_margin_cp=args.rank_margin_cp)
    if len(rank_set) < args.min_pairs:
        raise SystemExit(
            f"valid_pairs={len(rank_set)} below min_pairs={args.min_pairs}")

    broad_set, broad_collate = load_score_dataset(
        args.data, limit=args.max_rows, skip=args.skip_rows,
        in_memory=args.dataset_in_memory)
    if len(broad_set) == 0:
        raise SystemExit("no broad rows")

    print(f"child groups: {len(groups)}", flush=True)
    print(f"child rank pairs: {len(rank_set)}", flush=True)
    print(f"broad rows: {len(broad_set)}", flush=True)

    if args.init_from_nn:
        model = load_model_from_nn(args.init_from_nn, device=args.device)
    else:
        model = EnyoNNUE(init=args.init).to(args.device)
    forward_model = maybe_compile_model(model, args)

    rank_loader = DataLoader(
        rank_set, batch_size=args.child_batch_size, shuffle=True,
        collate_fn=collate_rank_pairs, **dataloader_kwargs(args))
    broad_loader = DataLoader(
        broad_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=broad_collate, **dataloader_kwargs(args))
    rank_iter = cycle(rank_loader)

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        rank_correct = 0
        rank_n = 0
        rank_gap_sum = 0.0
        rank_loss_sum = 0.0
        broad_excess_sum = 0.0
        broad_n = 0

        for broad_batch in broad_loader:
            broad_batch = move_batch(broad_batch, args.device)
            w, b, w_off, b_off, stm, y, _wdl, phase_scale, _source_ids = broad_batch
            if args.target_clamp > 0:
                y = torch.clamp(y, -args.target_clamp, args.target_clamp)

            rank_batch = move_batch(next(rank_iter), args.device)
            (bw, bb, bwo, bbo, bstm, bphase,
             ow, ob, owo, obo, ostm, ophase,
             target_gap, margin) = rank_batch

            with autocast_context(args):
                broad_pred = forward_model(w, b, w_off, b_off, stm, phase_scale)
                broad_abs = (broad_pred.float() - y).abs()
                broad_excess = torch.relu(broad_abs - args.broad_deadzone_cp)
                broad_loss = F.smooth_l1_loss(
                    broad_excess, torch.zeros_like(broad_excess),
                    beta=args.broad_beta, reduction="mean")

                best_pred = forward_model(bw, bb, bwo, bbo, bstm, bphase)
                other_pred = forward_model(ow, ob, owo, obo, ostm, ophase)
                pred_margin = best_pred.float() - other_pred.float()
                rank_loss = F.softplus((margin - pred_margin) / args.rank_temperature_cp).mean()
                loss = (args.ranking_weight * rank_loss
                        + args.broad_preserve_weight * broad_loss)

            opt.zero_grad()
            loss.backward()
            opt.step()

            rank_correct += int((pred_margin.detach() > 0).sum())
            rank_n += len(target_gap)
            rank_gap_sum += float(target_gap.sum())
            rank_loss_sum += float(rank_loss.detach()) * len(target_gap)
            broad_excess_sum += float(broad_excess.detach().sum())
            broad_n += len(y)

        print(
            f"epoch {epoch:4d}"
            f" rank_correct={rank_correct}/{max(1, rank_n)}"
            f" rank_loss={rank_loss_sum / max(1, rank_n):.6f}"
            f" target_gap={rank_gap_sum / max(1, rank_n):.2f}"
            f" broad_excess={broad_excess_sum / max(1, broad_n):.2f}",
            flush=True)

    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--child-targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-nn", default=None)
    ap.add_argument("--init-from-nn")
    ap.add_argument("--init", default="kaiming",
                    choices=["kaiming", "berserk-ish"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--child-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--target-clamp", type=float, default=1600.0)
    ap.add_argument("--ranking-weight", type=float, default=1.0)
    ap.add_argument("--broad-preserve-weight", type=float, default=0.1)
    ap.add_argument("--broad-deadzone-cp", type=float, default=40.0)
    ap.add_argument("--broad-beta", type=float, default=100.0)
    ap.add_argument("--rank-margin-cp", type=float, default=100.0)
    ap.add_argument("--rank-temperature-cp", type=float, default=50.0)
    ap.add_argument("--min-groups", type=int, default=1)
    ap.add_argument("--min-pairs", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--amp", default="off", choices=["off", "bf16"])
    ap.add_argument("--torch-compile", default=False,
                    action=argparse.BooleanOptionalAction)
    ap.add_argument("--dataset-in-memory", default=False,
                    action=argparse.BooleanOptionalAction)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--skip-rows", type=int, default=0)
    args = ap.parse_args()

    model = train(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), out)
    print(f"wrote {out}", flush=True)
    if args.out_nn:
        export_model(model, args.out_nn)
        print(f"wrote {args.out_nn}", flush=True)


if __name__ == "__main__":
    main()
