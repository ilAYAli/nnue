#!/usr/bin/env python3
"""Train Enyo NNUE with broad scalar loss plus search-aware move targets."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import chess
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import enyo_nnue as nn2
from lib.nnue_dataset import load_score_dataset
from lib.nnue_model import EnyoNNUE, export_model, load_model_from_nn


@dataclass
class ChildMove:
    uci: str
    rank: int
    gap_cp: float
    policy: float
    features: tuple[list[int], list[int], int, float]


def fen_features(fen: str) -> tuple[list[int], list[int], int, float]:
    pieces, stm = nn2.parse_fen(fen)
    pieces.sort(key=lambda item: item[2])
    return (
        nn2.features_from_pieces(pieces, nn2.WHITE),
        nn2.features_from_pieces(pieces, nn2.BLACK),
        stm,
        nn2.phase_scale_from_pieces(pieces),
    )


class SearchAwareTargetDataset(Dataset):
    def __init__(self, rows: list[dict], *, max_moves: int,
                 max_gap_cp: float) -> None:
        self.items: list[tuple[str, list[str], list[ChildMove]]] = []
        for row in rows:
            fen = str(row["fen"])
            tags = [str(tag) for tag in row.get("tags", [])]
            board = chess.Board(fen)
            children: list[ChildMove] = []
            moves = sorted(row.get("moves", []), key=lambda item: int(item.get("rank", 999)))
            if max_moves > 0:
                moves = moves[:max_moves]
            for item in moves:
                uci = str(item.get("uci", "")).lower()
                if not uci:
                    continue
                try:
                    move = chess.Move.from_uci(uci)
                except ValueError:
                    continue
                if move not in board.legal_moves:
                    continue
                child = board.copy(stack=False)
                child.push(move)
                gap = float(item.get("gap_cp", 0.0))
                if max_gap_cp > 0:
                    gap = min(gap, max_gap_cp)
                children.append(ChildMove(
                    uci=uci,
                    rank=int(item.get("rank", 999)),
                    gap_cp=gap,
                    policy=float(item.get("policy", 0.0)),
                    features=fen_features(child.fen()),
                ))
            if not children:
                continue
            if not any(child.rank == 1 for child in children):
                children[0].rank = 1
                children[0].gap_cp = 0.0
            policy_sum = sum(max(0.0, child.policy) for child in children)
            if policy_sum <= 0.0:
                for child in children:
                    child.policy = 1.0 if child.rank == 1 else 0.0
            else:
                for child in children:
                    child.policy = max(0.0, child.policy) / policy_sum
            self.items.append((str(row.get("id", "")), tags, children))

    @classmethod
    def from_jsonl(cls, path: str | Path, *, limit: int, max_moves: int,
                   max_gap_cp: float) -> "SearchAwareTargetDataset":
        rows = []
        with Path(path).expanduser().open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
                if limit > 0 and len(rows) >= limit:
                    break
        return cls(rows, max_moves=max_moves, max_gap_cp=max_gap_cp)

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


def collate_targets(batch):
    features = []
    group_offsets = [0]
    best_indices = []
    gaps = []
    policies = []
    tag_sets = []
    ids = []
    moves = []
    for target_id, tags, children in batch:
        ids.append(target_id)
        tag_sets.append(";".join(tags))
        start = group_offsets[-1]
        best_local = 0
        for i, child in enumerate(children):
            features.append(child.features)
            gaps.append(float(child.gap_cp))
            policies.append(float(child.policy))
            moves.append(child.uci)
            if child.rank == 1:
                best_local = i
        best_indices.append(start + best_local)
        group_offsets.append(start + len(children))
    return (
        *collate_feature_rows(features),
        torch.tensor(group_offsets, dtype=torch.long),
        torch.tensor(best_indices, dtype=torch.long),
        torch.tensor(gaps, dtype=torch.float32),
        torch.tensor(policies, dtype=torch.float32),
        ids,
        tag_sets,
        moves,
    )


def cycle(loader):
    while True:
        yield from loader


def to_device(items, device: str):
    return [item.to(device) if torch.is_tensor(item) else item for item in items]


def predict(model: EnyoNNUE, args, w_feats: torch.Tensor, b_feats: torch.Tensor,
            w_offsets: torch.Tensor, b_offsets: torch.Tensor,
            stm: torch.Tensor, phase_scale: torch.Tensor) -> torch.Tensor:
    if args.forward == "quantized":
        return model.quantized_forward(w_feats, b_feats, w_offsets, b_offsets, stm, phase_scale)
    return model(w_feats, b_feats, w_offsets, b_offsets, stm, phase_scale)


def grad_norm(param: torch.Tensor) -> float:
    if param.grad is None:
        return 0.0
    return float(param.grad.detach().norm())


def optimizer_param_groups(model: EnyoNNUE, args: argparse.Namespace):
    groups = [
        {"name": "input", "lr": args.lr * args.input_lr_mult,
         "params": [model.embed.weight, model.input_bias]},
        {"name": "l1", "lr": args.lr * args.l1_lr_mult,
         "params": [model.l1_weight, model.l1_bias]},
        {"name": "dense", "lr": args.lr * args.dense_lr_mult,
         "params": list(model.l2.parameters()) + list(model.output.parameters())},
    ]
    out = []
    for group in groups:
        params = [param for param in group["params"] if param.requires_grad]
        if params:
            out.append({"params": params, "lr": group["lr"], "name": group["name"]})
    return out


def target_loss_and_metrics(pred: torch.Tensor, group_offsets: torch.Tensor,
                            best_indices: torch.Tensor, gaps: torch.Tensor,
                            policies: torch.Tensor, args: argparse.Namespace):
    margin_losses = []
    policy_losses = []
    chosen_gaps = []
    top1 = 0
    top3 = 0
    for i in range(len(best_indices)):
        start = int(group_offsets[i])
        end = int(group_offsets[i + 1])
        group_pred = pred[start:end]
        group_gaps = gaps[start:end]
        group_policy = policies[start:end]
        best = int(best_indices[i]) - start
        best_pred = group_pred[best]
        margin_losses.append(F.smooth_l1_loss(
            group_pred - best_pred,
            group_gaps,
            beta=args.search_margin_beta,
            reduction="mean"))
        logits = -group_pred / max(args.search_policy_temperature_cp, 1e-6)
        policy_losses.append(-(group_policy * F.log_softmax(logits, dim=0)).sum())
        order = torch.argsort(group_pred)
        selected = int(order[0])
        if selected == best:
            top1 += 1
        if best in {int(v) for v in order[:min(3, len(order))]}:
            top3 += 1
        chosen_gaps.append(float(group_gaps[selected].detach().cpu()))
    margin_loss = torch.stack(margin_losses).mean() if margin_losses else pred.sum() * 0.0
    policy_loss = torch.stack(policy_losses).mean() if policy_losses else pred.sum() * 0.0
    metrics = {
        "top1": top1,
        "top3": top3,
        "targets": len(best_indices),
        "sum_gap": sum(chosen_gaps),
        "median_gap": float(statistics.median(chosen_gaps)) if chosen_gaps else 0.0,
        "worst_gap": max(chosen_gaps) if chosen_gaps else 0.0,
    }
    return margin_loss, policy_loss, metrics


@torch.no_grad()
def search_metrics(model: EnyoNNUE, loader: DataLoader, args: argparse.Namespace) -> dict[str, float]:
    model.eval()
    total = 0
    top1 = 0
    top3 = 0
    sum_gap = 0.0
    worst_gap = 0.0
    med_values = []
    for batch in loader:
        tensors = to_device(batch[:8], args.device)
        w, b, w_off, b_off, stm, phase_scale, group_offsets, best_indices = tensors
        gaps = batch[8].to(args.device)
        policies = batch[9].to(args.device)
        pred = predict(model, args, w, b, w_off, b_off, stm, phase_scale)
        _margin, _policy, metrics = target_loss_and_metrics(
            pred, group_offsets, best_indices, gaps, policies, args)
        total += metrics["targets"]
        top1 += metrics["top1"]
        top3 += metrics["top3"]
        sum_gap += metrics["sum_gap"]
        worst_gap = max(worst_gap, metrics["worst_gap"])
        med_values.append(metrics["median_gap"])
    model.train()
    return {
        "targets": float(total),
        "top1": float(top1),
        "top3": float(top3),
        "sum_gap": sum_gap,
        "worst_gap": worst_gap,
        "median_gap": float(statistics.median(med_values)) if med_values else 0.0,
    }


def set_trainable(model: EnyoNNUE, trainable: str) -> None:
    if trainable == "all":
        return
    for param in model.parameters():
        param.requires_grad_(False)
    if trainable == "input":
        model.embed.weight.requires_grad_(True)
        model.input_bias.requires_grad_(True)
    if trainable in {"float-head", "output"}:
        for param in model.output.parameters():
            param.requires_grad_(True)
    if trainable == "float-head":
        for param in model.l2.parameters():
            param.requires_grad_(True)


def train(args: argparse.Namespace) -> EnyoNNUE:
    broad_set, broad_collate = load_score_dataset(
        args.data, limit=args.max_rows, skip=args.skip_rows)
    target_set = SearchAwareTargetDataset.from_jsonl(
        args.targets,
        limit=args.search_target_limit,
        max_moves=args.search_max_moves,
        max_gap_cp=args.search_max_gap_cp)
    if len(target_set) == 0:
        raise SystemExit("no search-aware targets after filtering")
    print(f"broad rows: {len(broad_set)}", flush=True)
    print(f"search targets: {len(target_set)}", flush=True)

    if args.init_from_nn:
        model = load_model_from_nn(args.init_from_nn, device=args.device)
    else:
        model = EnyoNNUE(init=args.init).to(args.device)
    init_model = None
    if args.search_broad_target == "init":
        if not args.init_from_nn:
            raise SystemExit("--search-broad-target init requires --init-from-nn")
        init_model = load_model_from_nn(args.init_from_nn, device=args.device)
        init_model.eval()
        for param in init_model.parameters():
            param.requires_grad_(False)
        print("broad target: init net distillation", flush=True)
    else:
        print("broad target: teacher labels", flush=True)

    set_trainable(model, args.trainable)
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"trainable={args.trainable} params={trainable_params}", flush=True)

    broad_loader = DataLoader(
        broad_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=broad_collate, num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"))
    target_loader = DataLoader(
        target_set, batch_size=args.search_target_batch_size, shuffle=True,
        collate_fn=collate_targets, num_workers=0,
        pin_memory=args.device.startswith("cuda"))
    target_iter = cycle(target_loader)

    param_groups = optimizer_param_groups(model, args)
    for group in param_groups:
        print(f"optimizer group {group['name']} lr={group['lr']}", flush=True)
    opt = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        broad_mae_sum = 0.0
        broad_n = 0
        target_n = 0
        target_top1 = 0
        target_top3 = 0
        target_gap_sum = 0.0
        for batch_index, broad_batch in enumerate(broad_loader):
            broad_batch = to_device(broad_batch, args.device)
            w, b, w_off, b_off, stm, y, _wdl, phase_scale, _source_ids = broad_batch
            if init_model is not None:
                with torch.no_grad():
                    y = predict(init_model, args, w, b, w_off, b_off, stm, phase_scale)
            elif args.target_clamp > 0:
                y = torch.clamp(y, -args.target_clamp, args.target_clamp)
            pred = predict(model, args, w, b, w_off, b_off, stm, phase_scale)
            broad_loss = F.smooth_l1_loss(
                pred, y, beta=args.huber_beta, reduction="mean")

            target_batch = next(target_iter)
            tensors = to_device(target_batch[:8], args.device)
            tw, tb, two, tbo, tstm, tphase, group_offsets, best_indices = tensors
            gaps = target_batch[8].to(args.device)
            policies = target_batch[9].to(args.device)
            target_pred = predict(model, args, tw, tb, two, tbo, tstm, tphase)
            margin_loss, policy_loss, target_metrics = target_loss_and_metrics(
                target_pred, group_offsets, best_indices, gaps, policies, args)

            loss = (
                args.search_broad_weight * broad_loss
                + args.search_margin_weight * margin_loss
                + args.search_policy_weight * policy_loss)
            opt.zero_grad()
            loss.backward()
            if args.grad_norm_every > 0 and batch_index % args.grad_norm_every == 0:
                print(
                    "grad "
                    f"epoch={epoch} batch={batch_index} "
                    f"input={grad_norm(model.embed.weight):.6g} "
                    f"input_bias={grad_norm(model.input_bias):.6g} "
                    f"l1={grad_norm(model.l1_weight):.6g} "
                    f"l1_bias={grad_norm(model.l1_bias):.6g} "
                    f"l2={grad_norm(model.l2.weight):.6g} "
                    f"output={grad_norm(model.output.weight):.6g}",
                    flush=True)
            opt.step()

            broad_mae_sum += float((pred.detach() - y).abs().sum())
            broad_n += len(y)
            target_n += target_metrics["targets"]
            target_top1 += target_metrics["top1"]
            target_top3 += target_metrics["top3"]
            target_gap_sum += target_metrics["sum_gap"]

        metrics = search_metrics(model, target_loader, args)
        print(
            f"epoch {epoch:4d}"
            f" broad_mae={broad_mae_sum / max(1, broad_n):7.2f}"
            f" target_batch_top1={100.0 * target_top1 / max(1, target_n):5.1f}%"
            f" target_batch_top3={100.0 * target_top3 / max(1, target_n):5.1f}%"
            f" target_batch_gap={target_gap_sum / max(1, target_n):7.2f}"
            f" target_top1={int(metrics['top1'])}/{int(metrics['targets'])}"
            f" target_top3={int(metrics['top3'])}/{int(metrics['targets'])}"
            f" target_sum_gap={metrics['sum_gap']:.0f}"
            f" target_worst_gap={metrics['worst_gap']:.0f}",
            flush=True)

    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-nn", default=None)
    ap.add_argument("--init-from-nn", default=None)
    ap.add_argument("--init", default="kaiming",
                    choices=["kaiming", "quantized-kaiming", "berserk-ish"])
    ap.add_argument("--forward", default="float", choices=["float", "quantized"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--search-target-batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--input-lr-mult", type=float, default=1.0)
    ap.add_argument("--l1-lr-mult", type=float, default=1.0)
    ap.add_argument("--dense-lr-mult", type=float, default=1.0)
    ap.add_argument("--huber-beta", type=float, default=200.0)
    ap.add_argument("--target-clamp", type=float, default=800.0)
    ap.add_argument("--search-broad-weight", type=float, default=1.0)
    ap.add_argument("--search-broad-target", default="teacher", choices=["teacher", "init"])
    ap.add_argument("--search-margin-weight", type=float, default=1.0)
    ap.add_argument("--search-policy-weight", type=float, default=0.25)
    ap.add_argument("--search-margin-beta", type=float, default=100.0)
    ap.add_argument("--search-policy-temperature-cp", type=float, default=200.0)
    ap.add_argument("--search-max-gap-cp", type=float, default=800.0)
    ap.add_argument("--search-max-moves", type=int, default=0)
    ap.add_argument("--search-target-limit", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--skip-rows", type=int, default=0)
    ap.add_argument("--grad-norm-every", type=int, default=0)
    ap.add_argument("--trainable", default="all",
                    choices=["all", "input", "float-head", "output"])
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
