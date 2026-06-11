from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import enyo_nnue as nn2
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


class HardPairDataset(Dataset):
    def __init__(self, rows: list[dict], *, min_target_margin: float,
                 max_target_margin: float):
        grouped: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
        for row in rows:
            key = (
                row["hard_parent_fen"],
                row["hard_parent_played"],
                row["hard_parent_best"],
            )
            grouped[key][row["hard_child_kind"]] = row

        self.items = []
        for pair in grouped.values():
            if "played" not in pair or "best" not in pair:
                continue
            played = pair["played"]
            best = pair["best"]
            # Child scores are from child side-to-move POV. The parent mover
            # prefers the best child when played_score - best_score is positive.
            target_margin = float(played["score"]) - float(best["score"])
            if target_margin < min_target_margin:
                continue
            if max_target_margin > 0:
                target_margin = min(target_margin, max_target_margin)
            self.items.append((
                fen_features(played["fen"]),
                fen_features(best["fen"]),
                target_margin,
                float(played.get("hard_parent_loss_cp", target_margin)),
            ))

    @classmethod
    def from_jsonl(cls, path: str | Path, *, min_target_margin: float,
                   max_target_margin: float) -> "HardPairDataset":
        rows = []
        with Path(path).open() as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return cls(rows, min_target_margin=min_target_margin,
                   max_target_margin=max_target_margin)

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


def collate_pairs(batch):
    played_rows = [item[0] for item in batch]
    best_rows = [item[1] for item in batch]
    margins = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    weights = torch.tensor([item[3] for item in batch], dtype=torch.float32)
    return (*collate_feature_rows(played_rows),
            *collate_feature_rows(best_rows),
            margins, weights)


def cycle(loader):
    while True:
        yield from loader


def to_device(items, device: str):
    return [item.to(device) if torch.is_tensor(item) else item
            for item in items]


@torch.no_grad()
def project_exportable_parameters(model: EnyoNNUE) -> None:
    model.embed.weight.round_().clamp_(
        torch.iinfo(torch.int16).min, torch.iinfo(torch.int16).max)
    model.input_bias.round_().clamp_(
        torch.iinfo(torch.int16).min, torch.iinfo(torch.int16).max)
    model.l1_weight.round_().clamp_(
        torch.iinfo(torch.int8).min, torch.iinfo(torch.int8).max)
    model.l1_bias.round_().clamp_(
        torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max)


@torch.no_grad()
def pair_metrics(model: EnyoNNUE, loader: DataLoader, args) -> dict[str, float]:
    model.eval()
    n = 0
    abs_sum = 0.0
    correct = 0
    pred_sum = 0.0
    target_sum = 0.0
    for batch in loader:
        batch = to_device(batch, args.device)
        (pw, pb, pwo, pbo, pstm, pphase,
         bw, bb, bwo, bbo, bstm, bphase,
         target_margin, _weights) = batch
        pred_played = model(pw, pb, pwo, pbo, pstm, pphase)
        pred_best = model(bw, bb, bwo, bbo, bstm, bphase)
        pred_margin = pred_played - pred_best
        err = pred_margin - target_margin
        n += len(target_margin)
        abs_sum += float(err.abs().sum())
        correct += int((pred_margin > 0).sum())
        pred_sum += float(pred_margin.sum())
        target_sum += float(target_margin.sum())
    model.train()
    return {
        "pair_mae": abs_sum / max(1, n),
        "pair_correct": correct / max(1, n),
        "pred_margin": pred_sum / max(1, n),
        "target_margin": target_sum / max(1, n),
    }


def write_metadata(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def pairwise_metadata(args, *, epoch: int | None = None) -> dict:
    label_refs = ["enyo"]
    source_map = Path(args.data) / "source_map.json"
    if source_map.exists():
        try:
            raw = json.loads(source_map.read_text())
            if isinstance(raw, dict):
                label_refs.extend(str(name) for name in raw)
        except (OSError, json.JSONDecodeError):
            pass
    metadata = {
        "schema": "enyo.train_pairwise.v1",
        "kind": "pairwise_nnue_checkpoint",
        "data": str(args.data),
        "pairs": str(args.pairs),
        "position_refs": [str(args.data), str(args.pairs)],
        "label_refs": sorted(set(label_refs)),
        "init_from_nn": args.init_from_nn,
        "init": None if args.init_from_nn else args.init,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "pair_batch_size": args.pair_batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "huber_beta": args.huber_beta,
        "pair_beta": args.pair_beta,
        "pair_weight": args.pair_weight,
        "broad_weight": args.broad_weight,
        "target_clamp": args.target_clamp,
        "max_target_margin": args.max_target_margin,
        "min_target_margin": args.min_target_margin,
        "loss_weight_by_cp": bool(args.loss_weight_by_cp),
        "project_export_weights_each_step": bool(
            args.project_export_weights_each_step),
        "max_rows": args.max_rows,
        "skip_rows": args.skip_rows,
    }
    if epoch is not None:
        metadata["epoch"] = epoch
    return metadata


def save_model(model: EnyoNNUE, out: Path, out_nn: Path | None, device: str,
               metadata: dict | None = None
               ) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), out)
    print(f"wrote {out}", flush=True)
    if metadata is not None:
        write_metadata(out.with_suffix(".meta.json"), metadata)
    if out_nn is not None:
        out_nn.parent.mkdir(parents=True, exist_ok=True)
        export_model(model, out_nn)
        print(f"wrote {out_nn}", flush=True)
        if metadata is not None:
            write_metadata(out_nn.with_suffix(".meta.json"), metadata)
    model.to(device)


def maybe_save_checkpoint(model: EnyoNNUE, epoch: int, args) -> None:
    if not args.checkpoint_dir:
        return
    if args.checkpoint_every <= 0:
        return
    if (epoch + 1) % args.checkpoint_every != 0:
        return
    checkpoint_dir = Path(args.checkpoint_dir)
    stem = f"epoch-{epoch:04d}"
    save_model(
        model,
        checkpoint_dir / f"{stem}.pt",
        checkpoint_dir / f"{stem}.nn",
        args.device,
        pairwise_metadata(args, epoch=epoch))


def train(args) -> EnyoNNUE:
    use_broad = args.broad_weight > 0.0
    if use_broad:
        broad_set, broad_collate = load_score_dataset(
            args.data, limit=args.max_rows, skip=args.skip_rows)
    else:
        broad_set = None
        broad_collate = None
    pair_set = HardPairDataset.from_jsonl(
        args.pairs,
        min_target_margin=args.min_target_margin,
        max_target_margin=args.max_target_margin)
    if len(pair_set) == 0:
        raise SystemExit("no hard pairs after filtering")

    if use_broad:
        print(f"broad rows: {len(broad_set)}", flush=True)
    else:
        print("broad rows: skipped", flush=True)
    print(f"hard pairs: {len(pair_set)}", flush=True)

    if args.init_from_nn:
        model = load_model_from_nn(args.init_from_nn, device=args.device)
    else:
        model = EnyoNNUE(init=args.init).to(args.device)

    broad_loader = None
    if use_broad:
        broad_loader = DataLoader(
            broad_set, batch_size=args.batch_size, shuffle=True,
            collate_fn=broad_collate, num_workers=args.workers,
            pin_memory=args.device.startswith("cuda"))
    pair_loader = DataLoader(
        pair_set, batch_size=args.pair_batch_size, shuffle=True,
        collate_fn=collate_pairs, num_workers=0,
        pin_memory=args.device.startswith("cuda"))
    pair_iter = cycle(pair_loader)

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.project_export_weights_each_step:
        project_exportable_parameters(model)

    for epoch in range(args.epochs):
        broad_mae_sum = 0.0
        broad_n = 0
        pair_mae_sum = 0.0
        pair_correct = 0
        pair_n = 0
        steps = (
            len(broad_loader) if use_broad
            else args.steps_per_epoch if args.steps_per_epoch > 0
            else max(1, len(pair_loader))
        )
        broad_iter = iter(broad_loader) if use_broad else None
        for _step in range(steps):
            broad_loss = torch.zeros((), device=args.device)
            if use_broad:
                assert broad_iter is not None
                broad_batch = to_device(next(broad_iter), args.device)
                w, b, w_off, b_off, _counts, stm, y, _wdl, phase_scale, _source_ids = broad_batch
                if args.target_clamp > 0:
                    y = torch.clamp(y, -args.target_clamp, args.target_clamp)
                pred = model(w, b, w_off, b_off, stm, phase_scale)
                broad_loss = F.smooth_l1_loss(
                    pred, y, beta=args.huber_beta, reduction="mean")

                broad_err = (pred.detach() - y).abs()
                broad_mae_sum += float(broad_err.sum())
                broad_n += len(y)

            pair_batch = to_device(next(pair_iter), args.device)
            (pw, pb, pwo, pbo, pstm, pphase,
             bw, bb, bwo, bbo, bstm, bphase,
             target_margin, weights) = pair_batch
            pred_played = model(pw, pb, pwo, pbo, pstm, pphase)
            pred_best = model(bw, bb, bwo, bbo, bstm, bphase)
            pred_margin = pred_played - pred_best
            pair_losses = F.smooth_l1_loss(
                pred_margin, target_margin,
                beta=args.pair_beta, reduction="none")
            if args.loss_weight_by_cp:
                norm_weights = weights / weights.mean().clamp_min(1.0)
                pair_loss = (pair_losses * norm_weights).mean()
            else:
                pair_loss = pair_losses.mean()

            loss = args.broad_weight * broad_loss + args.pair_weight * pair_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            if args.project_export_weights_each_step:
                project_exportable_parameters(model)

            pair_err = (pred_margin.detach() - target_margin).abs()
            pair_mae_sum += float(pair_err.sum())
            pair_correct += int((pred_margin.detach() > 0).sum())
            pair_n += len(target_margin)

        metrics = pair_metrics(model, pair_loader, args)
        print(
            f"epoch {epoch:4d}"
            f" broad_mae={broad_mae_sum / max(1, broad_n):7.2f}"
            f" pair_batch_mae={pair_mae_sum / max(1, pair_n):7.2f}"
            f" pair_batch_correct={100.0 * pair_correct / max(1, pair_n):5.1f}%"
            f" pair_mae={metrics['pair_mae']:7.2f}"
            f" pair_correct={100.0 * metrics['pair_correct']:5.1f}%"
            f" pred_margin={metrics['pred_margin']:7.2f}"
            f" target_margin={metrics['target_margin']:7.2f}",
            flush=True)
        maybe_save_checkpoint(model, epoch, args)

    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-nn", default=None)
    ap.add_argument("--checkpoint-dir", default=None,
                    help="Optional directory for per-epoch .pt and .nn files.")
    ap.add_argument("--checkpoint-every", type=int, default=1,
                    help="Save every N epochs when --checkpoint-dir is set.")
    ap.add_argument("--init-from-nn", default=None)
    ap.add_argument("--init", default="kaiming",
                    choices=["kaiming", "berserk-ish"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--pair-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--huber-beta", type=float, default=200.0)
    ap.add_argument("--pair-beta", type=float, default=100.0)
    ap.add_argument("--pair-weight", type=float, default=1.0)
    ap.add_argument("--broad-weight", type=float, default=1.0)
    ap.add_argument("--steps-per-epoch", type=int, default=0,
                    help="Target-only steps per epoch when --broad-weight=0.")
    ap.add_argument("--target-clamp", type=float, default=1600.0)
    ap.add_argument("--max-target-margin", type=float, default=800.0)
    ap.add_argument("--min-target-margin", type=float, default=1.0)
    ap.add_argument("--loss-weight-by-cp", action="store_true")
    ap.add_argument("--project-export-weights-each-step", action="store_true",
                    help="Round/clamp int-quantized weights after each optimizer step.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--skip-rows", type=int, default=0)
    args = ap.parse_args()

    model = train(args)
    save_model(
        model,
        Path(args.out),
        Path(args.out_nn) if args.out_nn else None,
        args.device,
        pairwise_metadata(args))
    if args.out_nn:
        pair_set = HardPairDataset.from_jsonl(
            args.pairs,
            min_target_margin=args.min_target_margin,
            max_target_margin=args.max_target_margin)
        pair_loader = DataLoader(
            pair_set, batch_size=args.pair_batch_size, shuffle=False,
            collate_fn=collate_pairs, num_workers=0,
            pin_memory=args.device.startswith("cuda"))
        reloaded = load_model_from_nn(args.out_nn, device=args.device)
        metrics = pair_metrics(reloaded, pair_loader, args)
        print(
            "export_reload"
            f" pair_mae={metrics['pair_mae']:7.2f}"
            f" pair_correct={100.0 * metrics['pair_correct']:5.1f}%"
            f" pred_margin={metrics['pred_margin']:7.2f}"
            f" target_margin={metrics['target_margin']:7.2f}",
            flush=True)


if __name__ == "__main__":
    main()
