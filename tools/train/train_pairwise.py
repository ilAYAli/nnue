from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import chess
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

    @staticmethod
    def child_fen(parent_fen: str, move_uci: str) -> str:
        board = chess.Board(parent_fen)
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            raise ValueError(f"illegal move {move_uci} in {parent_fen}")
        board.push(move)
        return board.fen()

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

    @classmethod
    def from_sprt_scores_csv(cls, path: str | Path, *, min_target_margin: float,
                             max_target_margin: float) -> "HardPairDataset":
        by_target: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                by_target[(row["log"], row["ply"])].append(row)

        dataset = cls.__new__(cls)
        dataset.items = []
        for rows in by_target.values():
            best = next((row for row in rows if int(row["rank"]) == 1), None)
            if best is None:
                continue
            candidate_move = rows[0]["candidate_move"]
            played = next((row for row in rows if row["move"] == candidate_move), None)
            if played is None:
                continue

            # scores.csv stores child scores from the parent/root side POV.
            # After a legal move, side-to-move flips, so the child-side target
            # margin is the candidate gap over the oracle-best child.
            target_margin = float(played["gap_cp"])
            if target_margin < min_target_margin:
                continue
            if max_target_margin > 0:
                target_margin = min(target_margin, max_target_margin)

            parent_fen = played["fen"]
            played_fen = cls.child_fen(parent_fen, played["move"])
            best_fen = cls.child_fen(parent_fen, best["move"])
            dataset.items.append((
                fen_features(played_fen),
                fen_features(best_fen),
                target_margin,
                float(played["gap_cp"]),
            ))
        return dataset

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


def train(args) -> EnyoNNUE:
    broad_set, broad_collate = load_score_dataset(
        args.data, limit=args.max_rows, skip=args.skip_rows)
    if args.scores_csv:
        pair_set = HardPairDataset.from_sprt_scores_csv(
            args.scores_csv,
            min_target_margin=args.min_target_margin,
            max_target_margin=args.max_target_margin)
    else:
        pair_set = HardPairDataset.from_jsonl(
            args.pairs,
            min_target_margin=args.min_target_margin,
            max_target_margin=args.max_target_margin)
    if len(pair_set) == 0:
        raise SystemExit("no hard pairs after filtering")

    print(f"broad rows: {len(broad_set)}", flush=True)
    print(f"hard pairs: {len(pair_set)}", flush=True)

    if args.init_from_nn:
        model = load_model_from_nn(args.init_from_nn, device=args.device)
    else:
        model = EnyoNNUE(init=args.init).to(args.device)

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

    for epoch in range(args.epochs):
        broad_mae_sum = 0.0
        broad_n = 0
        pair_mae_sum = 0.0
        pair_correct = 0
        pair_n = 0
        for broad_batch in broad_loader:
            broad_batch = to_device(broad_batch, args.device)
            w, b, w_off, b_off, stm, y, _wdl, phase_scale, _source_ids = broad_batch
            if args.target_clamp > 0:
                y = torch.clamp(y, -args.target_clamp, args.target_clamp)
            pred = model(w, b, w_off, b_off, stm, phase_scale)
            broad_loss = F.smooth_l1_loss(
                pred, y, beta=args.huber_beta, reduction="mean")

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

            loss = broad_loss + args.pair_weight * pair_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            broad_err = (pred.detach() - y).abs()
            broad_mae_sum += float(broad_err.sum())
            broad_n += len(y)
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

    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pairs", default="")
    ap.add_argument("--scores-csv", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-nn", default=None)
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
    ap.add_argument("--target-clamp", type=float, default=1600.0)
    ap.add_argument("--max-target-margin", type=float, default=800.0)
    ap.add_argument("--min-target-margin", type=float, default=1.0)
    ap.add_argument("--loss-weight-by-cp", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--skip-rows", type=int, default=0)
    args = ap.parse_args()
    if bool(args.pairs) == bool(args.scores_csv):
        raise SystemExit("provide exactly one of --pairs or --scores-csv")

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
