"""Train TokenSelectionModel.

Usage:
  python -m training.train_token \
    --dataset labeling_out_full/token_selection_dataset.jsonl \
    --pools   labeling_out_full/package_pools.json \
    --run-log labeling_out_full/labeling_run_log.json \
    --output  training/checkpoints
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import partial
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Allow `python -m training.train_token` and direct invocation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset import TokenSelectionDataset, token_collate, worker_init_fn
from training.models import TokenSelectionFlat, TokenSelectionModel
from training.utils import PoolRegistry, load_split, seed_all


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="token_selection_dataset.jsonl")
    ap.add_argument("--pools", required=True, help="package_pools.json")
    ap.add_argument("--run-log", required=True, help="labeling_run_log.json")
    ap.add_argument("--output", default="training/checkpoints")
    ap.add_argument("--tb", default="training/tb_logs/token")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32, help="measures per batch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=5, help="early stopping patience (epochs)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def top_k_recall_per_measure(
    scores: torch.Tensor,          # (sum_P,)
    targets: torch.Tensor,         # (sum_P,)
    label_mask: torch.Tensor,      # (sum_P,)
    measure_id: torch.Tensor,      # (sum_P,)
    notes_in_measure: torch.Tensor,  # (M,)
) -> float:
    """Average top-K recall where K = notes_in_measure for each measure.

    For each measure, rank labeled tokens by score descending, take top-K,
    and compute (# of label=1 tokens in top-K) / (# of label=1 tokens total).
    """
    M = int(notes_in_measure.shape[0])
    total = 0.0
    counted = 0
    for m in range(M):
        sel = measure_id == m
        m_scores = scores[sel]
        m_targets = targets[sel]
        m_mask = label_mask[sel]
        # Only consider labeled tokens for the ranking
        labeled_idx = (m_mask > 0).nonzero(as_tuple=True)[0]
        if labeled_idx.numel() == 0:
            continue
        labeled_scores = m_scores[labeled_idx]
        labeled_targets = m_targets[labeled_idx]
        positive_count = int(labeled_targets.sum().item())
        if positive_count == 0:
            continue
        K = int(min(notes_in_measure[m].item(), labeled_idx.numel()))
        if K == 0:
            continue
        topk = torch.topk(labeled_scores, k=K).indices
        hit = int(labeled_targets[topk].sum().item())
        total += hit / positive_count
        counted += 1
    return total / counted if counted > 0 else 0.0


def run_epoch(
    model: TokenSelectionFlat,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    train: bool,
    writer: SummaryWriter,
    epoch: int,
    tag: str,
) -> Dict[str, float]:
    model.train(train)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    total_loss = 0.0
    total_rows = 0
    recall_sum = 0.0
    recall_batches = 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        measure = batch["measure"].to(device, non_blocking=True)
        pool = batch["pool"].to(device, non_blocking=True)
        context = batch["context"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        label_mask = batch["label_mask"].to(device, non_blocking=True)
        measure_id = batch["measure_id"]
        notes_in_measure = batch["notes_in_measure"]

        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(measure, pool, context)
            raw_loss = bce(logits, targets)
            masked = raw_loss * label_mask
            denom = label_mask.sum().clamp(min=1.0)
            loss = masked.sum() / denom
        if train:
            loss.backward()
            optimizer.step()

        n = int(label_mask.sum().item())
        total_loss += float(loss.item()) * n
        total_rows += n

        with torch.no_grad():
            r = top_k_recall_per_measure(
                logits.detach().cpu(),
                targets.detach().cpu(),
                label_mask.detach().cpu(),
                measure_id,
                notes_in_measure,
            )
        recall_sum += r
        recall_batches += 1

        if train and step % 100 == 0:
            gs = epoch * len(loader) + step
            writer.add_scalar(f"{tag}/step_loss", float(loss.item()), gs)
            writer.add_scalar(f"{tag}/step_recall", r, gs)

    elapsed = time.time() - t0
    avg_loss = total_loss / max(1, total_rows)
    avg_recall = recall_sum / max(1, recall_batches)
    return {
        "loss": avg_loss,
        "top_k_recall": avg_recall,
        "elapsed_sec": elapsed,
    }


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"device: {device}")

    print("loading pool registry ...")
    pools = PoolRegistry(args.pools)
    print(f"  {len(pools)} packages")

    print("loading train/val split from run log ...")
    train_pkgs, val_pkgs = load_split(args.run_log)
    print(f"  train={len(train_pkgs)} val={len(val_pkgs)}")

    offset_cache = args.dataset + ".offsets.npy"
    train_split_cache = args.dataset + ".train_split.npy"
    val_split_cache = args.dataset + ".val_split.npy"

    print("building / loading datasets ...")
    train_ds = TokenSelectionDataset(
        args.dataset, offset_cache,
        allowed_packages=train_pkgs,
        split_cache_path=train_split_cache,
    )
    val_ds = TokenSelectionDataset(
        args.dataset, offset_cache,
        allowed_packages=val_pkgs,
        split_cache_path=val_split_cache,
    )
    print(f"  train records={len(train_ds)} val records={len(val_ds)}")

    collate = partial(token_collate, pools=pools)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate,
        worker_init_fn=worker_init_fn, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
        worker_init_fn=worker_init_fn, pin_memory=(device.type == "cuda"),
    )

    core = TokenSelectionModel(hidden=args.hidden, dropout=args.dropout).to(device)
    model = TokenSelectionFlat(core).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.output, exist_ok=True)
    writer = SummaryWriter(args.tb)

    best_recall = -1.0
    best_epoch = -1
    patience_left = args.patience
    history = []

    for epoch in range(args.epochs):
        print(f"epoch {epoch+1}/{args.epochs}")
        tr = run_epoch(model, train_loader, optimizer, device, True, writer, epoch, "train")
        va = run_epoch(model, val_loader, optimizer, device, False, writer, epoch, "val")
        print(f"  train loss={tr['loss']:.4f} recall={tr['top_k_recall']:.4f} ({tr['elapsed_sec']:.1f}s)")
        print(f"  val   loss={va['loss']:.4f} recall={va['top_k_recall']:.4f} ({va['elapsed_sec']:.1f}s)")
        writer.add_scalar("epoch/train_loss", tr["loss"], epoch)
        writer.add_scalar("epoch/train_recall", tr["top_k_recall"], epoch)
        writer.add_scalar("epoch/val_loss", va["loss"], epoch)
        writer.add_scalar("epoch/val_recall", va["top_k_recall"], epoch)
        history.append({"epoch": epoch, "train": tr, "val": va})

        ckpt_path = os.path.join(args.output, f"token_epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": core.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train": tr, "val": va,
            "args": vars(args),
        }, ckpt_path)

        if va["top_k_recall"] > best_recall:
            best_recall = va["top_k_recall"]
            best_epoch = epoch
            best_path = os.path.join(args.output, "token_best.pt")
            torch.save({"epoch": epoch, "model_state": core.state_dict(), "val": va, "args": vars(args)}, best_path)
            patience_left = args.patience
            print(f"  new best val recall={best_recall:.4f} → {best_path}")
        else:
            patience_left -= 1
            print(f"  no improvement (patience left {patience_left})")
            if patience_left <= 0:
                print("early stopping")
                break

    writer.close()

    # Export best checkpoint as TorchScript
    print("exporting best checkpoint → token_selection_model.pt")
    best = torch.load(os.path.join(args.output, "token_best.pt"), map_location="cpu")
    export_core = TokenSelectionModel(hidden=args.hidden, dropout=args.dropout)
    export_core.load_state_dict(best["model_state"])
    export_core.eval()
    scripted = torch.jit.script(export_core)
    export_path = os.path.join(args.output, "token_selection_model.pt")
    scripted.save(export_path)
    print(f"saved {export_path}")

    with open(os.path.join(args.output, "token_history.json"), "w", encoding="utf-8") as f:
        json.dump({"best_epoch": best_epoch, "best_recall": best_recall, "history": history}, f, indent=2)


if __name__ == "__main__":
    main()
