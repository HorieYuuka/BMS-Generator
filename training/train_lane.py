"""Train LaneAssignmentModel.

Usage:
  python -m training.train_lane \
    --dataset labeling_out_full/lane_assignment_dataset.jsonl \
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset import LaneAssignmentDataset, lane_collate, worker_init_fn
from training.models import LaneAssignmentModel
from training.utils import PoolRegistry, load_split, seed_all


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="lane_assignment_dataset.jsonl")
    ap.add_argument("--pools", required=True, help="package_pools.json")
    ap.add_argument("--run-log", required=True, help="labeling_run_log.json")
    ap.add_argument("--output", default="training/checkpoints")
    ap.add_argument("--tb", default="training/tb_logs/lane")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--class-weights",
        default=None,
        help=(
            "Class weights for CrossEntropyLoss (lane 1..7). "
            "Either 'auto' (inverse-frequency from train sample), "
            "comma-separated 7 floats (e.g. '0.9,1.1,1.0,1.0,1.0,1.1,1.1'), "
            "or omit for uniform."
        ),
    )
    ap.add_argument(
        "--class-weight-power",
        type=float,
        default=1.0,
        help="Exponent for 'auto' weights: w_i ∝ (1/freq_i)^power. Default 1.0.",
    )
    ap.add_argument(
        "--class-weight-sample",
        type=int,
        default=200_000,
        help="Records to sample for 'auto' class-weight estimation (default 200k).",
    )
    ap.add_argument(
        "--smoke-test-steps",
        type=int,
        default=0,
        help="If >0, run only N batches per epoch for quick validation. 0 = full run.",
    )
    return ap.parse_args()


def compute_auto_class_weights(dataset_jsonl: str, train_split_npy: str,
                                n_sample: int = 200_000,
                                power: float = 1.0, seed: int = 0) -> torch.Tensor:
    """Estimate per-class weights from training labels via random sampling.

    `train_split_npy` is a numpy array of byte offsets into `dataset_jsonl`,
    pre-filtered to records belonging to training packages
    (see training/dataset.py:_JsonlIndexed). NOT indices into offsets.npy.

    Returns a length-7 tensor with weights normalized to mean 1.0.
    PyTorch CE with mean reduction also normalizes by weight-sum-of-seen,
    so the mean=1 step is mostly cosmetic but keeps printed values readable.
    """
    train_offsets = np.load(train_split_npy)  # byte offsets directly
    rng = np.random.default_rng(seed)
    n = min(n_sample, len(train_offsets))
    sampled_offsets = rng.choice(train_offsets, size=n, replace=False)

    counts = np.zeros(7, dtype=np.int64)
    with open(dataset_jsonl, "rb") as f:
        for off in sampled_offsets:
            f.seek(int(off))
            line = f.readline()
            try:
                rec = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            label = rec.get("label")
            if isinstance(label, int) and 1 <= label <= 7:
                counts[label - 1] += 1

    total = counts.sum()
    if total == 0:
        raise RuntimeError("auto weights: no labels found in sample")
    freq = counts / total
    inv = 1.0 / np.clip(freq, 1e-9, None)
    weights = inv ** power
    weights = weights / weights.mean()  # normalize so mean=1 (cosmetic)
    print(f"  auto class-weights: counts={counts.tolist()}")
    print(f"  auto class-weights: freq=[" + ", ".join(f"{x:.4f}" for x in freq) + "]")
    print(f"  auto class-weights: weights=[" + ", ".join(f"{x:.4f}" for x in weights) + "]")
    return torch.tensor(weights, dtype=torch.float32)


def parse_class_weights(spec: str, dataset_jsonl: str, train_split_npy: str,
                         sample: int, power: float,
                         seed: int) -> torch.Tensor:
    """Resolve --class-weights argument to a 7-vector tensor."""
    if spec == "auto":
        return compute_auto_class_weights(
            dataset_jsonl, train_split_npy,
            n_sample=sample, power=power, seed=seed,
        )
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 7:
        raise ValueError(f"--class-weights expects 7 values, got {len(parts)}")
    vals = [float(x) for x in parts]
    if any(v <= 0 for v in vals):
        raise ValueError(f"--class-weights must all be positive, got {vals}")
    return torch.tensor(vals, dtype=torch.float32)


def run_epoch(
    model: LaneAssignmentModel,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    train: bool,
    writer: SummaryWriter,
    epoch: int,
    tag: str,
    ce_loss: torch.nn.CrossEntropyLoss,
    smoke_test_steps: int = 0,
) -> Dict[str, float]:
    model.train(train)
    ce = ce_loss
    total_loss = 0.0
    total_examples = 0
    correct = 0
    confusion = np.zeros((7, 7), dtype=np.int64)
    t0 = time.time()

    for step, batch in enumerate(loader):
        if smoke_test_steps > 0 and step >= smoke_test_steps:
            break
        event = batch["event"].to(device, non_blocking=True)
        context = batch["context"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(event, context, mask)
            loss = ce(logits, label)
        if train:
            loss.backward()
            # Smoke check: loss/grads must be finite
            if smoke_test_steps > 0:
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        raise RuntimeError(f"non-finite grad at step {step}")
            optimizer.step()

        B = int(label.shape[0])
        total_loss += float(loss.item()) * B
        total_examples += B
        pred = logits.argmax(dim=-1)
        correct += int((pred == label).sum().item())
        if not train:
            # Per-lane confusion
            p_cpu = pred.detach().cpu().numpy()
            l_cpu = label.detach().cpu().numpy()
            for p, l in zip(p_cpu, l_cpu):
                confusion[l, p] += 1

        if train and step % 500 == 0:
            gs = epoch * len(loader) + step
            writer.add_scalar(f"{tag}/step_loss", float(loss.item()), gs)
            writer.add_scalar(f"{tag}/step_acc", correct / max(1, total_examples), gs)

    elapsed = time.time() - t0
    acc = correct / max(1, total_examples)
    return {
        "loss": total_loss / max(1, total_examples),
        "accuracy": acc,
        "elapsed_sec": elapsed,
        "confusion": confusion.tolist() if not train else None,
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
    train_ds = LaneAssignmentDataset(
        args.dataset, offset_cache,
        allowed_packages=train_pkgs,
        split_cache_path=train_split_cache,
    )
    val_ds = LaneAssignmentDataset(
        args.dataset, offset_cache,
        allowed_packages=val_pkgs,
        split_cache_path=val_split_cache,
    )
    print(f"  train records={len(train_ds)} val records={len(val_ds)}")

    collate = partial(lane_collate, pools=pools)
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

    # Resolve class weights (logged into history below)
    class_weights_tensor = None
    class_weights_list = None
    if args.class_weights is not None:
        class_weights_tensor = parse_class_weights(
            args.class_weights,
            args.dataset,
            train_split_cache,
            sample=args.class_weight_sample,
            power=args.class_weight_power,
            seed=args.seed,
        )
        class_weights_list = class_weights_tensor.tolist()
        class_weights_tensor = class_weights_tensor.to(device)
    ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights_tensor, reduction="mean")
    print(f"  class weights: {class_weights_list if class_weights_list else 'uniform'}")

    model = LaneAssignmentModel(hidden=args.hidden, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Guard against overwriting production checkpoints during smoke test
    output_dir = args.output
    tb_dir = args.tb
    if args.smoke_test_steps > 0:
        output_dir = os.path.join(args.output, "_smoke_test")
        tb_dir = os.path.join(args.tb, "_smoke_test")
        print(f"  smoke-test mode: redirecting checkpoints → {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)

    best_acc = -1.0
    best_epoch = -1
    patience_left = args.patience
    history = []

    for epoch in range(args.epochs):
        print(f"epoch {epoch+1}/{args.epochs}")
        tr = run_epoch(model, train_loader, optimizer, device, True, writer, epoch, "train",
                       ce_loss=ce_loss, smoke_test_steps=args.smoke_test_steps)
        va = run_epoch(model, val_loader, optimizer, device, False, writer, epoch, "val",
                       ce_loss=ce_loss, smoke_test_steps=args.smoke_test_steps)
        print(f"  train loss={tr['loss']:.4f} acc={tr['accuracy']:.4f} ({tr['elapsed_sec']:.1f}s)")
        print(f"  val   loss={va['loss']:.4f} acc={va['accuracy']:.4f} ({va['elapsed_sec']:.1f}s)")
        writer.add_scalar("epoch/train_loss", tr["loss"], epoch)
        writer.add_scalar("epoch/train_acc", tr["accuracy"], epoch)
        writer.add_scalar("epoch/val_loss", va["loss"], epoch)
        writer.add_scalar("epoch/val_acc", va["accuracy"], epoch)
        history.append({"epoch": epoch, "train": tr, "val": va})

        ckpt_path = os.path.join(output_dir, f"lane_epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train": tr, "val": va,
            "args": vars(args),
        }, ckpt_path)

        if va["accuracy"] > best_acc:
            best_acc = va["accuracy"]
            best_epoch = epoch
            best_path = os.path.join(output_dir, "lane_best.pt")
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val": va, "args": vars(args)}, best_path)
            patience_left = args.patience
            print(f"  new best val acc={best_acc:.4f} → {best_path}")
        else:
            patience_left -= 1
            print(f"  no improvement (patience left {patience_left})")
            if patience_left <= 0:
                print("early stopping")
                break

    writer.close()

    if args.smoke_test_steps > 0:
        print("smoke test complete — skipping TorchScript export to avoid overwriting production model")
    else:
        # Export best checkpoint as TorchScript
        print("exporting best checkpoint → lane_assignment_model.pt")
        best = torch.load(os.path.join(output_dir, "lane_best.pt"), map_location="cpu")
        export_model = LaneAssignmentModel(hidden=args.hidden, dropout=args.dropout)
        export_model.load_state_dict(best["model_state"])
        export_model.eval()
        scripted = torch.jit.script(export_model)
        export_path = os.path.join(output_dir, "lane_assignment_model.pt")
        scripted.save(export_path)
        print(f"saved {export_path}")

    with open(os.path.join(output_dir, "lane_history.json"), "w", encoding="utf-8") as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_acc": best_acc,
            "class_weights": class_weights_list,
            "class_weight_power": args.class_weight_power if args.class_weights == "auto" else None,
            "history": history,
        }, f, indent=2)


if __name__ == "__main__":
    main()
