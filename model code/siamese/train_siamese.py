#!/usr/bin/env python3
"""
Train Siamese Network for Offset Similarity — SenseGen Stage 2.5
=================================================================
Trains a Siamese CNN to distinguish same-pose vs different-pose offsets
using the base offset pool. The trained model is then used to filter
unseen offsets for virtual sample generation.

Training pairs:
  - Similar (label=1): two offsets from the SAME pose class
  - Dissimilar (label=0): two offsets from DIFFERENT pose classes

Split: 80% train / 20% test (within each class, avoiding instance leakage).

Supports WiFi, mmWave, and Ultrasound via --signal_type.

Usage:
  # mmWave
  python train_siamese.py --signal_type mmwave
  # WiFi
  python train_siamese.py --signal_type wifi --source volunteer_0
  # Ultrasound
  python train_siamese.py --signal_type ultrasound --source volunteer_5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from siamese_network import OffsetSiamese, OffsetEncoder, ContrastiveLoss


# ---------------------------------------------------------------------------
#  Per-modality defaults
# ---------------------------------------------------------------------------

SIGNAL_DEFAULTS = {
    "wifi": {
        "offset_dir": "offsets_WiFi",
        "output_dir": "checkpoints_siamese_wifi",
        "use_source_prefix": False,
        "source_prefix": "",
        "source": "volunteer_0",
        "class_prefix": "class_",
        "batch_size": 128,
    },
    "mmwave": {
        "offset_dir": "offsets_mmWave",
        "output_dir": "checkpoints_siamese_mmwave",
        "use_source_prefix": False,
        "source_prefix": "",
        "source": "volunteer_1",
        "class_prefix": "class_",
        "batch_size": 256,
    },
    "ultrasound": {
        "offset_dir": "offsets_ultrasound",
        "output_dir": "checkpoints_siamese_ultrasound",
        "use_source_prefix": False,
        "source_prefix": "",
        "source": "volunteer_5",
        "class_prefix": "",
        "batch_size": 96,
    },
}


def normalize_wifi_volunteer(value: str) -> str:
    value = str(value)
    if value.startswith("volunteer_"):
        return value
    if value.startswith("channel_"):
        value = value[len("channel_"):]
    return f"volunteer_{value}"


def resolve_source_roots(offset_dir: Path, source_prefix: str, source: str):
    names = [f"{source_prefix}{source}" if source_prefix else str(source)]
    canonical = normalize_wifi_volunteer(source)
    suffix = canonical[len("volunteer_"):]
    names.extend([canonical, f"channel_{suffix}"])
    for name in dict.fromkeys(names):
        static_base = offset_dir / "static" / name
        dynamic_base = offset_dir / "dynamic" / name
        if static_base.is_dir() and dynamic_base.is_dir():
            return static_base, dynamic_base
    return offset_dir / "static" / names[0], offset_dir / "dynamic" / names[0]


# ---------------------------------------------------------------------------
#  Dataset: Siamese pair generation from offset pool
# ---------------------------------------------------------------------------

class OffsetPairDataset(Dataset):
    """Generates Siamese pairs from the offset pool.

    Each epoch, generates balanced batches:
      - Half positive (same class), half negative (different class)

    Each item: (offset_a, offset_b, label) where label in {0, 1}.
    """

    def __init__(
        self,
        offset_dir: str,
        classes: List[str],
        source_prefix: str = "",
        source: str = "volunteer_1",
        pairs_per_class: int = 95,
        num_pairs_total: int = 8000,
        positive_ratio: float = 0.5,
        seed: int = 42,
        split_ratio: float = None,
        split_mode: str = "train",
    ):
        super().__init__()
        self.offset_dir = Path(offset_dir)
        self.classes = sorted(classes)
        self.source_prefix = source_prefix
        self.source = source
        self.positive_ratio = positive_ratio
        self.num_pairs_total = num_pairs_total
        self.num_positive = int(num_pairs_total * positive_ratio)
        self.num_negative = num_pairs_total - self.num_positive

        self.rng = np.random.RandomState(seed)
        self.split_ratio = split_ratio
        self.split_mode = split_mode

        # Load all offsets into memory: class_name -> list of (2, H, W) tensors
        self._offsets_by_class: Dict[str, List[np.ndarray]] = {}
        self._all_offsets: List[Tuple[str, np.ndarray]] = []
        self._load_offsets(classes, pairs_per_class, self.split_ratio, self.split_mode)

        if not self._all_offsets:
            raise RuntimeError(f"No offsets found in {offset_dir} for classes {classes}")

        # Compute per-channel mean/std for normalization
        self._compute_stats()

        # Build class index for negative sampling
        self._class_indices: Dict[str, List[int]] = {}
        offset_idx = 0
        for cls in self.classes:
            n = len(self._offsets_by_class.get(cls, []))
            if n > 0:
                self._class_indices[cls] = list(range(offset_idx, offset_idx + n))
                offset_idx += n

        self._offset_shape = self._all_offsets[0][1].shape
        print(f"[OffsetPairDataset] {len(self._all_offsets)} offsets across "
              f"{len(self._class_indices)} classes, shape={self._offset_shape}")

    def _load_offsets(self, classes: List[str], pairs_per_class: int,
                      split_ratio: float = None, split_mode: str = "train"):
        """Load offset files: static + dynamic concatenated.

        Supports two path patterns:
          - With source_prefix (WiFi): static/{source_prefix}{source}/{class}/pair_*.npy
          - Without prefix (mmWave/Ultrasound): static/{source}/{class}/pair_*.npy
        """
        static_base, dynamic_base = resolve_source_roots(
            self.offset_dir, self.source_prefix, self.source
        )

        for cls in classes:
            static_cls = static_base / cls
            dynamic_cls = dynamic_base / cls
            if not static_cls.is_dir() or not dynamic_cls.is_dir():
                print(f"  [SKIP] {cls}: missing static or dynamic dir")
                continue

            pair_files = sorted(static_cls.glob("pair_*.npy"))
            if pairs_per_class and len(pair_files) > pairs_per_class:
                pair_files = pair_files[:pairs_per_class]

            # Per-class split: train uses first split_ratio, test uses last (1-split_ratio)
            if split_ratio is not None and split_ratio < 1.0:
                n_total = len(pair_files)
                if split_mode == "train":
                    n_use = max(int(n_total * split_ratio), 2)
                    pair_files = pair_files[:n_use]
                elif split_mode == "test":
                    n_skip = int(n_total * split_ratio)
                    pair_files = pair_files[n_skip:]

            for sf in pair_files:
                df = dynamic_cls / sf.name
                if not df.exists():
                    continue
                static = np.load(sf).astype(np.float32)
                dynamic = np.load(df).astype(np.float32)
                offset = np.stack([static, dynamic], axis=0)  # (2, H, W)
                self._offsets_by_class.setdefault(cls, []).append(offset)
                self._all_offsets.append((cls, offset))

    def _compute_stats(self):
        """Compute per-channel mean/std across all loaded offsets."""
        if not self._all_offsets:
            self.channel_mean = np.zeros(2, dtype=np.float32)
            self.channel_std = np.ones(2, dtype=np.float32)
            return
        stacked = np.stack([off for _, off in self._all_offsets], axis=0)  # (N, 2, H, W)
        self.channel_mean = stacked.mean(axis=(0, 2, 3)).astype(np.float32)  # (2,)
        self.channel_std = np.maximum(stacked.std(axis=(0, 2, 3)), 1e-8).astype(np.float32)

    def _normalize(self, offset: np.ndarray) -> np.ndarray:
        """Z-score normalize per channel."""
        m = self.channel_mean.reshape(2, 1, 1)
        s = self.channel_std.reshape(2, 1, 1)
        return (offset - m) / s

    def __len__(self):
        return self.num_pairs_total

    def __getitem__(self, idx):
        # Positive or negative?
        is_positive = self.rng.rand() < self.positive_ratio

        if is_positive:
            # Same class: pick a class, then pick two different offsets
            cls = self.rng.choice(list(self._class_indices.keys()))
            indices = self._class_indices[cls]
            if len(indices) < 2:
                # Fallback: treat as negative
                is_positive = False
            else:
                i, j = self.rng.choice(indices, size=2, replace=False)
                oa = self._normalize(self._all_offsets[i][1])
                ob = self._normalize(self._all_offsets[j][1])
                label = 1.0

        if not is_positive:
            # Different class: pick two different classes
            cls_keys = list(self._class_indices.keys())
            if len(cls_keys) < 2:
                # Edge case: only one class, treat as positive
                cls = cls_keys[0]
                indices = self._class_indices[cls]
                i, j = self.rng.choice(indices, size=2, replace=False)
                oa = self._normalize(self._all_offsets[i][1])
                ob = self._normalize(self._all_offsets[j][1])
                label = 1.0
            else:
                cls_a, cls_b = self.rng.choice(cls_keys, size=2, replace=False)
                i = self.rng.choice(self._class_indices[cls_a])
                j = self.rng.choice(self._class_indices[cls_b])
                oa = self._normalize(self._all_offsets[i][1])
                ob = self._normalize(self._all_offsets[j][1])
                label = 0.0

        return (
            torch.from_numpy(oa).float(),
            torch.from_numpy(ob).float(),
            torch.tensor([label], dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Siamese Network for Offset Similarity (WiFi / mmWave / Ultrasound)",
    )

    # Modality
    parser.add_argument("--signal_type", type=str, default="mmwave",
                        choices=["wifi", "mmwave", "ultrasound"],
                        help="Signal modality (default: mmwave)")

    # Data paths
    parser.add_argument("--offset_dir", type=str, default=None,
                        help="Directory with base offset pool (static/ and dynamic/)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save model checkpoints")
    parser.add_argument("--source", type=str, default=None,
                        help="Volunteer/source directory (default: modality default)")
    parser.add_argument("--classes", type=str, nargs="+", default=None,
                        help="Classes to use (default: auto-discover from offset_dir)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Fraction of classes used for training (default: 0.8)")
    parser.add_argument("--pairs_per_class", type=int, default=None,
                        help="Max offset pairs per class (default: auto)")
    parser.add_argument("--train_pairs", type=int, default=2000,
                        help="Total pairs per training epoch (default: 2000)")
    parser.add_argument("--seed", type=int, default=42)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.5,
                        help="Contrastive loss margin")

    # System
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=25)

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_accuracy(model, loader, device):
    """Compute accuracy and average similarity for pos/neg pairs."""
    model.eval()
    correct = 0
    total = 0
    pos_sims = []
    neg_sims = []

    with torch.no_grad():
        for oa, ob, label in loader:
            oa = oa.to(device)
            ob = ob.to(device)
            label = label.to(device)

            similarity = model(oa, ob)
            pred = (similarity >= 0.5).float()

            correct += (pred == label).sum().item()
            total += label.size(0)

            pos_mask = label == 1.0
            neg_mask = label == 0.0
            if pos_mask.any():
                pos_sims.append(similarity[pos_mask].mean().item())
            if neg_mask.any():
                neg_sims.append(similarity[neg_mask].mean().item())

    acc = correct / max(total, 1)
    avg_pos = float(np.mean(pos_sims)) if pos_sims else 0.0
    avg_neg = float(np.mean(neg_sims)) if neg_sims else 0.0
    model.train()
    return acc, avg_pos, avg_neg


def _discover_classes(static_base: Path, class_prefix: str) -> List[str]:
    """Auto-discover class directories under static_base.

    If class_prefix is non-empty, filters by prefix (e.g. 'class_' for WiFi/mmWave).
    If class_prefix is empty (Ultrasound), takes all subdirs.
    """
    if not static_base.is_dir():
        return []
    if class_prefix:
        return sorted([
            d.name for d in static_base.iterdir()
            if d.is_dir() and d.name.startswith(class_prefix)
        ])
    else:
        return sorted([
            d.name for d in static_base.iterdir()
            if d.is_dir()
        ])


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    # Resolve modality defaults
    signal_type = args.signal_type
    defaults = SIGNAL_DEFAULTS.get(signal_type, SIGNAL_DEFAULTS["mmwave"])
    offset_dir = args.offset_dir or defaults["offset_dir"]
    output_dir = args.output_dir or defaults["output_dir"]
    source_prefix = defaults["source_prefix"]
    source = args.source or defaults["source"]
    if args.signal_type == "wifi":
        source = normalize_wifi_volunteer(source)
    class_prefix = defaults["class_prefix"]
    if args.batch_size is None:
        args.batch_size = defaults["batch_size"]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ------------------------------------------------------------------
    #  Discover classes
    # ------------------------------------------------------------------
    offset_path = Path(offset_dir)
    static_base, _ = resolve_source_roots(offset_path, source_prefix, source)

    if args.classes:
        all_classes = sorted(args.classes)
    else:
        all_classes = _discover_classes(static_base, class_prefix)

    if len(all_classes) < 2:
        raise RuntimeError(f"Need at least 2 classes, found {len(all_classes)} in {static_base}")

    # Train/test class split
    rng_cls = np.random.RandomState(args.seed)
    indices = rng_cls.permutation(len(all_classes))
    n_train = max(int(len(all_classes) * args.train_ratio), 1)
    n_train = min(n_train, len(all_classes) - 1)
    train_classes = [all_classes[i] for i in sorted(indices[:n_train])]
    test_classes = [all_classes[i] for i in sorted(indices[n_train:])]

    print(f"\n[Classes] {len(all_classes)} total -> "
          f"{len(train_classes)} train, {len(test_classes)} test")
    print(f"  Train: {train_classes}")
    print(f"  Test : {test_classes}")

    # ------------------------------------------------------------------
    #  Datasets & Loaders
    # ------------------------------------------------------------------
    pairs_per_class = args.pairs_per_class

    train_dataset = OffsetPairDataset(
        offset_dir=offset_dir,
        classes=train_classes,
        source_prefix=source_prefix,
        source=source,
        pairs_per_class=pairs_per_class,
        num_pairs_total=args.train_pairs,
        positive_ratio=0.5,
        seed=args.seed,
    )

    test_dataset = OffsetPairDataset(
        offset_dir=offset_dir,
        classes=test_classes,
        source_prefix=source_prefix,
        source=source,
        pairs_per_class=pairs_per_class,
        num_pairs_total=max(512, args.train_pairs // 4),
        positive_ratio=0.5,
        seed=args.seed + 1,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------
    #  Model, loss, optimizer
    # ------------------------------------------------------------------
    model = OffsetSiamese().to(device)
    criterion = ContrastiveLoss(margin=args.margin).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Total parameters: {total_params:,}")

    # Save normalization stats for inference
    os.makedirs(output_dir, exist_ok=True)
    norm_stats = {
        "signal_type": signal_type,
        "source_prefix": source_prefix,
        "source": source,
        "channel_mean": train_dataset.channel_mean.tolist(),
        "channel_std": train_dataset.channel_std.tolist(),
        "train_classes": train_classes,
        "test_classes": test_classes,
    }
    with open(os.path.join(output_dir, "norm_stats.json"), "w") as f:
        json.dump(norm_stats, f, indent=2)

    # ------------------------------------------------------------------
    #  Training loop
    # ------------------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  Training Siamese Network ({signal_type.upper()})")
    print(f"{'='*55}")
    print(f"  Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"  Margin: {args.margin}  |  Pairs/epoch: {args.train_pairs}")
    print(f"  Offset dir: {offset_dir}")
    print(f"  Output: {output_dir}")
    print(f"{'='*55}\n")

    best_acc = 0.0
    t_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_steps = 0

        for step, (oa, ob, label) in enumerate(train_loader):
            oa = oa.to(device)
            ob = ob.to(device)
            label = label.to(device)

            similarity = model(oa, ob)
            loss = criterion(similarity, label)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                pred = (similarity >= 0.5).float()
                acc = (pred == label).float().mean().item()

            epoch_loss += loss.item()
            epoch_acc += acc
            epoch_steps += 1

            if step % args.log_interval == 0:
                print(f"  [Epoch {epoch+1:3d} | Step {step:4d}] "
                      f"loss={loss.item():.4f}  acc={acc:.3f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()
        avg_loss = epoch_loss / epoch_steps
        avg_acc = epoch_acc / epoch_steps

        # Evaluate on test set
        test_acc, test_pos_sim, test_neg_sim = compute_accuracy(model, test_loader, device)

        elapsed = time.time() - t_start
        print(f"  [Epoch {epoch+1:3d}] train_loss={avg_loss:.4f}  train_acc={avg_acc:.3f}  "
              f"test_acc={test_acc:.3f}  pos_sim={test_pos_sim:.3f}  "
              f"neg_sim={test_neg_sim:.3f}  time={elapsed:.1f}s")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
                "norm_stats": norm_stats,
                "train_classes": train_classes,
                "test_classes": test_classes,
            }, os.path.join(output_dir, "siamese_best.pt"))
            print(f"    [Best] acc={best_acc:.3f} saved.")

        # Periodic checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
                "norm_stats": norm_stats,
            }, os.path.join(output_dir, f"siamese_epoch_{epoch+1:03d}.pt"))

    # Final save
    torch.save({
        "epoch": args.epochs - 1,
        "model_state_dict": model.state_dict(),
        "best_acc": best_acc,
        "norm_stats": norm_stats,
        "train_classes": train_classes,
        "test_classes": test_classes,
    }, os.path.join(output_dir, "siamese_final.pt"))

    print(f"\n{'='*55}")
    print(f"  Training complete. Best test accuracy: {best_acc:.3f}")
    print(f"  Model saved to: {output_dir}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
