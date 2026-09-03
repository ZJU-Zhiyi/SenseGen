"""
Unified CNN Classifier for SenseGen Base Dataset
=================================================
Supports three modalities via --signal_type:
  wifi       — CSI spectrograms grouped by volunteer (volunteer_X/class_XX/)
  mmwave     — Range-Doppler-Angle heatmaps (volunteer_X/class_XX/)
  ultrasound — Ultrasound spectrograms (volunteer_X/{class}/), 5-block CNN

Usage:
  python cnn_classifier/cnn_classify.py --signal_type wifi
  python cnn_classifier/cnn_classify.py --signal_type mmwave --data_dir "mmWave_data_proc/base"
  python cnn_classifier/cnn_classify.py --signal_type ultrasound --data_dir "ultrasound data_proc/base" --source volunteer_5
"""

import argparse, random, json
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from cnn_models import SpectrogramCNN, LabelSmoothingCrossEntropy


def normalize_wifi_volunteer(value) -> str:
    value = str(value)
    if value.startswith("volunteer_"):
        return value
    if value.startswith("channel_"):
        value = value[len("channel_"):]
    return f"volunteer_{value}"


def resolve_wifi_source(data_dir: str, volunteer: str) -> Path:
    volunteer = normalize_wifi_volunteer(volunteer)
    suffix = volunteer[len("volunteer_"):]
    root = Path(data_dir)
    for candidate in (root / volunteer, root / f"channel_{suffix}"):
        if candidate.is_dir():
            return candidate
    return root / volunteer


# ============================================================
#  Unified Dataset
# ============================================================

class UnifiedDataset(Dataset):
    """Load .npy files for any signal type.

    signal_type='wifi':        data_dir/volunteer_X/class_XX/*.npy
    signal_type='mmwave':      data_dir/source/class_XX/*.npy (single channel)
    signal_type='ultrasound':  data_dir/source/{class}/*.npy (single channel)
    """

    def __init__(self, data_dir: str, signal_type: str,
                 volunteers: List[str] = None, channels: List[int] = None,
                 source: str = "volunteer_1",
                 classes: List[str] = None):
        self.signal_type = signal_type
        self.channels = channels or [0]
        self.volunteers = [normalize_wifi_volunteer(v) for v in (volunteers or [])]
        self._legacy_channel_mode = signal_type == "wifi" and not self.volunteers
        self.source = source
        self.samples: List[Tuple[str, int]] = []
        self._multi_channel = False

        # Build class list
        if classes is None:
            if signal_type == "wifi":
                source_dirs = (
                    [resolve_wifi_source(data_dir, v) for v in self.volunteers]
                    if self.volunteers else
                    [Path(data_dir) / f"channel_{self.channels[0]}"]
                )
                classes = sorted({
                    d.name for source_dir in source_dirs if source_dir.is_dir()
                    for d in source_dir.iterdir()
                    if d.is_dir() and d.name.startswith("class_")
                })
            elif signal_type == "mmwave":
                src_dir = Path(data_dir) / source
                classes = sorted(
                    d.name for d in src_dir.iterdir()
                    if d.is_dir() and d.name.startswith("class_")
                )
            elif signal_type == "ultrasound":
                src_dir = Path(data_dir) / source
                classes = sorted(
                    d.name for d in src_dir.iterdir()
                    if d.is_dir()
                )
        self.classes = classes
        self.num_classes = len(classes)

        # Collect sample paths
        if signal_type == "wifi":
            if self.volunteers:
                for volunteer in self.volunteers:
                    source_dir = resolve_wifi_source(data_dir, volunteer)
                    for label_idx, cls_name in enumerate(self.classes):
                        cls_dir = source_dir / cls_name
                        if cls_dir.is_dir():
                            for npy_file in sorted(cls_dir.glob("*.npy")):
                                self.samples.append((str(npy_file), label_idx))
            elif len(self.channels) == 1:
                ch = self.channels[0]
                ch_dir = Path(data_dir) / f"channel_{ch}"
                for label_idx, cls_name in enumerate(self.classes):
                    cls_dir = ch_dir / cls_name
                    if cls_dir.is_dir():
                        for npy_file in sorted(cls_dir.glob("*.npy")):
                            self.samples.append((str(npy_file), label_idx))
            else:
                self._multi_channel = True
                from collections import defaultdict
                sample_map = defaultdict(dict)
                for ch in self.channels:
                    ch_dir = Path(data_dir) / f"channel_{ch}"
                    for label_idx, cls_name in enumerate(self.classes):
                        cls_dir = ch_dir / cls_name
                        if cls_dir.is_dir():
                            for npy_file in sorted(cls_dir.glob("*.npy")):
                                key = (cls_name, npy_file.name)
                                sample_map[key][ch] = str(npy_file)
                for (cls_name, sname), ch_paths in sorted(sample_map.items()):
                    if len(ch_paths) == len(self.channels):
                        paths = [ch_paths[ch] for ch in self.channels]
                        label_idx = self.classes.index(cls_name)
                        self.samples.append((tuple(paths), label_idx))
        else:
            # mmwave / ultrasound: flat directory structure
            src_dir = Path(data_dir) / source
            for label_idx, cls_name in enumerate(self.classes):
                cls_dir = src_dir / cls_name
                if cls_dir.is_dir():
                    for npy_file in sorted(cls_dir.glob("*.npy")):
                        self.samples.append((str(npy_file), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        if self._multi_channel:
            paths, label = item
            arrays = []
            for p in paths:
                arr = np.load(p).astype(np.float32)
                mn, mx = arr.min(), arr.max()
                if mx - mn > 1e-8:
                    arr = (arr - mn) / (mx - mn)
                arrays.append(arr)
            tensor = torch.from_numpy(np.stack(arrays, axis=0))  # (C, H, W)
        else:
            filepath, label = item
            arr = np.load(filepath).astype(np.float32)
            mn, mx = arr.min(), arr.max()
            if mx - mn > 1e-8:
                arr = (arr - mn) / (mx - mn)
            tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
        return tensor, label


# ============================================================
#  Training utilities
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (model(x).argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    return total_loss / total, 100.0 * correct / total, all_preds, all_labels


def eval_random_subsample(all_preds, all_labels, num_classes, k=5,
                           rounds=100, rng=None):
    """Random k-class subsampling accuracy averaged over rounds."""
    if rng is None:
        rng = np.random.RandomState(42)
    # Clamp k to num_classes
    k = min(k, num_classes)
    if k < 2:
        return float(100.0 * (all_preds == all_labels).mean()), 0.0, []
    acc_list = []
    for _ in range(rounds):
        chosen = sorted(rng.choice(num_classes, k, replace=False))
        mask = np.isin(all_labels, chosen)
        if mask.sum() == 0:
            continue
        sub_labels = all_labels[mask]
        sub_preds = all_preds[mask]
        label_map = {orig: new for new, orig in enumerate(chosen)}
        sub_labels_mapped = np.array([label_map[l] for l in sub_labels])
        sub_preds_mapped = np.array([label_map.get(p, -1) for p in sub_preds])
        valid = sub_preds_mapped >= 0
        acc = 100.0 * (sub_labels_mapped[valid] == sub_preds_mapped[valid]).mean()
        acc_list.append(acc)
    acc_arr = np.array(acc_list)
    return float(acc_arr.mean()), float(acc_arr.std()), acc_list


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self._step_count
        if step < self.warmup_steps:
            return [base_lr * step / self.warmup_steps
                    for base_lr in self.base_lrs]
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return [base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
                for base_lr in self.base_lrs]


# ============================================================
#  Default configs per modality
# ============================================================

SIGNAL_DEFAULTS = {
    "wifi": {
        "data_dir": "WiFi data_pad/base",
        "num_blocks": 4,
        "use_volunteers": True,
        "source": "volunteer_0",
        "batch_size": 64,
    },
    "mmwave": {
        "data_dir": "mmWave_data_proc/base",
        "num_blocks": 4,
        "use_source": True,
        "source": "volunteer_1",
        "batch_size": 128,
    },
    "ultrasound": {
        "data_dir": "ultrasound data_proc/base",
        "num_blocks": 5,
        "use_source": True,
        "source": "volunteer_5",
        "batch_size": 32,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="SenseGen Unified CNN Classifier (Base Dataset)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Modality selection
    parser.add_argument("--signal_type", type=str, default="wifi",
                        choices=["wifi", "mmwave", "ultrasound"],
                        help="Signal modality (default: wifi)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory (overrides signal_type default)")
    parser.add_argument("--volunteers", type=str, nargs="+", default=None,
                        help="WiFi volunteer directories (default: volunteer_0)")
    parser.add_argument("--channels", type=int, nargs="+", default=None,
                        help="[deprecated] Legacy multi-channel WiFi input")
    # mmWave / Ultrasound
    parser.add_argument("--source", type=str, default=None,
                        help="Volunteer/source directory (default: modality default)")
    # Training
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--test_size", type=float, default=0.2)
    # Evaluation
    parser.add_argument("--eval_rounds", type=int, default=500)
    parser.add_argument("--eval_classes_per_round", type=int, default=5)
    # Output
    parser.add_argument("--save_model", type=str, default=None,
                        help="Path for the saved model (default: cnn_classifier/cnn_best_<signal_type>.pt)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


# ============================================================
#  Main
# ============================================================

def main():
    args = parse_args()
    signal_type = args.signal_type
    defaults = SIGNAL_DEFAULTS[signal_type]

    # Resolve data_dir
    data_dir = args.data_dir or defaults["data_dir"]
    num_blocks = defaults["num_blocks"]
    if args.batch_size is None:
        args.batch_size = defaults["batch_size"]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build dataset
    volunteers = args.volunteers
    if signal_type == "wifi" and volunteers is None and args.channels is None:
        volunteers = ["volunteer_0"]
    channels = args.channels or [0]
    source_arg = args.source or defaults.get("source", "volunteer_0")
    if args.save_model is None:
        args.save_model = f"cnn_classifier/cnn_best_{signal_type}.pt"

    dataset = UnifiedDataset(
        data_dir=data_dir,
        signal_type=signal_type,
        volunteers=volunteers,
        channels=channels,
        source=source_arg,
    )
    num_classes = dataset.num_classes
    if num_classes < 2:
        raise ValueError(
            "CNN classification requires at least two classes. "
            "The packaged demo is intended for generation-path checks only."
        )
    in_channels = len(channels) if signal_type == "wifi" and not volunteers else 1

    print("=" * 60)
    print(f"  SenseGen Unified CNN Classifier -- {signal_type.upper()}")
    print("=" * 60)
    print(f"  Data dir    : {data_dir}")
    print(f"  Signal type : {signal_type}")
    print(f"  Classes     : {num_classes} ({', '.join(dataset.classes[:5])}...)")
    print(f"  Samples     : {len(dataset)}")
    if signal_type == "wifi":
        print(f"  Volunteers  : {volunteers or ['legacy channel mode']}")
    else:
        print(f"  Source      : {source_arg}")
    print(f"  In channels : {in_channels}")
    print(f"  CNN blocks  : {num_blocks}")
    print(f"  Epochs      : {args.epochs}  Batch: {args.batch_size}")
    print(f"  LR: {args.lr}  WD: {args.weight_decay}  Dropout: {args.dropout}")
    print(f"  Label Smooth: {args.label_smoothing}  Warmup: {args.warmup_epochs}")
    print(f"  Device      : {device}")

    # Train/Test split
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, random_state=args.seed,
        stratify=[s[1] for s in dataset.samples]
    )
    train_loader = DataLoader(Subset(dataset, train_idx),
                               args.batch_size, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_idx),
                              args.batch_size, shuffle=False)

    # Model
    model = SpectrogramCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=args.dropout,
        num_blocks=num_blocks,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params      : {n_params:,}")

    # Optimizer & Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=args.warmup_epochs,
        total_steps=args.epochs,
    )

    # Training loop
    print()
    print(f"{'Epoch':<8} {'LR':<10} {'Train Loss':<12} {'Train Acc':<10} "
          f"{'Test Loss':<12} {'Test Acc':<10}")
    print("-" * 62)

    best_acc = 0.0
    best_state = None

    for epoch in range(args.epochs):
        if epoch < args.warmup_epochs:
            lr = args.lr * (epoch + 1) / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc, _, _ = evaluate(
            model, test_loader, criterion, device
        )

        if epoch >= args.warmup_epochs and scheduler is not None:
            scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

        current_lr = optimizer.param_groups[0]['lr']
        print(f"{epoch:<8} {current_lr:<10.2e} {train_loss:<12.4f} "
              f"{train_acc:<10.2f} {test_loss:<12.4f} {test_acc:<10.2f}")

    # ---------- Final Evaluation ----------
    if best_state is not None:
        model.load_state_dict(best_state)

    _, test_acc, all_preds, all_labels = evaluate(
        model, test_loader, criterion, device
    )
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print()
    print("=" * 60)
    print(f"  Full {num_classes}-Class Test Accuracy : {test_acc:.2f}%")
    print("=" * 60)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    class_names = dataset.classes

    print()
    print(f"  Confusion Matrix ({num_classes} classes):")
    short_names = [n.replace("class_", "c") if n.startswith("class_") else n
                   for n in class_names]
    header = "        " + " ".join(f"{n:>5s}" for n in short_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = " ".join(f"{v:5d}" for v in row)
        print(f"  {class_names[i]:7s} {row_str}")

    print()
    print("  Per-Class Accuracy:")
    for i, name in enumerate(class_names):
        total_cls = cm[i].sum()
        cls_acc = 100.0 * cm[i, i] / total_cls if total_cls > 0 else 0.0
        print(f"    {name}: {cls_acc:5.1f}%  ({total_cls} samples)")

    # Random K-Class Subsampling
    k = args.eval_classes_per_round
    rounds = args.eval_rounds
    rng = np.random.RandomState(args.seed)
    mean_acc, std_acc, acc_list = eval_random_subsample(
        all_preds, all_labels, num_classes, k=k, rounds=rounds, rng=rng
    )

    print()
    print("=" * 60)
    print(f"  Random {k}-Class Subsampling Evaluation ({rounds} rounds)")
    print("=" * 60)
    print(f"  Mean  : {mean_acc:.2f}%")
    print(f"  Std   : {std_acc:.2f}%")
    print(f"  Median: {np.median(acc_list):.2f}%")
    print(f"  Min / Max : {min(acc_list):.2f}% / {max(acc_list):.2f}%")

    # ---------- Save ----------
    if args.save_model and best_state is not None:
        save_path = Path(args.save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model_state": best_state,
            "model_class": "SpectrogramCNN",
            "signal_type": signal_type,
            "num_blocks": num_blocks,
            "in_channels": in_channels,
            "num_classes": num_classes,
            "classes": dataset.classes,
            "volunteers": volunteers,
            "channels": args.channels,
            "source": source_arg,
            "dropout": args.dropout,
            "full_test_acc": test_acc,
            f"{k}class_avg_acc": mean_acc,
            f"{k}class_std_acc": std_acc,
        }, save_path)

        per_class_acc = {}
        for i, name in enumerate(class_names):
            total_cls = int(cm[i].sum())
            cls_acc = round(100.0 * cm[i, i] / total_cls, 2) if total_cls > 0 else 0.0
            per_class_acc[name] = {"accuracy": cls_acc, "samples": total_cls}

        config = {
            "model": f"SpectrogramCNN ({num_blocks}-block ResNet)",
            "signal_type": signal_type,
            "data_dir": data_dir,
            "volunteers": volunteers,
            "channels": args.channels,
            "source": source_arg,
            "in_channels": in_channels,
            "num_classes": num_classes,
            "classes": dataset.classes,
            "test_size": args.test_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "label_smoothing": args.label_smoothing,
            "warmup_epochs": args.warmup_epochs,
            "seed": args.seed,
            "results": {
                "full_test_acc": round(test_acc, 2),
                "per_class_acc": per_class_acc,
                f"{k}class_avg_acc": round(mean_acc, 2),
                f"{k}class_std_acc": round(std_acc, 2),
                f"{k}class_median_acc": round(float(np.median(acc_list)), 2),
                f"{k}class_min_acc": round(float(min(acc_list)), 2),
                f"{k}class_max_acc": round(float(max(acc_list)), 2),
                "eval_rounds": rounds,
                "eval_classes_per_round": k,
            }
        }
        config_path = save_path.with_suffix('.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print()
        print(f"  Model saved to : {save_path}")
        print(f"  Config saved to: {config_path}")

    print()
    print("=" * 60)
    print("  Training Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()

