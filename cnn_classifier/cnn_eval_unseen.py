"""
Unified CNN Evaluation on Unseen Classes (Train from Scratch)
==============================================================
Supports three modalities via --signal_type.
Two scenarios for each:
  Scenario 1 -- Pure 3-Shot Training (3 real samples per class)
  Scenario 2 -- 3-Shot + Generated Virtual Samples

Usage:
  # WiFi
  python cnn_classifier/cnn_eval_unseen.py --signal_type wifi --real_dir "WiFi data_pad/unseen"
  # mmWave
  python cnn_classifier/cnn_eval_unseen.py --signal_type mmwave --real_dir "mmWave_data_proc/unseen" --source volunteer_1
  # Ultrasound
  python cnn_classifier/cnn_eval_unseen.py --signal_type ultrasound --real_dir "ultrasound data_proc/unseen" --source volunteer_5
"""

import argparse, json, warnings, os, re
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")


def normalize_wifi_volunteer(value) -> str:
    value = str(value)
    if value.startswith("volunteer_"):
        return value
    if value.startswith("channel_"):
        value = value[len("channel_"):]
    return f"volunteer_{value}"


def resolve_wifi_source(root: Path, volunteer: str) -> Path:
    volunteer = normalize_wifi_volunteer(volunteer)
    suffix = volunteer[len("volunteer_"):]
    for candidate in (root / volunteer, root / f"channel_{suffix}"):
        if candidate.is_dir():
            return candidate
    return root / volunteer

from cnn_models import SpectrogramCNN, LabelSmoothingCrossEntropy


# ============================================================
#  Signal-type defaults
# ============================================================

SIGNAL_DEFAULTS = {
    "wifi": {
        "real_dir": "WiFi data_pad/unseen",
        "gen_dir": "generated_WiFi",
        "split_info_dir": "split_info_wifi",
        "source": "volunteer_0",
        "use_channels": True,
        "num_blocks": 4,
        "batch_size": 64,
    },
    "mmwave": {
        "real_dir": "mmWave_data_proc/unseen",
        "gen_dir": "generated_mmWave",
        "split_info_dir": "split_info_mmwave",
        "source": "volunteer_1",
        "use_source": True,
        "num_blocks": 4,
        "batch_size": 128,
    },
    "ultrasound": {
        "real_dir": "ultrasound data_proc/unseen",
        "gen_dir": "generated_ultrasound",
        "split_info_dir": "split_info_ultrasound",
        "source": "volunteer_5",
        "use_source": True,
        "num_blocks": 5,
        "batch_size": 32,
    },
}


# ============================================================
#  Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="SenseGen Unified CNN - Unseen Class Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Modality
    parser.add_argument("--signal_type", type=str, default="wifi",
                        choices=["wifi", "mmwave", "ultrasound"],
                        help="Signal modality (default: wifi)")

    # Data
    parser.add_argument("--real_dir", type=str, default=None,
                        help="Directory with real unseen data")
    parser.add_argument("--source", "--volunteer", dest="source", type=str, default=None,
                        help="Volunteer/source directory (default: modality default)")
    parser.add_argument("--channels", type=int, nargs="+", default=None,
                        help="[deprecated] Legacy WiFi channel ids")
    parser.add_argument("--gen_dir", type=str, default=None,
                        help="Directory with generated samples")
    parser.add_argument("--gen_samples", type=int, default=None,
                        help="Max generated samples per class")
    parser.add_argument("--gen_only", "--skip_pure_few_shot", action="store_true",
                        dest="gen_only",
                        help="Skip pure few-shot baseline; evaluate only shot + generated samples")
    parser.add_argument("--classes", type=str, nargs="+", default=None,
                        help="Unseen class names")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Number of classes to use (first N)")
    parser.add_argument("--num_shots", type=int, default=3,
                        help="Number of real support samples per class")
    parser.add_argument("--split_info_dir", type=str, default=None,
                        help="Directory with per-class split_info.json files from extract_offset.py")

    # Training
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="Early stopping patience (0=disabled)")
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-3,
                        help="Min improvement for early stopping")
    parser.add_argument("--batch_size", type=int, default=None)

    # Trials
    parser.add_argument("--trials", type=int, default=1,
                        help="Number of repeated trials")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-epoch output")

    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed. If omitted, multi-trial runs use "
                             "42, 43, ...; if provided, all trials reuse it.")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


# ============================================================
#  Data loading
# ============================================================

def load_sample_wifi(path: Path, num_channels: int) -> np.ndarray:
    """Load WiFi .npy, normalize, shape -> (C, H, W)."""
    arr = np.load(str(path)).astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx - mn > 1e-8:
        arr = (arr - mn) / (mx - mn)
    if arr.ndim == 2:
        arr = np.stack([arr] * num_channels, axis=0)
    return arr


def load_sample_single(path: Path) -> np.ndarray:
    """Load single-channel .npy (mmWave/Ultrasound), normalize -> (1, H, W)."""
    arr = np.load(str(path)).astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx - mn > 1e-8:
        arr = (arr - mn) / (mx - mn)
    return arr[np.newaxis, :, :]




def _select_generated_files(files, max_gen: int, rng: np.random.RandomState):
    """Select generated files evenly across generation shots.

    Generated files are named like ``shot01_s00_g0000.npy``. When
    ``max_gen`` is smaller than the total number of generated files, this
    distributes the quota across shots instead of taking the first ``max_gen``
    files in lexical order.

    Fallback: if filenames do not contain a ``shotNN_`` prefix, the previous
    behaviour (shuffle all files, then take the first ``max_gen``) is used.
    """
    files = list(files)
    if max_gen is None or max_gen >= len(files):
        return files

    shot_re = re.compile(r"^shot(\d+)_")
    groups = {}
    unparsed = []
    for f in files:
        m = shot_re.match(f.name)
        if m:
            groups.setdefault(int(m.group(1)), []).append(f)
        else:
            unparsed.append(f)

    if not groups:
        rng.shuffle(files)
        return files[:max_gen]

    shot_ids = sorted(groups)
    pools = {shot_id: list(groups[shot_id]) for shot_id in shot_ids}
    for shot_id in shot_ids:
        rng.shuffle(pools[shot_id])

    selected = []
    active = {shot_id: True for shot_id in shot_ids}
    while len(selected) < max_gen:
        progressed = False
        for shot_id in shot_ids:
            if not active.get(shot_id):
                continue
            pool = pools[shot_id]
            if pool:
                selected.append(pool.pop())
                progressed = True
                if len(selected) >= max_gen:
                    break
            else:
                active[shot_id] = False
        if not progressed:
            break

    if len(selected) < max_gen and unparsed:
        rng.shuffle(unparsed)
        selected.extend(unparsed[:max_gen - len(selected)])

    return selected[:max_gen]

def load_scenario_data(args, classes, use_gen=False, seed=42):
    """Load shot/eval split for any signal type. Returns (X, y, X_test, y_test)."""
    signal_type = args.signal_type
    defaults = SIGNAL_DEFAULTS[signal_type]
    source = args.source or defaults.get("source", "")
    channels = args.channels or [0]
    legacy_wifi = getattr(args, "legacy_wifi_channels", False)
    num_ch = len(channels) if legacy_wifi else 1

    Xt, yt = [], []
    Xe, ye = [], []
    rng = np.random.RandomState(seed)

    for ci, cn in enumerate(classes):
        # Build class dir path
        if defaults.get("use_channels"):
            if legacy_wifi:
                cls_dir = Path(args.real_dir) / f"channel_{channels[0]}" / cn
            else:
                cls_dir = resolve_wifi_source(Path(args.real_dir), source) / cn
        else:
            cls_dir = Path(args.real_dir) / source / cn

        if not cls_dir.is_dir():
            continue

        # Load split info from extract_offset.py first, then fall back to gen_dir.
        info_path = None
        if args.split_info_dir:
            canonical_info = Path(args.split_info_dir) / str(source) / cn / "split_info.json"
            legacy_info = Path(args.split_info_dir) / cn / "split_info.json"
            info_path = canonical_info if canonical_info.exists() else legacy_info
        if (info_path is None or not info_path.exists()) and args.gen_dir:
            info_path = Path(args.gen_dir) / cn / "split_info.json"
        shot_indices = None
        eval_indices = None
        if info_path and info_path.exists():
            with open(info_path, "r", encoding="utf-8") as f:
                split_info = json.load(f)
            shot_indices = split_info.get("shot_indices", [])
            eval_indices = split_info.get("eval_indices", [])

        # Map real files by sample index
        real_files = sorted(cls_dir.glob("*.npy"))
        idx_to_path = {}
        for f in real_files:
            m = re.match(r"sample_(\d+)\.npy", f.name)
            if m:
                idx_to_path[int(m.group(1))] = f
            else:
                idx_to_path[len(idx_to_path)] = f

        # Determine shot indices
        if shot_indices is not None and len(shot_indices) >= args.num_shots:
            chosen_shot = shot_indices[:args.num_shots]
        else:
            available = sorted(idx_to_path.keys())
            if len(available) >= args.num_shots:
                chosen = rng.choice(len(available), args.num_shots, replace=False)
                chosen_shot = [available[int(c)] for c in chosen]
            else:
                chosen_shot = available

        # Determine eval indices (remaining real samples)
        if eval_indices:
            chosen_eval = [i for i in eval_indices if i not in chosen_shot]
        else:
            chosen_eval = [i for i in sorted(idx_to_path.keys()) if i not in chosen_shot]

        # Load real train (shot) samples
        for si in chosen_shot:
            if si in idx_to_path:
                if defaults.get("use_channels"):
                    arr = load_sample_wifi(idx_to_path[si], num_ch)
                else:
                    arr = load_sample_single(idx_to_path[si])
                Xt.append(arr)
                yt.append(ci)

        # Load real eval samples
        for ei in chosen_eval:
            if ei in idx_to_path:
                if defaults.get("use_channels"):
                    arr = load_sample_wifi(idx_to_path[ei], num_ch)
                else:
                    arr = load_sample_single(idx_to_path[ei])
                Xe.append(arr)
                ye.append(ci)

        # Load generated samples
        if use_gen and args.gen_dir:
            gen_cls_dir = Path(args.gen_dir) / cn
            if gen_cls_dir.is_dir():
                gen_files = list(gen_cls_dir.glob("*.npy"))
                max_gen = args.gen_samples or len(gen_files)
                gen_files = _select_generated_files(gen_files, max_gen, rng)
                for gf in gen_files:
                    if defaults.get("use_channels"):
                        arr = load_sample_wifi(gf, num_ch)
                    else:
                        arr = load_sample_single(gf)
                    Xt.append(arr)
                    yt.append(ci)

    if len(Xt) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    Xt_arr = np.stack(Xt)
    yt_arr = np.array(yt, dtype=np.int64)
    Xe_arr = np.stack(Xe) if Xe else np.array([])
    ye_arr = np.array(ye, dtype=np.int64) if ye else np.array([])
    return Xt_arr, yt_arr, Xe_arr, ye_arr


# ============================================================
#  Training helpers
# ============================================================

def train_epoch_eval(model, loader, optimizer, criterion, device):
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
def evaluate_model(model, loader, criterion, device):
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


def train_from_scratch(model, train_loader, test_loader, args, device, quiet=False):
    """Train model from scratch, return best test acc, preds, labels."""
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=args.warmup_epochs, total_steps=args.epochs,
    )

    best_acc = 0.0
    best_state = None
    best_preds, best_labels = None, None
    no_improve = 0

    if not quiet:
        print(f"{'Epoch':<8} {'LR':<10} {'Train Loss':<12} {'Train Acc':<10} "
              f"{'Test Loss':<12} {'Test Acc':<10}")
        print("-" * 62)

    for epoch in range(args.epochs):
        if epoch < args.warmup_epochs:
            lr = args.lr * (epoch + 1) / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        train_loss, train_acc = train_epoch_eval(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc, preds, labels = evaluate_model(model, test_loader, criterion, device)

        if epoch >= args.warmup_epochs:
            scheduler.step()

        improved = test_acc > best_acc + args.early_stop_min_delta

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_preds = preds
            best_labels = labels

        if improved:
            no_improve = 0
        elif epoch >= args.warmup_epochs:
            no_improve += 1

        if not quiet:
            current_lr = optimizer.param_groups[0]['lr']
            marker = " *" if improved else ""
            print(f"{epoch:<8} {current_lr:<10.2e} {train_loss:<12.4e} {train_acc:<10.2f} "
                  f"{test_loss:<12.4e} {test_acc:<10.2f}{marker}")

        # Early stopping
        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            if not quiet:
                print(f"  Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)
        return best_acc, best_preds, best_labels
    return 0.0, [], []


def print_full_evaluation(preds, labels, class_names, title=""):
    """Print confusion matrix + per-class accuracy, return overall accuracy."""
    if len(preds) == 0:
        print("  No evaluation data.")
        return 0.0

    preds = np.array(preds)
    labels = np.array(labels)
    acc = 100.0 * (preds == labels).mean()

    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print(f"  Accuracy : {acc:.2f}%")

    # Confusion matrix
    present_labels = sorted(set(labels))
    if len(present_labels) > 1:
        cm = confusion_matrix(labels, preds)
        print()
        print("  Confusion Matrix:")
        header = "       " + " ".join(f"{n:>5s}" for n in class_names)
        print(header)
        for i in range(len(cm)):
            row = " ".join(f"{cm[i, j]:5d}" for j in range(len(cm)))
            print(f"  {class_names[i]:>4s} {row}")

        print()
        print("  Per-Class Accuracy:")
        for i, name in enumerate(class_names):
            if i < len(cm):
                total_cls = cm[i].sum()
                cls_acc = 100.0 * cm[i, i] / total_cls if total_cls > 0 else 0.0
                print(f"    {name}: {cls_acc:5.1f}%  ({total_cls} samples)")

    return acc


# ============================================================
#  Main
# ============================================================

def main():
    args = parse_args()
    signal_type = args.signal_type
    defaults = SIGNAL_DEFAULTS[signal_type]

    # Resolve args from defaults
    real_dir = args.real_dir or defaults["real_dir"]
    gen_dir = args.gen_dir or defaults["gen_dir"]
    legacy_wifi_channels = signal_type == "wifi" and args.channels is not None and args.source is None
    source = args.source or defaults["source"]
    if signal_type == "wifi":
        source = normalize_wifi_volunteer(source)
    args.real_dir = real_dir
    args.gen_dir = gen_dir
    args.split_info_dir = args.split_info_dir or defaults["split_info_dir"]
    args.source = source
    args.legacy_wifi_channels = legacy_wifi_channels
    if args.batch_size is None:
        args.batch_size = defaults["batch_size"]
    num_blocks = defaults["num_blocks"]
    num_ch = len(args.channels or [0]) if legacy_wifi_channels else 1

    # Resolve classes
    if args.classes is not None:
        classes = args.classes
    else:
        # Auto-discover from data directory
        if defaults.get("use_channels"):
            if legacy_wifi_channels:
                cls_dir = Path(real_dir) / f"channel_{args.channels[0]}"
            else:
                cls_dir = resolve_wifi_source(Path(real_dir), source)
        else:
            cls_dir = Path(real_dir) / source
        if cls_dir.is_dir():
            classes = sorted(
                d.name for d in cls_dir.iterdir()
                if d.is_dir() and (d.name.startswith("class_") or not d.name.startswith("."))
            )
        else:
            classes = []

    if args.num_classes is not None:
        classes = classes[:args.num_classes]

    num_unseen = len(classes)
    if num_unseen < 2:
        raise ValueError(
            "CNN unseen evaluation requires at least two classes. "
            "The packaged demo is intended for generation-path checks only."
        )

    seed_was_set = args.seed is not None
    base_seed = args.seed if seed_was_set else 42
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    n_trials = args.trials
    quiet = args.quiet or n_trials > 1
    seed_policy = (
        f"fixed ({base_seed})"
        if seed_was_set else
        (f"auto-increment ({base_seed}..{base_seed + n_trials - 1})"
         if n_trials > 1 else f"default ({base_seed})")
    )

    print("=" * 60)
    print(f"  SenseGen Unified CNN Eval -- {signal_type.upper()} Unseen")
    print("=" * 60)
    print(f"  Real data   : {real_dir}")
    if defaults.get("use_channels"):
        if legacy_wifi_channels:
            print(f"  Channels    : {args.channels} (legacy mode)")
        else:
            print(f"  Volunteer   : {source}")
    else:
        print(f"  Source      : {source}")
    print(f"  Gen data    : {gen_dir}")
    print(f"  Classes     : {classes}")
    print(f"  Shots       : {args.num_shots}")
    print(f"  Epochs      : {args.epochs}  LR: {args.lr}  WD: {args.weight_decay}")
    print(f"  Dropout     : {args.dropout}  LabelSmooth: {args.label_smoothing}")
    print(f"  Trials      : {n_trials}")
    print(f"  Seed policy : {seed_policy}")
    print(f"  Eval mode   : {'Shot + Generated only' if args.gen_only else 'Pure few-shot + Shot + Generated'}")
    print(f"  Device      : {device}")
    print()

    all_acc1, all_acc2 = [], []

    for trial in range(n_trials):
        trial_seed = base_seed if seed_was_set else base_seed + trial
        np.random.seed(trial_seed)
        torch.manual_seed(trial_seed)

        if n_trials > 1:
            print()
            print("=" * 60)
            print(f"  Trial {trial + 1}/{n_trials}  (seed={trial_seed})")
            print("=" * 60)

        # ===== Scenario 1: Pure {args.num_shots}-Shot =====
        if not args.gen_only:
            if n_trials == 1:
                print("-" * 60)
                print(f"Scenario 1: Pure {args.num_shots}-Shot Training (from scratch)")
                print("-" * 60)

            Xt1, yt1, Xe1, ye1 = load_scenario_data(
                args, classes, use_gen=False, seed=trial_seed)

            model1 = SpectrogramCNN(
                in_channels=num_ch, num_classes=num_unseen,
                dropout=args.dropout, num_blocks=num_blocks,
            ).to(device)
            n_params = sum(p.numel() for p in model1.parameters())

            if n_trials == 1:
                print(f"  Train: {len(Xt1)}  Test: {len(Xe1)}")
                print(f"  Model params: {n_params:,}")
                print()

            if len(Xt1) > 0:
                train_loader1 = DataLoader(
                    TensorDataset(torch.from_numpy(Xt1), torch.from_numpy(yt1).long()),
                    max(1, min(args.batch_size, len(Xt1))), shuffle=True)
                test_loader1 = DataLoader(
                    TensorDataset(torch.from_numpy(Xe1), torch.from_numpy(ye1).long()),
                    min(args.batch_size, len(Xe1)), shuffle=False) if len(Xe1) > 0 else None

                if test_loader1 is not None:
                    acc1, preds1, labels1 = train_from_scratch(model1, train_loader1, test_loader1, args, device, quiet)
                else:
                    acc1, preds1, labels1 = 0.0, [], []
            else:
                acc1, preds1, labels1 = 0.0, [], []

            acc1 = print_full_evaluation(preds1, labels1, classes,
                                         title=f"Pure {args.num_shots}-Shot (Trial {trial+1})" if n_trials > 1 else f"Pure {args.num_shots}-Shot Training (from scratch)")
            all_acc1.append(acc1)

        # ===== Scenario 2: {args.num_shots}-Shot + Gen =====
        if n_trials == 1:
            print()
            print("-" * 60)
            print(f"Scenario 2: {args.num_shots}-Shot + Generated Virtual Samples")
            print("-" * 60)

        Xt2, yt2, Xe2, ye2 = load_scenario_data(
            args, classes, use_gen=True, seed=trial_seed)
        gen_count = len(Xt2) - args.num_shots * num_unseen

        if gen_count <= 0:
            if n_trials == 1:
                print("  [SKIP] No generated samples found")
            all_acc2.append(0.0)
        else:
            if n_trials == 1:
                print(f"  Train: {len(Xt2)} ({args.num_shots}-shot + {gen_count} generated)  Test: {len(Xe2)}")

            model2 = SpectrogramCNN(
                in_channels=num_ch, num_classes=num_unseen,
                dropout=args.dropout, num_blocks=num_blocks,
            ).to(device)
            n_params = sum(p.numel() for p in model2.parameters())
            if n_trials == 1:
                print(f"  Model params: {n_params:,}")
                print()

            train_loader2 = DataLoader(
                TensorDataset(torch.from_numpy(Xt2), torch.from_numpy(yt2).long()),
                args.batch_size, shuffle=True)
            test_loader2 = DataLoader(
                TensorDataset(torch.from_numpy(Xe2), torch.from_numpy(ye2).long()),
                min(args.batch_size, len(Xe2)), shuffle=False) if len(Xe2) > 0 else None

            if test_loader2 is not None:
                acc2, preds2, labels2 = train_from_scratch(model2, train_loader2, test_loader2, args, device, quiet)
            else:
                acc2, preds2, labels2 = 0.0, [], []

            acc2 = print_full_evaluation(preds2, labels2, classes,
                                         title=f"{args.num_shots}-Shot+Gen (Trial {trial+1})" if n_trials > 1 else f"{args.num_shots}-Shot + Generated Virtual Samples")
            all_acc2.append(acc2)

    # ===== Final Summary =====
    print()
    print("=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)

    if n_trials == 1:
        print(f"  {'Scenario':<45s} {'Accuracy':>10s}")
        print(f"  {'-'*45} {'-'*10}")
        if not args.gen_only:
            print(f"  {'1. Pure ' + str(args.num_shots) + '-Shot (from scratch)':<45s} {all_acc1[0]:>9.2f}%")
        print(f"  {(('1. ' if args.gen_only else '2. ') + str(args.num_shots) + '-Shot + Generated'):<45s} {all_acc2[0]:>9.2f}%")
    else:
        print(f"  Per-trial breakdown:")
        if args.gen_only:
            print(f"  {'Trial':>7s} {str(args.num_shots) + '-Shot+Gen':>14s}")
            print(f"  {'-'*7} {'-'*14}")
        else:
            print(f"  {'Trial':>7s} {'Pure ' + str(args.num_shots) + '-Shot':>14s} {str(args.num_shots) + '-Shot+Gen':>14s}")
            print(f"  {'-'*7} {'-'*14} {'-'*14}")
        for t in range(n_trials):
            s2 = f"{all_acc2[t]:.2f}%" if t < len(all_acc2) else "N/A"
            if args.gen_only:
                print(f"  {t+1:>7d} {s2:>14s}")
            else:
                s1 = f"{all_acc1[t]:.2f}%" if t < len(all_acc1) else "N/A"
                print(f"  {t+1:>7d} {s1:>14s} {s2:>14s}")
        print()
        print(f"  Average over {n_trials} trials:")
        print(f"  {'Scenario':<45s} {'Mean':>8s} {'+/-Std':>8s} {'Min':>8s} {'Max':>8s}")
        print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        summary_rows = []
        if not args.gen_only:
            summary_rows.append((f"1. Pure {args.num_shots}-Shot", all_acc1))
        summary_rows.append(((f"1. " if args.gen_only else f"2. ") + f"{args.num_shots}-Shot + Generated", all_acc2))
        for name, vals in summary_rows:
            arr = np.array(vals)
            print(f"  {name:<45s} {arr.mean():>7.2f}% {arr.std(ddof=1):>7.2f}% "
                  f"{arr.min():>7.2f}% {arr.max():>7.2f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()


