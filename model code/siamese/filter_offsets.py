#!/usr/bin/env python3
"""
Offset Similarity Filtering -- SenseGen Stage 2.5 Inference
===========================================================
Uses the trained Siamese network to filter the base offset pool by
comparing unseen-generated offsets against base offsets. Only base
offsets that are similar to unseen offsets are retained for virtual
sample generation.

Pipeline:
  1. Extract offsets from unseen data (via extract_offset.py --signal_type ...)
  2. Load trained Siamese model
  3. Compare each unseen offset to the base offset pool
  4. Select base offsets with similarity > threshold (or top-K per class)
  5. Output: filtered offset directory for use in sample generation

The unseen offsets themselves are NOT used for sample generation 
they only act as queries to select the best base offsets.

Supports WiFi, mmWave, and Ultrasound via --signal_type.

Usage:
  # mmWave
  python filter_offsets.py --signal_type mmwave \
      --siamese_checkpoint checkpoints_siamese_mmwave/siamese_best.pt

  # WiFi
  python filter_offsets.py --signal_type wifi --source volunteer_0 \
      --siamese_checkpoint checkpoints_siamese_wifi/siamese_best.pt

  # Ultrasound
  python filter_offsets.py --signal_type ultrasound --source volunteer_5 \
      --siamese_checkpoint checkpoints_siamese_ultrasound/siamese_best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from siamese_network import OffsetSiamese


def link_or_copy(src: Path, dst: Path) -> None:
    """Use hardlinks for filtered offsets when possible to avoid duplicate data."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def source_dir_name(source_prefix: str, source: str) -> str:
    return f"{source_prefix}{source}" if source_prefix else str(source)


def normalize_wifi_volunteer(value: str) -> str:
    value = str(value)
    if value.startswith("volunteer_"):
        return value
    if value.startswith("channel_"):
        value = value[len("channel_"):]
    return f"volunteer_{value}"


def source_roots(offset_dir: str, source_prefix: str, source: str) -> Tuple[Path, Path]:
    names = [source_dir_name(source_prefix, source)]
    canonical = normalize_wifi_volunteer(source)
    names.extend([canonical, f"channel_{canonical[len('volunteer_'):]}"])
    for source_dir in dict.fromkeys(names):
        static_root = Path(offset_dir) / "static" / source_dir
        dynamic_root = Path(offset_dir) / "dynamic" / source_dir
        if static_root.is_dir() and dynamic_root.is_dir():
            return static_root, dynamic_root
    source_dir = names[0]
    return Path(offset_dir) / "static" / source_dir, Path(offset_dir) / "dynamic" / source_dir


def build_base_path_index(
    base_offset_dir: str,
    source_prefix: str,
    source: str,
    base_offsets_by_class: Dict[str, List[np.ndarray]],
) -> List[Tuple[Optional[Path], Optional[Path]]]:
    """Build paths aligned with the flattened base offset array list."""
    base_static_root, base_dynamic_root = source_roots(base_offset_dir, source_prefix, source)
    base_paths: List[Tuple[Optional[Path], Optional[Path]]] = []
    for cls in sorted(base_offsets_by_class.keys()):
        pair_paths = []
        for src_static in sorted((base_static_root / cls).glob("pair_*.npy")):
            src_dynamic = base_dynamic_root / cls / src_static.name
            if src_dynamic.exists():
                pair_paths.append((src_static, src_dynamic))
        for idx, _ in enumerate(base_offsets_by_class[cls]):
            if idx < len(pair_paths):
                base_paths.append(pair_paths[idx])
            else:
                base_paths.append((None, None))
    return base_paths


def index_relative_path(path: Path, index_dir: Path) -> str:
    """Store paths relative to the index file location when possible."""
    try:
        return os.path.relpath(path, start=index_dir).replace(os.sep, "/")
    except ValueError:
        return path.as_posix()


def write_offset_index(
    output_root: Path,
    source_prefix: str,
    source: str,
    unseen_class: str,
    selected_indices: set,
    scores: np.ndarray,
    base_flat: List[Tuple[str, int, np.ndarray]],
    base_paths: List[Tuple[Optional[Path], Optional[Path]]],
    metadata: dict,
) -> Path:
    """Write a lightweight index that points back to selected base offset files."""
    index_dir = output_root / "index" / source_dir_name(source_prefix, source)
    index_dir.mkdir(parents=True, exist_ok=True)
    selected_offsets = []
    for idx in sorted(selected_indices):
        if idx >= len(base_paths):
            continue
        static_path, dynamic_path = base_paths[idx]
        if static_path is None or dynamic_path is None:
            continue
        base_class, local_idx, _ = base_flat[idx]
        selected_offsets.append({
            "base_index": int(idx),
            "base_class": base_class,
            "local_index": int(local_idx),
            "pair_file": static_path.name,
            "static_path": index_relative_path(static_path, index_dir),
            "dynamic_path": index_relative_path(dynamic_path, index_dir),
            "score": float(scores[idx]),
        })

    payload = {
        "version": 1,
        "type": "sensegen_offset_index",
        "source": source,
        "source_dir": source_dir_name(source_prefix, source),
        "unseen_class": unseen_class,
        "num_selected": len(selected_offsets),
        "selected_offsets": selected_offsets,
    }
    payload.update(metadata)

    index_path = index_dir / f"{unseen_class}.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return index_path


# ---------------------------------------------------------------------------
#  Per-modality defaults
# ---------------------------------------------------------------------------

SIGNAL_DEFAULTS = {
    "wifi": {
        "base_offset_dir": "offsets_WiFi",
        "unseen_offset_dir": "offsets_unseen_wifi",
        "output_dir": "offsets_filtered_wifi",
        "split_info_dir": "split_info_wifi",
        "use_source_prefix": False,
        "source_prefix": "",
        "source": "volunteer_0",
        "base_source": "volunteer_0",
        "class_prefix": "class_",
        "batch_size": 256,
    },
    "mmwave": {
        "base_offset_dir": "offsets_mmWave",
        "unseen_offset_dir": "offsets_unseen_mmwave",
        "output_dir": "offsets_filtered_mmwave",
        "split_info_dir": "split_info_mmwave",
        "use_source_prefix": False,
        "source_prefix": "",
        "source": "volunteer_1",
        "base_source": "volunteer_1",
        "class_prefix": "class_",
        "batch_size": 512,
    },
    "ultrasound": {
        "base_offset_dir": "offsets_ultrasound",
        "unseen_offset_dir": "offsets_unseen_ultrasound",
        "output_dir": "offsets_filtered_ultrasound",
        "split_info_dir": "split_info_ultrasound",
        "use_source_prefix": False,
        "source_prefix": "",
        "source": "volunteer_5",
        "base_source": "volunteer_5",
        "class_prefix": "",
        "batch_size": 96,
    },
}


# ---------------------------------------------------------------------------
#  Offset loading utilities
# ---------------------------------------------------------------------------

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


def load_offset_pool(
    offset_dir: str,
    source_prefix: str = "",
    source: str = "volunteer_1",
    classes: Optional[List[str]] = None,
    max_offsets_per_class: Optional[int] = None,
    class_prefix: str = "class_",
) -> Tuple[Dict[str, List[np.ndarray]], np.ndarray, np.ndarray, Tuple[int, int]]:
    """Load all offsets from a directory into memory.

    Each offset is concatenated [static, dynamic] -> (2, H, W).

    Supports two path patterns:
      - With source_prefix (WiFi): static/{source_prefix}{source}/{class}/pair_*.npy
      - Without prefix (mmWave/Ultrasound): static/{source}/{class}/pair_*.npy

    Returns:
        offsets_by_class: {class_name: [array(2, H, W), ...]}
        channel_mean: (2,) per-channel mean
        channel_std:  (2,) per-channel std
        shape: (H, W) spatial shape
    """
    static_base, dynamic_base = source_roots(offset_dir, source_prefix, source)

    if classes is None:
        classes = _discover_classes(static_base, class_prefix)

    offsets_by_class: Dict[str, List[np.ndarray]] = {}
    all_offsets: List[np.ndarray] = []

    for cls in classes:
        static_cls = static_base / cls
        dynamic_cls = dynamic_base / cls
        if not static_cls.is_dir() or not dynamic_cls.is_dir():
            continue

        pair_files = sorted(static_cls.glob("pair_*.npy"))
        if max_offsets_per_class and len(pair_files) > max_offsets_per_class:
            pair_files = pair_files[:max_offsets_per_class]

        cls_list = []
        for sf in pair_files:
            df = dynamic_cls / sf.name
            if not df.exists():
                continue
            static = np.load(sf).astype(np.float32)
            dynamic = np.load(df).astype(np.float32)
            offset = np.stack([static, dynamic], axis=0)  # (2, H, W)
            cls_list.append(offset)
            all_offsets.append(offset)

        if cls_list:
            offsets_by_class[cls] = cls_list

    if not all_offsets:
        raise RuntimeError(f"No offsets found in {offset_dir}")

    # Compute stats
    stacked = np.stack(all_offsets, axis=0)  # (N, 2, H, W)
    channel_mean = stacked.mean(axis=(0, 2, 3))  # (2,)
    channel_std = np.maximum(stacked.std(axis=(0, 2, 3)), 1e-8)  # (2,)
    shape = all_offsets[0].shape[1:]  # (H, W)

    print(f"[Load] {len(all_offsets)} offsets across {len(offsets_by_class)} classes, "
          f"shape={shape}")

    return offsets_by_class, channel_mean, channel_std, shape


def normalize_offset(offset: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score normalize per channel."""
    m = mean.reshape(2, 1, 1)
    s = std.reshape(2, 1, 1)
    return (offset - m) / s


# ---------------------------------------------------------------------------
#  Similarity computation
# ---------------------------------------------------------------------------

def compute_similarity_matrix(
    model: OffsetSiamese,
    unseen_offsets: List[np.ndarray],
    base_offsets: List[np.ndarray],
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
    verbose: bool = True,
) -> np.ndarray:
    """Compute pairwise similarity between unseen and base offsets.

    Args:
        model: Trained OffsetSiamese.
        unseen_offsets: List of (2, H, W) offset arrays.
        base_offsets: List of (2, H, W) offset arrays.
        norm_mean: (2,) per-channel mean.
        norm_std: (2,) per-channel std.
        device: torch device.
        batch_size: Max pairs per forward pass.
        verbose: Print progress.

    Returns:
        sim_matrix: (N_unseen, N_base) similarity scores in [0, 1].
    """
    model.eval()
    n_unseen = len(unseen_offsets)
    n_base = len(base_offsets)

    # Normalize all offsets
    unseen_norm = [normalize_offset(o, norm_mean, norm_std) for o in unseen_offsets]
    base_norm = [normalize_offset(o, norm_mean, norm_std) for o in base_offsets]

    # Convert to tensors
    unseen_t = torch.from_numpy(np.stack(unseen_norm, axis=0)).float()  # (Nu, 2, H, W)
    base_t = torch.from_numpy(np.stack(base_norm, axis=0)).float()      # (Nb, 2, H, W)

    sim_matrix = np.zeros((n_unseen, n_base), dtype=np.float32)

    if verbose:
        print(f"  Computing {n_unseen}x{n_base} similarity matrix...")

    total_pairs = n_unseen * n_base
    pairs_done = 0

    with torch.no_grad():
        for i in range(0, n_unseen, batch_size):
            i_end = min(i + batch_size, n_unseen)
            u_batch = unseen_t[i:i_end].to(device)  # (bu, 2, H, W)

            for j in range(0, n_base, batch_size):
                j_end = min(j + batch_size, n_base)
                b_batch = base_t[j:j_end].to(device)  # (bb, 2, H, W)

                bu = u_batch.shape[0]
                bb = b_batch.shape[0]

                # Expand for pairwise: (bu*bb, 2, H, W)
                u_exp = u_batch.unsqueeze(1).expand(-1, bb, -1, -1, -1).reshape(-1, 2, *u_batch.shape[2:])
                b_exp = b_batch.unsqueeze(0).expand(bu, -1, -1, -1, -1).reshape(-1, 2, *b_batch.shape[2:])

                scores = model(u_exp, b_exp)  # (bu*bb, 1)
                sim_matrix[i:i_end, j:j_end] = scores.reshape(bu, bb).cpu().numpy()

                pairs_done += bu * bb
                if verbose and pairs_done % (batch_size * batch_size * 4) == 0:
                    print(f"    {pairs_done}/{total_pairs} pairs done")

    if verbose:
        print(f"  Similarity matrix complete: mean={sim_matrix.mean():.3f}, "
              f"max={sim_matrix.max():.3f}, min={sim_matrix.min():.3f}")

    return sim_matrix


# ---------------------------------------------------------------------------
#  Filtering strategies
# ---------------------------------------------------------------------------

def filter_by_threshold(
    sim_matrix: np.ndarray,
    base_offsets: List[np.ndarray],
    threshold: float = 0.7,
    upper_threshold: float = 1.0,
) -> Tuple[set, np.ndarray]:
    """Select base offsets whose max similarity is in [threshold, upper_threshold]."""
    max_sim_per_base = sim_matrix.max(axis=0)  # (N_base,)
    scores = max_sim_per_base
    mask = (max_sim_per_base >= threshold) & (max_sim_per_base <= upper_threshold)
    selected = set(np.where(mask)[0])
    return selected, scores


def filter_topk(
    sim_matrix: np.ndarray,
    base_offsets: List[np.ndarray],
    top_k: int = 100,
    upper_threshold: float = 1.0,
) -> Tuple[set, np.ndarray]:
    """Select top-K base offsets by max similarity to unseen offsets.

    When upper_threshold < 1.0, first filters out offsets exceeding the upper
    bound, then selects top-K from the remainder.
    """
    max_sim_per_base = sim_matrix.max(axis=0)  # (N_base,)
    scores = max_sim_per_base
    if upper_threshold < 1.0:
        eligible = np.where(max_sim_per_base <= upper_threshold)[0]
        if len(eligible) <= top_k:
            selected = set(eligible)
        else:
            top_among = np.argsort(max_sim_per_base[eligible])[-top_k:]
            selected = set(eligible[top_among])
    else:
        top_indices = np.argsort(max_sim_per_base)[-top_k:]
        selected = set(top_indices)
    return selected, scores


def filter_by_consensus(
    sim_matrix: np.ndarray,
    base_offsets: List[np.ndarray],
    threshold: float = 0.7,
    min_votes: int = 2,
    upper_threshold: float = 1.0,
) -> Tuple[set, np.ndarray]:
    """Select base offsets similar to >= min_votes unseen offsets, in [threshold, upper_threshold]."""
    votes = (sim_matrix >= threshold).sum(axis=0)  # (N_base,)
    max_sim_per_base = sim_matrix.max(axis=0)
    scores = max_sim_per_base
    mask = (votes >= min_votes) & (max_sim_per_base <= upper_threshold)
    selected = set(np.where(mask)[0])
    return selected, scores


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Offset Similarity Filtering -- SenseGen Stage 2.5 (WiFi / mmWave / Ultrasound)",
    )

    # Modality
    parser.add_argument("--signal_type", type=str, default="mmwave",
                        choices=["wifi", "mmwave", "ultrasound"],
                        help="Signal modality (default: mmwave)")

    # Checkpoint
    parser.add_argument("--siamese_checkpoint", type=str, required=True,
                        help="Path to trained Siamese model checkpoint (.pt)")

    # Data paths
    parser.add_argument("--base_offset_dir", type=str, default=None,
                        help="Directory with base offset pool")
    parser.add_argument("--unseen_offset_dir", type=str, default=None,
                        help="Directory with unseen offset queries")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for filtered offsets")
    parser.add_argument("--source", type=str, default=None,
                        help="Target volunteer/source for unseen queries and outputs")
    parser.add_argument("--base_source", type=str, default=None,
                        help="Volunteer/source of the base offset pool "
                             "(default: modality default)")
    parser.add_argument("--unseen_classes", type=str, nargs="+", default=None,
                        help="Unseen class names (default: auto-discover)")
    parser.add_argument("--split_info_dir", type=str, default=None,
                        help="Optional few-shot split info directory for run traceability")

    # Legacy compatibility
    parser.add_argument("--volunteer", type=str, default=None,
                        help="[deprecated] Use --source instead")

    # Filtering
    parser.add_argument("--mode", type=str, default="threshold",
                        choices=["threshold", "top_k", "consensus"],
                        help="Filtering mode (default: threshold)")
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="Similarity threshold for threshold/consensus modes")
    parser.add_argument("--upper_threshold", type=float, default=1.0,
                        help="Upper similarity bound (1.0 = no upper limit)")
    parser.add_argument("--top_k", type=int, default=100,
                        help="Number of top offsets to select (top_k mode)")
    parser.add_argument("--min_votes", type=int, default=2,
                        help="Min unseen offset votes (consensus mode)")

    # Output
    parser.add_argument("--per_class", action="store_true",
                        help="Filter independently for each unseen class")
    parser.add_argument("--copy_files", action="store_true",
                        help="Copy selected offset files to output dir")
    parser.add_argument("--save_index", action="store_true",
                        help="Save lightweight JSON indexes that point to selected base offsets")
    parser.add_argument("--index_only", action="store_true",
                        help="Save indexes and skip copying selected .npy offset files")
    parser.add_argument("--save_report", action="store_true",
                        help="Save JSON filtering report")

    # System
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for similarity computation "
                             "(default: modality tuned)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve modality defaults
    signal_type = args.signal_type
    defaults = SIGNAL_DEFAULTS.get(signal_type, SIGNAL_DEFAULTS["mmwave"])
    base_offset_dir = args.base_offset_dir or defaults["base_offset_dir"]
    unseen_offset_dir = args.unseen_offset_dir or defaults["unseen_offset_dir"]
    output_dir = args.output_dir or defaults["output_dir"]
    args.split_info_dir = args.split_info_dir or defaults["split_info_dir"]
    source_prefix = defaults["source_prefix"]
    class_prefix = defaults["class_prefix"]
    if args.batch_size is None:
        args.batch_size = defaults["batch_size"]

    # Source: prefer --source, fallback to --volunteer (legacy), then defaults
    source = args.source or args.volunteer or defaults["source"]
    base_source = args.base_source or defaults.get("base_source", source)
    if args.signal_type == "wifi":
        source = normalize_wifi_volunteer(source)
        base_source = normalize_wifi_volunteer(base_source)

    print(f"[Sources] base={base_source}, target={source}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ------------------------------------------------------------------
    #  Load Siamese model
    # ------------------------------------------------------------------
    print(f"\n[Model] Loading from {args.siamese_checkpoint}")
    checkpoint = torch.load(args.siamese_checkpoint, map_location=device)
    model = OffsetSiamese().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Use norm stats from checkpoint if available
    norm_stats = checkpoint.get("norm_stats", {})
    ckpt_mean = np.array(norm_stats.get("channel_mean", [0.0, 0.0]), dtype=np.float32)
    ckpt_std = np.array(norm_stats.get("channel_std", [1.0, 1.0]), dtype=np.float32)

    print(f"  Train classes: {checkpoint.get('train_classes', 'N/A')}")
    print(f"  Best accuracy: {checkpoint.get('best_acc', 'N/A')}")

    # ------------------------------------------------------------------
    #  Load offset pools
    # ------------------------------------------------------------------
    print(f"\n[Base offsets] {base_offset_dir}")
    base_offsets_by_class, base_mean, base_std, base_shape = load_offset_pool(
        base_offset_dir,
        source_prefix=source_prefix,
        source=base_source,
        class_prefix=class_prefix,
    )

    # Use checkpoint stats if available, otherwise compute from data.
    # Do not use [1, 1] blindly when checkpoint stats are missing: that can
    # saturate the model and produce a degenerate all-1.0 similarity matrix.
    if norm_stats and np.all(np.isfinite(ckpt_mean)) and np.all(ckpt_std > 0):
        norm_mean = ckpt_mean
        norm_std = ckpt_std
    else:
        norm_mean = base_mean
        norm_std = base_std

    print(f"  [Norm] checkpoint mean={norm_mean.tolist()} std={norm_std.tolist()}")
    print(f"  [Norm] base pool  mean={base_mean.tolist()} std={base_std.tolist()}")

    # Flatten base offsets for indexing
    base_flat: List[Tuple[str, int, np.ndarray]] = []
    base_arrays: List[np.ndarray] = []
    for cls in sorted(base_offsets_by_class.keys()):
        for idx, off in enumerate(base_offsets_by_class[cls]):
            base_flat.append((cls, idx, off))
            base_arrays.append(off)
    base_paths = build_base_path_index(
        base_offset_dir, source_prefix, base_source, base_offsets_by_class
    )

    n_total = len(base_arrays)
    print(f"  Total base offsets: {n_total}")

    print(f"\n[Unseen offsets] {unseen_offset_dir}")
    if args.split_info_dir:
        print(f"  Split info: {args.split_info_dir}")
    unseen_offsets_by_class, _, _, _ = load_offset_pool(
        unseen_offset_dir,
        source_prefix=source_prefix,
        source=source,
        classes=args.unseen_classes,
        class_prefix=class_prefix,
    )

    unseen_classes = sorted(unseen_offsets_by_class.keys())
    print(f"  Unseen classes: {unseen_classes}")

    # ------------------------------------------------------------------
    #  Filtering
    # ------------------------------------------------------------------
    mode = args.mode
    output_root = Path(output_dir)
    os.makedirs(output_root, exist_ok=True)

    per_class_reports = {}
    total_selected = 0
    t_total = time.time()

    for unseen_cls in unseen_classes:
        unseen_cls_offsets = unseen_offsets_by_class[unseen_cls]
        print(f"\n{'='*55}")
        print(f"  Filtering for unseen class: {unseen_cls}")
        print(f"  {len(unseen_cls_offsets)} unseen queries x {n_total} base offsets")
        print(f"{'='*55}")

        # Compute similarity: unseen class x all base
        t0 = time.time()
        sim_matrix = compute_similarity_matrix(
            model=model,
            unseen_offsets=unseen_cls_offsets,
            base_offsets=base_arrays,
            norm_mean=norm_mean,
            norm_std=norm_std,
            device=device,
            batch_size=args.batch_size,
            verbose=False,
        )
        print(f"  Similarity computed in {time.time()-t0:.1f}s, "
              f"shape={sim_matrix.shape}")
        print(f"  Similarity stats: min={sim_matrix.min():.3f}, "
              f"max={sim_matrix.max():.3f}, mean={sim_matrix.mean():.3f}")

        # Apply filter
        if mode == "top_k":
            selected_indices, scores = filter_topk(sim_matrix, base_arrays, top_k=args.top_k, upper_threshold=args.upper_threshold)
        elif mode == "consensus":
            selected_indices, scores = filter_by_consensus(
                sim_matrix, base_arrays,
                threshold=args.threshold, min_votes=args.min_votes,
                upper_threshold=args.upper_threshold,
            )
        else:
            selected_indices, scores = filter_by_threshold(
                sim_matrix, base_arrays, threshold=args.threshold,
                upper_threshold=args.upper_threshold,
            )

        sim_mean = float(scores[list(selected_indices)].mean()) if selected_indices else 0.0
        n_sel = len(selected_indices)
        print(f"  Selected: {n_sel}/{n_total} base offsets "
              f"(mean_sim={sim_mean:.3f})")
        if n_sel == n_total and sim_mean >= 0.999:
            print(f"  [WARN] All base offsets selected with mean_sim={sim_mean:.3f}. "
                  f"This usually means train/filter offset distributions mismatch "
                  f"(for example, different offset extraction settings), or the "
                  f"checkpoint norm_stats is invalid.")

        total_selected += n_sel

        # Per-base-class stats for this unseen class
        selected_set = set(selected_indices)
        cls_stats = {}
        for cls in sorted(base_offsets_by_class.keys()):
            cls_total = len(base_offsets_by_class[cls])
            cls_sel = sum(1 for i in range(len(base_flat))
                          if base_flat[i][0] == cls and i in selected_set)
            cls_mean_sim = float(scores[
                [i for i in range(len(base_flat)) if base_flat[i][0] == cls]
            ].mean()) if cls_total > 0 else 0.0
            cls_stats[cls] = {
                "total": cls_total,
                "selected": cls_sel,
                "ratio": cls_sel / cls_total if cls_total > 0 else 0.0,
                "mean_similarity": cls_mean_sim,
            }

        per_class_reports[unseen_cls] = {
            "num_unseen_queries": len(unseen_cls_offsets),
            "num_base_selected": n_sel,
            "selection_ratio": n_sel / max(n_total, 1),
            "mean_sim_selected": sim_mean,
            "max_sim": float(scores.max()),
            "per_base_class": cls_stats,
        }

        if args.save_index or args.index_only:
            index_path = write_offset_index(
                output_root=output_root,
                source_prefix=source_prefix,
                source=source,
                unseen_class=unseen_cls,
                selected_indices=selected_indices,
                scores=scores,
                base_flat=base_flat,
                base_paths=base_paths,
                metadata={
                    "signal_type": signal_type,
                    "base_source": base_source,
                    "target_source": source,
                    "base_offset_dir": str(Path(base_offset_dir).as_posix()),
                    "unseen_offset_dir": str(Path(unseen_offset_dir).as_posix()),
                    "split_info_dir": str(Path(args.split_info_dir).as_posix()) if args.split_info_dir else None,
                    "mode": mode,
                    "threshold": args.threshold if mode in ("threshold", "consensus") else None,
                    "upper_threshold": args.upper_threshold,
                    "top_k": args.top_k if mode == "top_k" else None,
                    "min_votes": args.min_votes if mode == "consensus" else None,
                },
            )
            print(f"  Index saved: {index_path}")

        # Copy selected base offsets to per-unseen-class output dir.
        if args.copy_files and not args.index_only and selected_indices:
            static_out = output_root / "static" / source_dir_name(source_prefix, source) / unseen_cls
            dynamic_out = output_root / "dynamic" / source_dir_name(source_prefix, source) / unseen_cls
            static_out.mkdir(parents=True, exist_ok=True)
            dynamic_out.mkdir(parents=True, exist_ok=True)

            copied = 0
            for idx in selected_indices:
                cls, _, _ = base_flat[idx]
                src_static, src_dynamic = base_paths[idx]
                if src_static is not None and src_dynamic is not None:
                    new_name = f"{cls}_{src_static.name}"
                    link_or_copy(src_static, static_out / new_name)
                    link_or_copy(src_dynamic, dynamic_out / new_name)
                    copied += 1
            print(f"  Copied {copied} pairs -> .../{{static,dynamic}}/{unseen_cls}/")

        # Save per-class report
        if args.save_report:
            report = {
                "signal_type": signal_type,
                "mode": mode,
                "threshold": args.threshold if mode in ("threshold", "consensus") else None,
                "upper_threshold": args.upper_threshold,
                "top_k": args.top_k,
                "min_votes": args.min_votes if mode == "consensus" else None,
            }
            report.update(per_class_reports[unseen_cls])
            report_path = output_root / f"filter_report_{unseen_cls}.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

    # Summary
    elapsed = time.time() - t_total
    print(f"\n{'='*55}")
    print(f"  PER-CLASS FILTERING COMPLETE ({elapsed:.1f}s)")
    print(f"{'='*55}")
    for unseen_cls in unseen_classes:
        r = per_class_reports[unseen_cls]
        print(f"  {unseen_cls}: {r['num_base_selected']}/{n_total} selected "
              f"(mean_sim={r['mean_sim_selected']:.3f})")
    print(f"  Total selected across all unseen classes: {total_selected}")
    print(f"  Output: {output_dir}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
