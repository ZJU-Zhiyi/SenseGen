#!/usr/bin/env python3
"""
Behavioral Diversity Characterization - SenseGen Stage 2 (Unified)
====================================================================
Extracts static & dynamic offsets from paired samples within each class.
Supports three modalities via --signal_type.

Usage:
  python "model code/extract_offset.py" --signal_type wifi
  python "model code/extract_offset.py" --signal_type mmwave
  python "model code/extract_offset.py" --signal_type ultrasound

Output structure:
  WiFi:        {output_dir}/static/volunteer_X/class_XX/pair_00_01.npy ...
  mmWave:     {output_dir}/static/volunteer_X/class_XX/pair_00_01.npy ...
  Ultrasound:  {output_dir}/static/volunteer_X/{class}/pair_00_01.npy ...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter


# ============================================================
#  Per-modality defaults
# ============================================================

SIGNAL_DEFAULTS = {
    "wifi": {
        "data_dir": "WiFi data_pad/base",
        "unseen_data_dir": "WiFi data_pad/unseen",
        "output_dir": "offsets_WiFi",
        "unseen_output_dir": "offsets_unseen_wifi",
        "split_info_dir": "split_info_wifi",
        "use_channels": False,
        "sg_window": 51,
        "sg_order": 3,
        "sg_axis": 1,
        "max_pairs": None,
        "volunteers": ["volunteer_0"],
    },
    "mmwave": {
        "data_dir": "mmWave_data_proc/base",
        "unseen_data_dir": "mmWave_data_proc/unseen",
        "output_dir": "offsets_mmWave",
        "unseen_output_dir": "offsets_unseen_mmwave",
        "split_info_dir": "split_info_mmwave",
        "use_channels": False,
        "sg_window": 5,
        "sg_order": 3,
        "sg_axis": 0,
        "max_pairs": None,
        "volunteers": ["volunteer_1"],
    },

    "ultrasound": {
        "data_dir": "ultrasound data_proc/base",
        "unseen_data_dir": "ultrasound data_proc/unseen",
        "output_dir": "offsets_ultrasound",
        "unseen_output_dir": "offsets_unseen_ultrasound",
        "split_info_dir": "split_info_ultrasound",
        "use_channels": False,
        "sg_window": 51,
        "sg_order": 3,
        "sg_axis": 1,
        "max_pairs": 780,
        "volunteers": ["volunteer_5"],
    },
}


# ============================================================
#  Core offset computation
# ============================================================

def compute_offset(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Element-wise difference between two samples."""
    return (s1 - s2).astype(np.float32)


def normalize_wifi_volunteer(value: str) -> str:
    value = str(value)
    if value.startswith("volunteer_"):
        return value
    if value.startswith("channel_"):
        value = value[len("channel_"):]
    return f"volunteer_{value}"


def resolve_wifi_source(root: Path, volunteer: str) -> Optional[Path]:
    volunteer = normalize_wifi_volunteer(volunteer)
    suffix = volunteer[len("volunteer_"):]
    for candidate in (root / volunteer, root / f"channel_{suffix}"):
        if candidate.is_dir():
            return candidate
    return None


def decompose_offset(
    offset: np.ndarray,
    window_length: int = 51,
    polyorder: int = 3,
    axis: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose offset into static (smoothed trend) and dynamic (residual).

    Args:
        offset: (H, W) offset array.
        window_length: Savitzky-Golay window (must be odd, <= W).
        polyorder: Polynomial order for S-G filter.
        axis: Axis along which to apply the filter.

    Returns:
        static: Smoothed trend, same shape as offset.
        dynamic: High-frequency residual, same shape as offset.
    """
    static = savgol_filter(offset, window_length, polyorder, axis=axis, mode="nearest")
    return static.astype(np.float32), (offset - static).astype(np.float32)


# ============================================================
#  Class-level processing
# ============================================================

def sample_id_from_name(path: Path, fallback: int) -> int:
    """Return the numeric id from sample_XX.npy, or fallback ordinal."""
    match = re.match(r"sample_(\d+)\.npy$", path.name)
    if match:
        return int(match.group(1))
    return fallback


def process_class(
    class_dir: Path,
    static_dir: Path,
    dynamic_dir: Path,
    max_pairs: Optional[int],
    sg_window: int,
    sg_order: int,
    sg_axis: int,
    rng: np.random.RandomState,
    sample_files: Optional[List[Path]] = None,
) -> int:
    """Process one class: pair all samples, decompose offsets, save to disk.

    Returns:
        Number of pairs processed.
    """
    npy_files = sorted(sample_files) if sample_files is not None else sorted(class_dir.glob("*.npy"))
    if len(npy_files) < 2:
        return 0

    # Load all samples into memory and preserve their sample_XX ids so
    # downstream diffusion/sample offset lookup can match pair filenames.
    samples_list = []
    sample_ids = []
    for ordinal, f in enumerate(npy_files):
        arr = np.load(f).astype(np.float32)
        if arr.ndim != 2:
            print(f"  [WARN] Unexpected shape {arr.shape} in {f}, skipping")
            continue
        samples_list.append(arr)
        sample_ids.append(sample_id_from_name(f, ordinal))

    if len(samples_list) < 2:
        return 0

    samples = np.stack(samples_list, axis=0)  # (N, H, W)
    n = samples.shape[0]

    # Generate all unique pairs
    indices = list(range(n))
    pairs = list(combinations(indices, 2))
    if max_pairs and len(pairs) > max_pairs:
        chosen = rng.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[i] for i in sorted(chosen)]

    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(dynamic_dir, exist_ok=True)

    for i, j in pairs:
        raw = compute_offset(samples[i], samples[j])
        static, dynamic = decompose_offset(raw, sg_window, sg_order, sg_axis)

        pair_name = f"pair_{sample_ids[i]:02d}_{sample_ids[j]:02d}.npy"
        np.save(static_dir / pair_name, static)
        np.save(dynamic_dir / pair_name, dynamic)

    return len(pairs)


def load_or_create_fewshot_split(
    class_dir: Path,
    class_name: str,
    source_name: str,
    signal_type: str,
    num_shots: int,
    split_info_dir: Path,
    force_split: bool,
    seed: int,
    rng: np.random.RandomState,
    split_cache: Dict[str, dict],
) -> Tuple[List[Path], dict]:
    """Select few-shot samples for one class and persist reusable split info."""
    if num_shots < 1:
        raise ValueError("--num_shots must be >= 1")

    cache_key = f"{source_name}/{class_name}"
    if cache_key in split_cache:
        split_info = split_cache[cache_key]
    else:
        split_path = split_info_dir / source_name / class_name / "split_info.json"
        if split_path.is_file() and not force_split:
            with open(split_path, "r", encoding="utf-8") as f:
                split_info = json.load(f)
            split_info["split_mode"] = split_info.get("split_mode", "pre-defined")
        else:
            all_files = sorted(class_dir.glob("*.npy"))
            sample_ids = [sample_id_from_name(f, ordinal) for ordinal, f in enumerate(all_files)]
            n_total = len(all_files)

            if n_total <= num_shots:
                chosen_positions = list(range(n_total))
                split_mode = "all samples"
            else:
                chosen_positions = sorted(rng.choice(n_total, num_shots, replace=False).tolist())
                split_mode = f"pseudo-random (seed={seed})"

            shot_indices = [sample_ids[i] for i in chosen_positions]
            shot_set = set(shot_indices)
            eval_indices = [sid for sid in sample_ids if sid not in shot_set]

            split_info = {
                "signal_type": signal_type,
                "source": source_name,
                "class": class_name,
                "total_real": n_total,
                "num_shots": len(shot_indices),
                "num_eval": len(eval_indices),
                "shot_indices": shot_indices,
                "eval_indices": eval_indices,
                "seed": seed,
                "split_mode": split_mode,
                "fewshot_offset_pairs": len(shot_indices) * (len(shot_indices) - 1) // 2,
            }

            split_path.parent.mkdir(parents=True, exist_ok=True)
            with open(split_path, "w", encoding="utf-8") as f:
                json.dump(split_info, f, indent=2, ensure_ascii=False)

        split_cache[cache_key] = split_info

    id_to_path = {
        sample_id_from_name(f, ordinal): f
        for ordinal, f in enumerate(sorted(class_dir.glob("*.npy")))
    }
    selected_files = [id_to_path[sid] for sid in split_info.get("shot_indices", []) if sid in id_to_path]
    return selected_files, split_info


# ============================================================
#  Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="SenseGen Stage 2 (Unified) - Behavioral Diversity Characterization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Modality
    parser.add_argument("--signal_type", type=str, default="wifi",
                        choices=["wifi", "mmwave", "ultrasound"],
                        help="Signal modality (default: wifi)")

    # Data
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory (overrides signal_type default)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output root directory (overrides signal_type default)")

    # WiFi-specific
    parser.add_argument("--channels", type=str, nargs="+", default=None,
                        help="[deprecated] Legacy WiFi channel ids; mapped to volunteer_X")

    # mmWave / Ultrasound
    parser.add_argument("--volunteers", type=str, nargs="+", default=None,
                        help="Volunteer/source directories (default: modality default)")

    # Offset
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Max pairs per class (overrides default; 0 = all)")
    parser.add_argument("--sg_window", type=int, default=None,
                        help="S-G window length, must be odd (overrides default)")
    parser.add_argument("--sg_order", type=int, default=None,
                        help="S-G polynomial order (overrides default)")
    parser.add_argument("--sg_axis", type=int, default=None,
                        help="S-G filter axis (overrides default)")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--fewshot_only", action="store_true",
                        help="Extract offsets only among selected few-shot support samples")
    parser.add_argument("--num_shots", type=int, default=3,
                        help="Few-shot support samples per class (default: 3)")
    parser.add_argument("--split_info_dir", type=str, default=None,
                        help="Directory for reusable few-shot split_info.json files")
    parser.add_argument("--force_split", action="store_true",
                        help="Regenerate split info instead of reusing existing files")
    return parser.parse_args()


# ============================================================
#  Main
# ============================================================

def main() -> None:
    args = parse_args()
    signal_type = args.signal_type
    defaults = SIGNAL_DEFAULTS[signal_type]

    # Resolve parameters: CLI overrides > signal defaults
    data_dir = args.data_dir or (defaults["unseen_data_dir"] if args.fewshot_only else defaults["data_dir"])
    output_dir = args.output_dir or (defaults["unseen_output_dir"] if args.fewshot_only else defaults["output_dir"])
    use_channels = defaults["use_channels"]
    sg_window = args.sg_window if args.sg_window is not None else defaults["sg_window"]
    sg_order = args.sg_order if args.sg_order is not None else defaults["sg_order"]
    sg_axis = args.sg_axis if args.sg_axis is not None else defaults["sg_axis"]
    max_pairs = args.max_pairs if args.max_pairs is not None else defaults["max_pairs"]
    max_pairs = max_pairs if max_pairs != 0 else None  # 0 = all pairs
    split_info_dir = args.split_info_dir
    if args.fewshot_only and split_info_dir is None:
        split_info_dir = defaults["split_info_dir"]

    if signal_type == "wifi":
        legacy_channels = args.channels or []
        volunteers = args.volunteers or [normalize_wifi_volunteer(ch) for ch in legacy_channels]
        volunteers = volunteers or defaults["volunteers"]
        volunteers = [normalize_wifi_volunteer(v) for v in volunteers]
        channels = []
        use_channels = False
    elif use_channels:
        channels = args.channels or []
        volunteers = []
    else:
        volunteers = args.volunteers or defaults.get("volunteers", ["volunteer_1"])
        channels = []

    # Validation
    if sg_window % 2 == 0:
        print("[ERROR] sg_window must be odd")
        sys.exit(1)
    if not os.path.isdir(data_dir):
        print(f"[ERROR] data_dir not found: {data_dir}")
        sys.exit(1)

    rng = np.random.RandomState(args.seed)

    # Detect sample shape
    first_sample = None
    if use_channels:
        for ch in channels:
            ch_dir = Path(data_dir) / f"channel_{ch}"
            if ch_dir.is_dir():
                for cls in sorted(ch_dir.iterdir()):
                    if cls.is_dir():
                        files = sorted(cls.glob("*.npy"))
                        if files:
                            first_sample = np.load(files[0])
                            break
                if first_sample is not None:
                    break
    else:
        for v in volunteers:
            vdir = resolve_wifi_source(Path(data_dir), v) if signal_type == "wifi" else Path(data_dir) / v
            if vdir is None:
                continue
            if vdir.is_dir():
                for cls in sorted(vdir.iterdir()):
                    if cls.is_dir():
                        files = sorted(cls.glob("*.npy"))
                        if files:
                            first_sample = np.load(files[0])
                            break
                if first_sample is not None:
                    break

    sample_shape = first_sample.shape if first_sample is not None else "unknown"

    # Header
    print("=" * 60)
    print(f"  SenseGen {signal_type.upper()} - Stage 2: Offset Extraction")
    print("=" * 60)
    print(f"  Data dir      : {data_dir}")
    print(f"  Output dir    : {output_dir}")
    if use_channels:
        print(f"  Channels      : {channels}")
    else:
        print(f"  Volunteers    : {volunteers}")
    print(f"  Sample shape  : {sample_shape}")
    print(f"  Max pairs     : {max_pairs or 'all'}")
    print(f"  S-G window    : {sg_window}")
    print(f"  S-G order     : {sg_order}")
    print(f"  S-G axis      : {sg_axis}")
    print(f"  Seed          : {args.seed}")
    if args.fewshot_only:
        print(f"  Few-shot only : {args.num_shots}-shot")
        print(f"  Split info    : {split_info_dir}")
    print()

    t0 = time.time()
    total_pairs = 0
    total_classes = 0
    split_cache: Dict[str, dict] = {}

    if use_channels:
        # WiFi path: iterate channels → class dirs
        for ch in channels:
            ch_src = Path(data_dir) / f"channel_{ch}"
            if not ch_src.is_dir():
                print(f"[WARN] channel_{ch} not found, skipping")
                continue

            for class_dir in sorted(ch_src.iterdir()):
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name

                static_out = Path(output_dir) / "static" / f"channel_{ch}" / class_name
                dynamic_out = Path(output_dir) / "dynamic" / f"channel_{ch}" / class_name
                sample_files = None
                split_info = None
                if args.fewshot_only:
                    sample_files, split_info = load_or_create_fewshot_split(
                        class_dir=class_dir,
                        class_name=class_name,
                        source_name=f"channel_{ch}",
                        signal_type=signal_type,
                        num_shots=args.num_shots,
                        split_info_dir=Path(split_info_dir),
                        force_split=args.force_split,
                        seed=args.seed,
                        rng=rng,
                        split_cache=split_cache,
                    )

                n_pairs = process_class(
                    class_dir, static_out, dynamic_out,
                    max_pairs, sg_window, sg_order, sg_axis, rng,
                    sample_files=sample_files,
                )
                total_pairs += n_pairs
                total_classes += 1
                n_samples = len(list(class_dir.glob("*.npy")))
                if split_info:
                    print(f"  ch{ch}/{class_name}: {n_samples} samples, "
                          f"{split_info['num_shots']} shots {split_info['shot_indices']} -> {n_pairs} pairs")
                else:
                    print(f"  ch{ch}/{class_name}: {n_samples} samples -> {n_pairs} pairs")
    else:
        # mmWave / Ultrasound path: iterate volunteers → class dirs
        for volunteer in volunteers:
            vol_src = resolve_wifi_source(Path(data_dir), volunteer) if signal_type == "wifi" else Path(data_dir) / volunteer
            if vol_src is None or not vol_src.is_dir():
                print(f"[WARN] {volunteer} not found, skipping")
                continue

            for class_dir in sorted(vol_src.iterdir()):
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name

                static_out = Path(output_dir) / "static" / volunteer / class_name
                dynamic_out = Path(output_dir) / "dynamic" / volunteer / class_name
                sample_files = None
                split_info = None
                if args.fewshot_only:
                    sample_files, split_info = load_or_create_fewshot_split(
                        class_dir=class_dir,
                        class_name=class_name,
                        source_name=volunteer,
                        signal_type=signal_type,
                        num_shots=args.num_shots,
                        split_info_dir=Path(split_info_dir),
                        force_split=args.force_split,
                        seed=args.seed,
                        rng=rng,
                        split_cache=split_cache,
                    )

                n_pairs = process_class(
                    class_dir, static_out, dynamic_out,
                    max_pairs, sg_window, sg_order, sg_axis, rng,
                    sample_files=sample_files,
                )
                total_pairs += n_pairs
                total_classes += 1
                n_samples = len(list(class_dir.glob("*.npy")))
                if split_info:
                    print(f"  {volunteer}/{class_name}: {n_samples} samples, "
                          f"{split_info['num_shots']} shots {split_info['shot_indices']} -> {n_pairs} pairs")
                else:
                    print(f"  {volunteer}/{class_name}: {n_samples} samples -> {n_pairs} pairs")

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"  Total classes : {total_classes}")
    print(f"  Total pairs   : {total_pairs}")
    print(f"  Total files   : {total_pairs * 2} (static + dynamic)")
    print(f"  Time          : {elapsed:.1f}s")
    print(f"  Output        : {output_dir}")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
