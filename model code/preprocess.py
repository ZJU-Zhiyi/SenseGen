#!/usr/bin/env python3
"""
Unified SenseGen Stage 1 preprocessor.

Supports:
  wifi       : (61, 250) -> zero-pad -> (64, 256)
  mmwave     : (1, 17, 256, 128) -> range mean/log/z-score -> (16, 128)
  ultrasound : (224, 224) -> zero-pad -> (256, 256)

The modality-specific scripts remain available, but this file is the stable
entry point used by the usage guides.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


WIFI_SOURCE_SHAPE = (61, 250)
WIFI_TARGET_SHAPE = (64, 256)

MMWAVE_SOURCE_SHAPE = (1, 17, 256, 128)
MMWAVE_TARGET_SHAPE = (16, 128)
MMWAVE_DROP_FRAME_DEFAULT = 15
MMWAVE_LOG_EPS_DEFAULT = 1e-3

ULTRASOUND_SOURCE_SHAPE = (224, 224)
ULTRASOUND_TARGET_SHAPE = (256, 256)
ULTRASOUND_ID_PATTERN = re.compile(r"-(\d+)_mat_\d+\.npy$")

DEFAULTS = {
    "wifi": {
        "src": "WiFi data",
        "dst": "WiFi data_pad",
        "channels": [],
        "volunteers": ["volunteer_0"],
    },
    "mmwave": {
        "src": "mmWave_data",
        "dst": "mmWave_data_proc",
        "channels": [],
        "volunteers": ["volunteer_1"],
    },
    "ultrasound": {
        "src": "ultrasound data_split",
        "dst": "ultrasound data_proc",
        "channels": [],
        "volunteers": ["volunteer_5"],
    },
}


def pad_2d(src_path: Path, source_shape: Tuple[int, int], target_shape: Tuple[int, int]) -> np.ndarray:
    arr = np.load(src_path).astype(np.float32)
    if arr.ndim != 2 or arr.shape != source_shape:
        raise ValueError(f"Unexpected shape {arr.shape}; expected {source_shape}")
    padded = np.zeros(target_shape, dtype=np.float32)
    padded[: source_shape[0], : source_shape[1]] = arr
    return padded


def process_wifi_file(src_path: Path) -> np.ndarray:
    return pad_2d(src_path, WIFI_SOURCE_SHAPE, WIFI_TARGET_SHAPE)


def process_ultrasound_file(src_path: Path) -> np.ndarray:
    return pad_2d(src_path, ULTRASOUND_SOURCE_SHAPE, ULTRASOUND_TARGET_SHAPE)


def process_mmwave_file(src_path: Path, drop_frame: int, log_eps: float) -> np.ndarray:
    arr = np.load(src_path).astype(np.float32)
    if arr.ndim != 4 or arr.shape != MMWAVE_SOURCE_SHAPE:
        raise ValueError(f"Unexpected shape {arr.shape}; expected {MMWAVE_SOURCE_SHAPE}")
    arr = arr[0]
    arr = np.mean(arr, axis=1)
    arr = np.delete(arr, drop_frame, axis=0)
    return np.log10(arr + log_eps).astype(np.float32)


def normalize_wifi_volunteer(value: str) -> str:
    """Return the canonical WiFi source name (``volunteer_X``)."""
    value = str(value)
    if value.startswith("volunteer_"):
        return value
    if value.startswith("channel_"):
        value = value[len("channel_"):]
    return f"volunteer_{value}"


def legacy_channel_name(volunteer: str) -> str:
    suffix = normalize_wifi_volunteer(volunteer)[len("volunteer_"):]
    return f"channel_{suffix}"


def discover_wifi_source_dirs(src_root: Path, volunteers: List[str]):
    """Yield ``(split, volunteer, directory)`` with legacy channel fallback."""
    splits = [None]
    splits.extend(
        d.name for d in sorted(src_root.iterdir())
        if d.is_dir() and d.name in {"base", "unseen", "reserved"}
    )
    seen = set()
    for split in splits:
        parent = src_root if split is None else src_root / split
        for raw_volunteer in volunteers:
            volunteer = normalize_wifi_volunteer(raw_volunteer)
            candidates = [parent / volunteer, parent / legacy_channel_name(volunteer)]
            source_dir = next((p for p in candidates if p.is_dir()), None)
            if source_dir is None:
                continue
            key = source_dir.resolve()
            if key in seen:
                continue
            seen.add(key)
            yield split, volunteer, source_dir


def collect_split_files(split_src: Path, volunteers: List[str]) -> List[Path]:
    files: List[Path] = []
    for volunteer in volunteers:
        vol_src = split_src / volunteer
        if not vol_src.is_dir():
            print(f"  [WARN] Volunteer dir not found: {vol_src}, skipping.")
            continue
        files.extend(sorted(vol_src.rglob("*.npy")))
    return sorted(files)


def compute_mmwave_stats(files: List[Path], drop_frame: int, log_eps: float) -> Tuple[float, float]:
    if not files:
        return 0.0, 1.0

    n = 0
    mean = 0.0
    m2 = 0.0
    for fpath in files:
        values = process_mmwave_file(fpath, drop_frame, log_eps).ravel()
        for x in values:
            n += 1
            delta = float(x) - mean
            mean += delta / n
            delta2 = float(x) - mean
            m2 += delta * delta2
    return float(mean), float(np.sqrt(m2 / max(n - 1, 1)))


def write_array(src_path: Path, dst_path: Path, arr: np.ndarray, dry_run: bool) -> None:
    if dry_run:
        print(f"  {src_path} -> {dst_path}")
        return
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst_path, arr.astype(np.float32))


def run_wifi(args) -> None:
    src_root = Path(args.src)
    if not src_root.is_dir():
        raise SystemExit(f"[ERROR] Source directory not found: {src_root}")

    volunteers = [normalize_wifi_volunteer(v) for v in args.volunteers]
    sources = list(discover_wifi_source_dirs(src_root, volunteers))
    files = [path for _, _, source_dir in sources for path in source_dir.rglob("*.npy")]

    print("=" * 60)
    print("SenseGen WiFi -- Stage 1: Preprocessing")
    print("=" * 60)
    print(f"  Source      : {src_root}")
    print(f"  Destination : {args.dst}")
    print(f"  Volunteers  : {volunteers}")
    print(f"  Dry run     : {args.dry_run}")
    print(f"  Shape       : {WIFI_SOURCE_SHAPE} -> {WIFI_TARGET_SHAPE}")
    print(f"  Files       : {len(files)}")
    print()

    t0 = time.time()
    processed = 0
    skipped = 0
    for split, volunteer, source_dir in sources:
        for src_path in sorted(source_dir.rglob("*.npy")):
            try:
                arr = process_wifi_file(src_path)
            except Exception as exc:
                print(f"  [SKIP] {src_path}: {exc}")
                skipped += 1
                continue
            rel = src_path.relative_to(source_dir)
            dst_root = Path(args.dst) / split if split is not None else Path(args.dst)
            write_array(src_path, dst_root / volunteer / rel, arr, args.dry_run)
            processed += 1

    elapsed = time.time() - t0
    if args.dry_run:
        print(f"\nDone. Dry run: {processed} files checked, {skipped} skipped in {elapsed:.1f}s")
    else:
        print(f"\nDone. {processed} processed, {skipped} skipped in {elapsed:.1f}s")


def run_mmwave(args) -> None:
    src_root = Path(args.src)
    if not src_root.is_dir():
        raise SystemExit(f"[ERROR] Source directory not found: {src_root}")

    print("=" * 60)
    print("SenseGen mmWave -- Stage 1: Preprocessing")
    print("=" * 60)
    print(f"  Source      : {src_root}")
    print(f"  Destination : {args.dst}")
    print(f"  Splits      : {args.splits}")
    print(f"  Volunteers  : {args.volunteers}")
    print(f"  Drop frame  : {args.drop_frame}")
    print(f"  Log epsilon : {args.log_eps}")
    print(f"  Dry run     : {args.dry_run}")
    print(f"  Shape       : {MMWAVE_SOURCE_SHAPE} -> {MMWAVE_TARGET_SHAPE}")
    print()

    t0 = time.time()
    base_mean: Optional[float] = None
    base_std: Optional[float] = None
    stats = {
        "signal_type": "mmwave",
        "drop_frame": args.drop_frame,
        "log_eps": args.log_eps,
        "input_shape": list(MMWAVE_SOURCE_SHAPE),
        "output_shape": list(MMWAVE_TARGET_SHAPE),
        "splits": {},
    }

    for split in args.splits:
        split_src = src_root / split
        split_dst = Path(args.dst) / split
        if not split_src.is_dir():
            print(f"  [WARN] Split dir not found: {split_src}, skipping.")
            continue

        files = collect_split_files(split_src, args.volunteers)
        if split != "base" and args.shared_stats and base_mean is not None and base_std is not None:
            mean, std = base_mean, base_std
            print(f"\n[{split}] Using base stats: mean={mean:.6f}, std={std:.6f}")
        else:
            print(f"  Computing normalization stats for [{split}] ...")
            mean, std = compute_mmwave_stats(files, args.drop_frame, args.log_eps)
            print(f"    mean={mean:.6f}, std={std:.6f}")

        processed = 0
        skipped = 0
        if args.dry_run:
            print(f"  [{split}] {len(files)} files would be processed")
        else:
            for src_path in files:
                try:
                    arr = process_mmwave_file(src_path, args.drop_frame, args.log_eps)
                    arr = (arr - mean) / (std + 1e-8)
                except Exception as exc:
                    print(f"  [SKIP] {src_path}: {exc}")
                    skipped += 1
                    continue
                rel = src_path.relative_to(split_src)
                write_array(src_path, split_dst / rel, arr, args.dry_run)
                processed += 1
            print(f"  [{split}] {processed} processed, {skipped} skipped")

        stats["splits"][split] = {"mean": mean, "std": std}
        if split == "base":
            base_mean, base_std = mean, std

    elapsed = time.time() - t0
    if args.dry_run:
        print(f"\nDone in {elapsed:.1f}s. Dry run: no files written.")
        return

    os.makedirs(args.dst, exist_ok=True)
    stats_path = Path(args.dst) / "preprocess_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nDone in {elapsed:.1f}s. Stats saved to {stats_path}")


def run_ultrasound(args) -> None:
    src_root = Path(args.src)
    if not src_root.is_dir():
        raise SystemExit(f"[ERROR] Source directory not found: {src_root}")

    flat_mode = not args.splits

    print("=" * 60)
    print("SenseGen Ultrasound -- Stage 1: Preprocessing")
    print("=" * 60)
    print(f"  Source       : {src_root}")
    print(f"  Destination  : {args.dst}")
    print(f"  Mode         : {'flat' if flat_mode else 'split'}")
    if not flat_mode:
        print(f"  Splits       : {args.splits}")
    print(f"  Volunteers   : {args.volunteers}")
    print(f"  Dry run      : {args.dry_run}")
    print(f"  Shape        : {ULTRASOUND_SOURCE_SHAPE} -> {ULTRASOUND_TARGET_SHAPE}")
    print()

    t0 = time.time()
    total_processed = 0
    total_skipped = 0

    jobs = []
    if flat_mode:
        for volunteer in args.volunteers:
            jobs.append((src_root / volunteer, Path(args.dst) / volunteer, volunteer))
    else:
        for split in args.splits:
            for volunteer in args.volunteers:
                label = f"{split}/{volunteer}"
                jobs.append((src_root / split / volunteer, Path(args.dst) / split / volunteer, label))

    for src_dir, dst_dir, label in jobs:
        if not src_dir.is_dir():
            print(f"  [WARN] Source dir not found: {src_dir}, skipping.")
            continue
        files = sorted(src_dir.rglob("*.npy"))
        processed = 0
        skipped = 0
        for src_path in files:
            try:
                arr = process_ultrasound_file(src_path)
            except Exception as exc:
                print(f"  [SKIP] {src_path}: {exc}")
                skipped += 1
                continue
            rel = src_path.relative_to(src_dir)
            match = ULTRASOUND_ID_PATTERN.search(src_path.name)
            if match is not None:
                rel = rel.with_name(f"sample_{int(match.group(1)):02d}.npy")
            write_array(src_path, dst_dir / rel, arr, args.dry_run)
            processed += 1
        total_processed += processed
        total_skipped += skipped
        action = "would process" if args.dry_run else "processed"
        print(f"  [{label}] {processed} {action}, {skipped} skipped")

    elapsed = time.time() - t0
    if args.dry_run:
        print(f"\nDone. Dry run: {total_processed} files checked, {total_skipped} skipped in {elapsed:.1f}s")
        return

    os.makedirs(args.dst, exist_ok=True)
    meta = {
        "signal_type": "ultrasound",
        "input_shape": list(ULTRASOUND_SOURCE_SHAPE),
        "output_shape": list(ULTRASOUND_TARGET_SHAPE),
        "dtype": "float32",
        "normalization": "none (data already pre-normalized to [0,1])",
        "mode": "flat" if flat_mode else "split",
        "total_processed": total_processed,
        "total_skipped": total_skipped,
    }
    meta_path = Path(args.dst) / "preprocess_stats.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {total_processed} processed, {total_skipped} skipped in {elapsed:.1f}s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified SenseGen Stage 1 preprocessor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python "model code/preprocess.py" --signal_type wifi --volunteers volunteer_0
  python "model code/preprocess.py" --signal_type mmwave --volunteers volunteer_2 --shared_stats
  python "model code/preprocess.py" --signal_type ultrasound --volunteers volunteer_2 --splits base unseen
        """,
    )
    parser.add_argument("--signal_type", choices=["wifi", "mmwave", "ultrasound"], default="wifi")
    parser.add_argument("--src", type=str, default=None, help="Source root directory")
    parser.add_argument("--dst", type=str, default=None, help="Output root directory")
    parser.add_argument("--channels", nargs="+", default=None,
                        help="[deprecated] Legacy WiFi channel ids; mapped to volunteer_X")
    parser.add_argument("--volunteers", nargs="+", default=None,
                        help="Volunteer/source directories (default: modality default)")
    parser.add_argument("--splits", nargs="*", default=None,
                        help="Split names. mmWave default: base unseen; ultrasound default: flat mode")
    parser.add_argument("--drop_frame", type=int, default=MMWAVE_DROP_FRAME_DEFAULT,
                        help="mmWave frame index to drop")
    parser.add_argument("--log_eps", type=float, default=MMWAVE_LOG_EPS_DEFAULT,
                        help="mmWave log10(x+eps) epsilon")
    parser.add_argument("--shared_stats", action="store_true",
                        help="mmWave: use base stats for non-base splits")
    parser.add_argument("--dry_run", action="store_true",
                        help="List/check files without writing outputs")

    args = parser.parse_args()
    defaults = DEFAULTS[args.signal_type]
    if args.src is None:
        args.src = defaults["src"]
    if args.dst is None:
        args.dst = defaults["dst"]
    if args.channels is None:
        args.channels = defaults["channels"]
    if args.volunteers is None:
        if args.signal_type == "wifi" and args.channels:
            args.volunteers = [normalize_wifi_volunteer(ch) for ch in args.channels]
        else:
            args.volunteers = defaults["volunteers"]
    if args.signal_type == "mmwave" and args.splits is None:
        args.splits = ["base", "unseen"]
    return args


def main() -> None:
    args = parse_args()
    if args.signal_type == "wifi":
        run_wifi(args)
    elif args.signal_type == "mmwave":
        run_mmwave(args)
    elif args.signal_type == "ultrasound":
        run_ultrasound(args)
    else:
        raise SystemExit(f"[ERROR] Unsupported signal_type: {args.signal_type}")


if __name__ == "__main__":
    main()
