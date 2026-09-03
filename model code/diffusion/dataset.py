"""
Dataset loader for SenseGen diffusion model.
Loads wireless sensing arrays and paired behavioral offsets.

Data flow alignment:
  Stage 1 (preprocess.py): zero-pad + normalize -> modality data directory
  Stage 2 (extract_offset.py): paired diff + S-G filter -> offsets_WiFi/{static,dynamic}/
  Stage 3 (this module): loads both and pairs them for diffusion training
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


VALID_OFFSET_MODES = {"both", "static", "dynamic_condition", "dynamic"}
MODALITY_DEFAULT_SOURCES = {
    "wifi": ["volunteer_0"],
    "mmwave": ["volunteer_1"],
    "ultrasound": ["volunteer_5"],
}


def normalize_wifi_volunteer(value) -> str:
    value = str(value)
    if value.startswith("volunteer_"):
        return value
    if value.startswith("channel_"):
        value = value[len("channel_"):]
    return f"volunteer_{value}"


def resolve_source_dir(root: Path, source, source_prefix: str, signal_type: str) -> Path:
    names = [f"{source_prefix}{source}" if source_prefix else str(source)]
    if signal_type == "wifi":
        canonical = normalize_wifi_volunteer(source)
        names.extend([canonical, f"channel_{canonical[len('volunteer_'):]}"])
    for name in dict.fromkeys(names):
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return root / names[0]


class SenseGenDataset(Dataset):
    """Dataset for modality-specific sensing arrays with offset conditioning.

    Expected directory layout (from Stage 1 & 2):
        {data_dir}/
          volunteer_X/
            class_XX/
              sample_00.npy   (64, 256) float32
              ...

        {offset_dir}/
          static/
            volunteer_X/
              class_XX/
                pair_00_01.npy   (64, 256) float32
                ...
          dynamic/
            volunteer_X/
              class_XX/
                pair_00_01.npy   (64, 256) float32
                ...

    Each item returns:
        sample:     (1, H, W) float32 — clean spectrogram (normalized)
        cond:       (2, H, W) float32 — [static_offset, dynamic_offset] (normalized)
        class_name: str
        sample_idx: int
    """

    def __init__(
        self,
        data_dir: str = "WiFi data_pad/base",
        offset_dir: str = "offsets_WiFi",
        sources: Optional[List] = None,
        channels: Optional[List] = None,
        classes: Optional[List[str]] = None,
        normalize: bool = True,
        signal_type: str = "wifi",
        source_prefix: str = "",
        offset_mode: str = "both",
        offset_scale_min: float = 1.0,
        offset_scale_max: float = 1.0,
        max_samples_per_epoch: Optional[int] = None,
    ):
        super().__init__()
        if offset_mode not in VALID_OFFSET_MODES:
            raise ValueError(
                f"Invalid offset_mode={offset_mode!r}; "
                f"expected one of {sorted(VALID_OFFSET_MODES)}"
            )
        self.data_dir = Path(data_dir)
        self.offset_dir = Path(offset_dir)
        raw_sources = sources if sources is not None else channels
        raw_sources = raw_sources or MODALITY_DEFAULT_SOURCES.get(signal_type, ["volunteer_0"])
        self.sources = [normalize_wifi_volunteer(v) for v in raw_sources] if signal_type == "wifi" else list(raw_sources)
        self.channels = self.sources  # backward-compatible attribute
        self.source_prefix = source_prefix
        self.normalize = normalize
        self.offset_mode = offset_mode
        self.offset_scale_min = offset_scale_min
        self.offset_scale_max = offset_scale_max
        self.max_samples_per_epoch = max_samples_per_epoch
        self.signal_type = signal_type

        # Set source_prefix from signal_type
        if signal_type in ("mmwave", "ultrasound"):
            self.source_prefix = ""

        # Build index: list of (source, class_name, sample_idx, sample_path)
        self._samples: List[Tuple[str, str, int, Path]] = []
        # Per-sample offset map: sample_path -> list of (static_path, dynamic_path)
        self._offset_map: Dict[Path, List[Tuple[Path, Path]]] = {}
        # Active sample subset used by __getitem__; resampled each epoch
        self._active_samples: List[Tuple[str, str, int, Path]] = []

        self._build_index(classes)
        self.resample_samples(seed=None)

        # Detect sample shape from first file (was hardcoded 256x128)
        self.sample_shape = None
        if self._samples:
            first_path = self._samples[0][3]
            self.sample_shape = np.load(first_path).shape
            print(f"[SenseGenDataset] Sample shape: {self.sample_shape}")

        if self.normalize:
            self._compute_stats()

    def _build_index(self, classes: Optional[List[str]]):
        """Scan data directory and build sample <-> offset mapping."""
        for ch in self.sources:
            ch_dir = resolve_source_dir(self.data_dir, ch, self.source_prefix, self.signal_type)
            if not ch_dir.is_dir():
                print(f"[SenseGenDataset] WARN: source {ch} not found at {ch_dir}")
                continue

            for class_dir in sorted(ch_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name
                if classes and class_name not in classes:
                    continue

                for fpath in sorted(class_dir.glob("sample_*.npy")):
                    match = re.match(r"sample_(\d+)\.npy", fpath.name)
                    if not match:
                        continue
                    sample_idx = int(match.group(1))
                    self._samples.append((ch, class_name, sample_idx, fpath))

                    static_dir = resolve_source_dir(self.offset_dir / "static", ch, self.source_prefix, self.signal_type) / class_name
                    dynamic_dir = resolve_source_dir(self.offset_dir / "dynamic", ch, self.source_prefix, self.signal_type) / class_name

                    offsets = self._find_sample_offsets(sample_idx, static_dir, dynamic_dir)
                    self._offset_map[fpath] = offsets

        n_classes = len(set(s[1] for s in self._samples))
        print(f"[SenseGenDataset] {len(self._samples)} samples, "
              f"{n_classes} classes, {len(self.sources)} sources")
        print(f"[SenseGenDataset] Offset mode: {self.offset_mode}")

    @staticmethod
    def _find_sample_offsets(
        sample_idx: int,
        static_dir: Path,
        dynamic_dir: Path,
    ) -> List[Tuple[Path, Path]]:
        """Find all offset pairs that include the given sample index.

        Matches Stage 2 naming: pair_{i}_{j}.npy
        """
        offsets = []
        if not static_dir.is_dir() or not dynamic_dir.is_dir():
            return offsets

        for fname in sorted(static_dir.glob("*pair_*.npy")):
            match = re.match(r"pair_(\d+)_(\d+)\.npy", fname.name)
            if not match:
                continue
            i, j = int(match.group(1)), int(match.group(2))
            if sample_idx in (i, j):
                dpath = dynamic_dir / fname.name
                if dpath.is_file():
                    offsets.append((static_dir / fname.name, dpath))

        return offsets

    def resample_samples(self, seed: Optional[int] = None):
        """Randomly select up to max_samples_per_epoch samples for this epoch.

        When max_samples_per_epoch is None, the active set is the full sample
        list. Otherwise, samples are sampled without replacement using a
        reproducible RandomState derived from the provided seed.

        Args:
            seed: Random seed for this epoch. None = fresh entropy.
        """
        if self.max_samples_per_epoch is None:
            self._active_samples = list(self._samples)
            return

        rng = np.random.RandomState(seed)
        if len(self._samples) <= self.max_samples_per_epoch:
            self._active_samples = list(self._samples)
        else:
            idxs = rng.choice(
                len(self._samples),
                size=self.max_samples_per_epoch,
                replace=False,
            )
            idxs = sorted(idxs)
            self._active_samples = [self._samples[i] for i in idxs]

    def _compute_stats(self):
        """Compute global mean/std across all loaded samples."""
        if len(self._samples) == 0:
            raise RuntimeError(
                f"No .npy samples found in {self.data_dir}. "
                f"Please check:\n"
                f"  1. Stage 1 (preprocess.py) has been run\n"
                f"  2. --data_dir points to the correct directory\n"
                f"     e.g. --data_dir '../WiFi data_pad/base'\n"
                f"  3. The directory contains volunteer_*/class_*/sample_*.npy files"
            )

        all_sum = 0.0
        all_sq = 0.0
        count = 0

        for _, _, _, path in self._samples:
            arr = np.load(path).astype(np.float32)
            all_sum += arr.sum()
            all_sq += (arr ** 2).sum()
            count += arr.size

        self.mean = float(all_sum / count)
        self.std = float(max((all_sq / count - self.mean ** 2) ** 0.5, 1e-6))
        print(f"[SenseGenDataset] Normalization: mean={self.mean:.6f}, std={self.std:.6f}")

    def __len__(self) -> int:
        return len(self._active_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
        ch, class_name, sample_idx, sample_path = self._active_samples[idx]

        # Load spectrogram
        sample = np.load(sample_path).astype(np.float32)
        sample = torch.from_numpy(sample).unsqueeze(0)  # (1, H, W)

        if self.normalize:
            sample = (sample - self.mean) / self.std

        # Randomly pick one offset pair for this sample
        offsets = self._offset_map.get(sample_path, [])
        if len(offsets) == 0:
            # Fallback: zero offset (should not happen if Stage 2 ran)
            static = torch.zeros(1, *self.sample_shape, dtype=torch.float32)
            dynamic = torch.zeros(1, *self.sample_shape, dtype=torch.float32)
        else:
            idx_pair = np.random.randint(0, len(offsets))
            static_path, dynamic_path = offsets[idx_pair]
            static = torch.from_numpy(
                np.load(static_path).astype(np.float32)
            ).unsqueeze(0)
            dynamic = torch.from_numpy(
                np.load(dynamic_path).astype(np.float32)
            ).unsqueeze(0)
            expected = (1, *self.sample_shape)
            assert static.shape == expected, f"Bad static shape: {static.shape}, expected {expected}"
            assert dynamic.shape == expected, f"Bad dynamic shape: {dynamic.shape}, expected {expected}"

            if self.normalize:
                static = static / (self.std + 1e-8)
                dynamic = dynamic / (self.std + 1e-8)

            # Offset scaling: fixed if min==max, random in [min, max] otherwise
            if self.offset_scale_min < 1.0 or self.offset_scale_max < 1.0:
                scale = np.random.uniform(self.offset_scale_min, self.offset_scale_max)
                static = static * scale
                dynamic = dynamic * scale

        if self.offset_mode == "static":
            dynamic = torch.zeros_like(dynamic)
        elif self.offset_mode in ("dynamic_condition", "dynamic"):
            static = torch.zeros_like(static)

        cond = torch.cat([static, dynamic], dim=0)  # (2, H, W)

        return sample, cond, class_name, sample_idx


class OffsetPoolSampler:
    """Manages offset pool for cross-class / unseen-class generation.

    Collects offset pairs across specified classes for transfer sampling.
    """

    def __init__(
        self,
        offset_dir: str = "offsets_WiFi",
        sources: Optional[List] = None,
        channels: Optional[List] = None,
        classes: Optional[List[str]] = None,
        sample_shape: Optional[Tuple[int, int]] = None,
        signal_type: str = "wifi",
        source_prefix: str = "",
    ):
        self.offset_dir = Path(offset_dir)
        raw_sources = sources if sources is not None else channels
        raw_sources = raw_sources or MODALITY_DEFAULT_SOURCES.get(signal_type, ["volunteer_0"])
        self.sources = [normalize_wifi_volunteer(v) for v in raw_sources] if signal_type == "wifi" else list(raw_sources)
        self.channels = self.sources
        self.source_prefix = source_prefix
        self.sample_shape = sample_shape or (64, 256)
        self.signal_type = signal_type

        # Set source_prefix from signal_type
        if signal_type in ("mmwave", "ultrasound"):
            self.source_prefix = ""

        # All offsets: list of (static_path, dynamic_path, channel, class_name)
        self._offsets: List[Tuple[Path, Path, int, str]] = []
        self._build_pool(classes)

    def _resolve_index_path(self, raw_path: str, index_path: Path) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return (index_path.parent / path).resolve()

    def _load_index_pool(self, classes: Optional[List[str]]) -> bool:
        found_index = False
        for ch in self.sources:
            index_base = resolve_source_dir(self.offset_dir / "index", ch, self.source_prefix, self.signal_type)
            if not index_base.is_dir():
                continue

            if classes:
                index_files = [index_base / f"{cls}.json" for cls in classes]
            else:
                index_files = sorted(index_base.glob("*.json"))

            for index_path in index_files:
                if not index_path.is_file():
                    continue
                found_index = True
                with open(index_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                class_name = payload.get("unseen_class") or index_path.stem
                for item in payload.get("selected_offsets", []):
                    static_raw = item.get("static_path")
                    dynamic_raw = item.get("dynamic_path")
                    if not static_raw or not dynamic_raw:
                        continue
                    static_path = self._resolve_index_path(static_raw, index_path)
                    dynamic_path = self._resolve_index_path(dynamic_raw, index_path)
                    if static_path.is_file() and dynamic_path.is_file():
                        self._offsets.append((static_path, dynamic_path, ch, class_name))
        return found_index

    def _build_pool(self, classes: Optional[List[str]]):
        if self._load_index_pool(classes):
            print(f"[OffsetPool] Loaded indexed pool from {self.offset_dir}")
            print(f"[OffsetPool] {len(self._offsets)} offset pairs across "
                  f"{len(set(o[2] for o in self._offsets))} sources, "
                  f"{len(set(o[3] for o in self._offsets))} classes")
            return

        for ch in self.sources:
            static_base = resolve_source_dir(self.offset_dir / "static", ch, self.source_prefix, self.signal_type)
            if not static_base.is_dir():
                continue
            for class_dir in sorted(static_base.iterdir()):
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name
                if classes and class_name not in classes:
                    continue
                dynamic_dir = resolve_source_dir(self.offset_dir / "dynamic", ch, self.source_prefix, self.signal_type) / class_name
                for fpath in sorted(class_dir.glob("*pair_*.npy")):
                    dpath = dynamic_dir / fpath.name
                    if dpath.is_file():
                        self._offsets.append((fpath, dpath, ch, class_name))

        print(f"[OffsetPool] {len(self._offsets)} offset pairs across "
              f"{len(set(o[2] for o in self._offsets))} sources, "
              f"{len(set(o[3] for o in self._offsets))} classes")

    def sample(
        self,
        n: int = 1,
        source: Optional[str] = None,
        channel: Optional[int] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample random offset pairs from the pool.

        Returns:
            static:  (n, 1, H, W)
            dynamic: (n, 1, H, W)
        """
        if rng is None:
            rng = np.random.RandomState()

        pool = self._offsets
        requested_source = source if source is not None else channel
        if requested_source is not None:
            requested_source = normalize_wifi_volunteer(requested_source) if self.signal_type == "wifi" else requested_source
            pool = [o for o in pool if o[2] == requested_source]

        if len(pool) == 0:
            shape = self.sample_shape or (64, 256)
            return (
                torch.zeros(n, 1, *shape, dtype=torch.float32),
                torch.zeros(n, 1, *shape, dtype=torch.float32),
            )

        idxs = rng.choice(len(pool), size=n, replace=True)
        static_list, dynamic_list = [], []
        for idx in idxs:
            s_path, d_path, _, _ = pool[idx]
            static_list.append(np.load(s_path).astype(np.float32))
            dynamic_list.append(np.load(d_path).astype(np.float32))

        static = torch.from_numpy(np.stack(static_list, axis=0)).unsqueeze(1)
        dynamic = torch.from_numpy(np.stack(dynamic_list, axis=0)).unsqueeze(1)
        return static, dynamic
