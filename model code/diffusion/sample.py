"""
Sampling / Generation script for SenseGen Diffusion Model.
Default: full-chain DDPM, 3-shot generation with 20 virtual samples per shot.
Pseudo-random shot selection with reproducibility via fixed seed.

Data flow (aligned with upstream Stage 1 & 2):
  Stage 1 (preprocess_*.py): zero-pad + normalize -> preprocessed .npy
  Stage 2 (extract_offset.py): paired diff + S-G filter -> modality offset dirs
  Stage 3 (this script): offset-conditioned DDPM/SDEdit -> generated .npy

Supports WiFi, mmWave, and Ultrasound via --signal_type.
When --base_offset_dir points to Siamese-filtered offsets, use --per_class_offsets
to select class-specific pools for unseen class generation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

# Support both module import and direct script execution
try:
    from .config import SenseGenConfig, UNetConfig, DiffusionConfig
    from .unet import UNetModel
    from .diffusion import GaussianDiffusion
    from .dataset import OffsetPoolSampler, normalize_wifi_volunteer, resolve_source_dir
except ImportError:
    from config import SenseGenConfig, UNetConfig, DiffusionConfig
    from unet import UNetModel
    from diffusion import GaussianDiffusion
    from dataset import OffsetPoolSampler, normalize_wifi_volunteer, resolve_source_dir


MODALITY_SAMPLE_BATCH_DEFAULTS = {
    "wifi": 20,
    "mmwave": 20,
    "ultrasound": 20,
}

MODALITY_DATA_DEFAULTS = {
    "wifi": "WiFi data_pad/unseen",
    "mmwave": "mmWave_data_proc/unseen",
    "ultrasound": "ultrasound data_proc/unseen",
}

MODALITY_SOURCE_DEFAULTS = {
    "wifi": "volunteer_0",
    "mmwave": "volunteer_1",
    "ultrasound": "volunteer_5",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="SenseGen Stage 3 -- few-shot wireless sensing generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use defaults from checkpoint
  python sample.py --checkpoint ckpt.pt --device cuda

  # Override data paths
  python sample.py --checkpoint ckpt.pt --data_dir "WiFi data_pad/unseen"

  # Specify class and seed
  python sample.py --checkpoint ckpt.pt --class_name class_00 --seed 42

  # Multiple classes
  python sample.py --checkpoint ckpt.pt --class_name "class_00,class_01" --device cuda

  # Manual shot indices
  python sample.py --checkpoint ckpt.pt --class_name class_03 --shot_indices "0,5"
        """,
    )

    # Checkpoint (required)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--device", type=str, default="cuda")

    # Few-shot generation
    parser.add_argument("--num_shots", type=int, default=3,
                        help="Number of real samples to use as conditioning anchors")
    parser.add_argument("--samples_per_shot", type=int, default=20,
                        help="Virtual samples to generate per real sample")
    parser.add_argument("--sample_batch_size", type=int, default=None,
                        help="Generation batch size per denoising call "
                             "(default: modality tuned; wifi/mmwave/ultrasound=20)")

    # Data paths (override checkpoint defaults)
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory with real spectrograms "
                             "(default: from checkpoint config)")
    parser.add_argument("--base_offset_dir", type=str, default=None,
                        help="Base offset pool directory for unseen generation "
                             "(set to Siamese-filtered dir for filtered offsets)")

    # Sample selection
    parser.add_argument("--signal_type", type=str, default=None,
                        choices=["wifi", "mmwave", "ultrasound"],
                        help="Signal type: wifi, mmwave, or ultrasound (auto-detected from checkpoint if not set)")
    parser.add_argument("--source", "--volunteer", dest="source", type=str, default=None,
                        help="Volunteer/source directory (default: modality default)")
    parser.add_argument("--channel", type=str, default=None,
                        help="[deprecated] Legacy WiFi channel id; mapped to volunteer_X")
    parser.add_argument("--class_name", type=str, default=None,
                        help="Class to sample from. Comma-separated for multiple (e.g. class_00,class_01)."
                        " Default: all classes.")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Number of classes to generate for (first N sorted; used when --class_name not set)")
    parser.add_argument("--shot_indices", type=str, default=None,
                        help="Comma-separated sample indices (e.g. '0,5'). "
                             "If not set, pseudo-randomly selected via --seed.")
    parser.add_argument("--split_info_dir", type=str, default=None,
                        help="Directory with per-class split_info.json files from extract_offset.py")

    # Normalization overrides (read from checkpoint by default)
    parser.add_argument("--data_mean", type=float, default=None,
                        help="Override dataset mean (default: from checkpoint)")
    parser.add_argument("--data_std", type=float, default=None,
                        help="Override dataset std (default: from checkpoint)")

    parser.add_argument("--cfg_scale", type=float, default=1.0,
                        help="Classifier-Free Guidance scale (1.0 = no guidance, 2.0-5.0 recommended)")
    parser.add_argument("--offset_scale", type=float, default=1.0,
                        help="Scale factor for offset conditioning (1.0=full, 0.5=half, 0.0=zero offsets)")
    parser.add_argument("--offset_mode", type=str, default=None,
                        choices=["both", "static", "dynamic_condition", "dynamic"],
                        help="Override checkpoint offset ablation mode. Default: use checkpoint config.")
    parser.add_argument("--sdedit_t", type=int, default=None,
                        help="SDEdit starting timestep (0-1000). Noises anchor to this step then denoises with offset. "
                             "None = start from pure noise (original). Recommended: 300-700 (higher = more variation)")
    parser.add_argument("--denormalize", dest="denormalize", action="store_true", default=True,
                        help="Denormalize generated outputs back to data scale (default: enabled)")
    parser.add_argument("--no_denormalize", "--keep_normalized", dest="denormalize", action="store_false",
                        help="Keep generated outputs in normalized model scale")
    parser.add_argument("--per_class_offsets", action="store_true",
                        help="Use class-specific filtered offsets (one pool per unseen class). "
                             "Requires --base_offset_dir pointing to a Siamese-filtered directory.")

    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for generated .npy files")
    parser.add_argument("--progress", action="store_true", default=False,
                        help="Show sampling progress bar")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ---------------------------------------------------------------------------
#  Data scanning  -- matches Stage 1 (preprocess.py) output structure
# ---------------------------------------------------------------------------

def _discover_classes(data_dir: str, source, source_prefix: str = "",
                      class_prefix: str = "class_", signal_type: str = "wifi") -> List[str]:
    """Discover all class directories in the data directory.

    If class_prefix is non-empty (WiFi/mmWave), filters by prefix.
    If class_prefix is empty (Ultrasound), takes all subdirs.
    """
    ch_dir = resolve_source_dir(Path(data_dir), source, source_prefix, signal_type)
    if not ch_dir.is_dir():
        return []
    if class_prefix:
        classes = sorted(
            d.name for d in ch_dir.iterdir()
            if d.is_dir() and d.name.startswith(class_prefix)
        )
    else:
        classes = sorted(
            d.name for d in ch_dir.iterdir()
            if d.is_dir()
        )
    return classes


def scan_samples(
    data_dir: str,
    source,
    class_name: Optional[str] = None,
    source_prefix: str = "",
    signal_type: str = "wifi",
) -> List[Tuple[Path, str, int]]:
    """Scan data directory  -- matches preprocess.py output layout.

    Expected structure (from Stage 1):
        WiFi:       {data_dir}/volunteer_X/class_XX/sample_*.npy
        mmWave:     {data_dir}/volunteer_X/class_XX/sample_*.npy
        Ultrasound: {data_dir}/volunteer_X/{class}/sample_*.npy

    Returns sorted list of (path, class_name, sample_idx).
    """
    samples = []

    ch_dir = resolve_source_dir(Path(data_dir), source, source_prefix, signal_type)
    if not ch_dir.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {ch_dir}\n"
            f"  Make sure Stage 1 has been run and --source_prefix is correct. "
            f"and --data_dir points to the correct location."
        )

    for class_dir in sorted(ch_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        cn = class_dir.name
        if class_name and cn != class_name:
            continue
        for fpath in sorted(class_dir.glob("sample_*.npy")):
            match = re.match(r"sample_(\d+)\.npy", fpath.name)
            if match:
                samples.append((fpath, cn, int(match.group(1))))

    return sorted(samples, key=lambda x: (x[1], x[2]))


# ---------------------------------------------------------------------------
#  Offset lookup  -- matches Stage 2 (extract_offset.py) output structure
# ---------------------------------------------------------------------------

def find_offsets_for_sample(
    offset_dir: str,
    source,
    class_name: str,
    sample_idx: int,
    source_prefix: str = "",
    signal_type: str = "wifi",
) -> List[Tuple[Path, Path]]:
    """Find all offset pairs for a sample  -- matches extract_offset.py output.

    Expected structure (from Stage 2 extract_offset.py):
        WiFi:       {offset_dir}/{static,dynamic}/volunteer_X/class_XX/pair_*.npy
        mmWave:     {offset_dir}/{static,dynamic}/volunteer_X/class_XX/pair_*.npy
        Ultrasound: {offset_dir}/{static,dynamic}/volunteer_X/{class}/pair_*.npy

    Returns list of (static_path, dynamic_path) for pairs involving sample_idx.
    """
    static_ch_dir = resolve_source_dir(Path(offset_dir) / "static", source, source_prefix, signal_type) / class_name
    dynamic_ch_dir = resolve_source_dir(Path(offset_dir) / "dynamic", source, source_prefix, signal_type) / class_name

    offsets = []
    if not static_ch_dir.is_dir():
        return offsets

    for fname in sorted(static_ch_dir.glob("pair_*.npy")):
        match = re.match(r"pair_(\d+)_(\d+)\.npy", fname.name)
        if not match:
            continue
        i, j = int(match.group(1)), int(match.group(2))
        if sample_idx in (i, j):
            dpath = dynamic_ch_dir / fname.name
            if dpath.is_file():
                offsets.append((static_ch_dir / fname.name, dpath))

    return offsets


# ---------------------------------------------------------------------------
#  Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_batch(
    diffusion: GaussianDiffusion,
    cond: torch.Tensor,
    progress: bool = False,
    cfg_scale: float = 1.0,
    anchor: torch.Tensor = None,
    sdedit_t: int = None,
) -> torch.Tensor:
    """Generate spectrograms conditioned on offsets, optionally with SDEdit anchor.

    Args:
        cond: (B, 2, H, W) -- [static_offset, dynamic_offset]
        progress: whether to show tqdm progress bar
        cfg_scale: classifier-free guidance scale (1.0 = no guidance)
        anchor: (B, 1, H, W) normalized anchor for SDEdit (None = pure noise start)
        sdedit_t: SDEdit starting timestep (None = pure noise start)

    Returns:
        (B, 1, H, W) -- generated spectrograms
    """
    if anchor is not None and sdedit_t is not None and sdedit_t > 0:
        samples = diffusion.p_sample_loop_from(
            cond, anchor, sdedit_t, device=cond.device,
            progress=progress, cfg_scale=cfg_scale,
        )
    else:
        shape = (1, cond.shape[2], cond.shape[3])
        device = cond.device
        samples = diffusion.p_sample_loop(cond, shape, device=device,
                                          progress=progress, cfg_scale=cfg_scale)

    return samples


def apply_offset_mode(
    static: torch.Tensor,
    dynamic: torch.Tensor,
    offset_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the same static/dynamic ablation mask used during training."""
    if offset_mode == "static":
        dynamic = torch.zeros_like(dynamic)
    elif offset_mode in ("dynamic_condition", "dynamic"):
        static = torch.zeros_like(static)
    elif offset_mode != "both":
        raise ValueError(f"Invalid offset_mode={offset_mode!r}")
    return static, dynamic


# ---------------------------------------------------------------------------
#  Path resolution helper
# ---------------------------------------------------------------------------

def _resolve_path(path: str, label: str = "path") -> str:
    """Resolve a path while preserving user-facing relative paths when possible."""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return path
    # Try relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_resolved = os.path.join(script_dir, path)
    if os.path.exists(script_resolved):
        return script_resolved
    return path


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    #  Load checkpoint
    # ------------------------------------------------------------------
    print(f"\n[Checkpoint] Loading from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config")
    if config is None:
        raise RuntimeError("Checkpoint missing 'config' field.")

    state_dict = ckpt.get("ema", ckpt.get("model"))
    stats = ckpt.get("dataset_stats", {})
    # Use checkpoint defaults for data paths unless overridden. If a checkpoint
    # carries stale local paths, fall back to the current modality defaults.
    tc = config.training
    cfg_offset_mode = args.offset_mode or getattr(tc, "offset_mode", "both")
    tc.offset_mode = cfg_offset_mode
    config.unet.use_dynamic_timestep = cfg_offset_mode in ("both", "dynamic")
    cfg_src_prefix = getattr(tc, "source_prefix", "") if hasattr(tc, "source_prefix") else ""
    cfg_signal_type = getattr(tc, "signal_type", "wifi") if hasattr(tc, "signal_type") else "wifi"
    if args.signal_type is not None:
        cfg_signal_type = args.signal_type
        cfg_src_prefix = ""
    source_default = MODALITY_SOURCE_DEFAULTS.get(cfg_signal_type, "volunteer_0")
    data_dir = args.data_dir or getattr(tc, "data_dir", MODALITY_DATA_DEFAULTS.get(cfg_signal_type))
    if args.data_dir is None and data_dir and not os.path.exists(data_dir):
        data_dir = MODALITY_DATA_DEFAULTS.get(cfg_signal_type, data_dir)
    if args.source is not None:
        source = args.source
    elif args.channel is not None:
        source = normalize_wifi_volunteer(args.channel) if cfg_signal_type == "wifi" else args.channel
    elif "sources" in getattr(tc, "__dict__", {}) and tc.sources:
        source = tc.sources[0]
    elif cfg_signal_type == "wifi":
        legacy_channels = getattr(tc, "channels", None) or ["0"]
        source = normalize_wifi_volunteer(legacy_channels[0])
    else:
        source = getattr(tc, "volunteer", source_default)
    if cfg_signal_type == "wifi":
        source = normalize_wifi_volunteer(source)
        cfg_src_prefix = ""
    if args.sample_batch_size is None:
        args.sample_batch_size = MODALITY_SAMPLE_BATCH_DEFAULTS.get(cfg_signal_type, 20)
    if args.sdedit_t is not None:
        if args.sdedit_t < 0 or args.sdedit_t > config.diffusion.num_timesteps:
            raise ValueError(
                f"--sdedit_t must be in [0, {config.diffusion.num_timesteps}], got {args.sdedit_t}"
            )

    # Normalization stats
    data_mean = args.data_mean if args.data_mean is not None else stats.get("mean", 0.0)
    data_std = args.data_std if args.data_std is not None else stats.get("std", 1.0)
    assert data_std > 1e-8, f"Invalid data_std={data_std}"

    # ------------------------------------------------------------------
    #  Build model
    # ------------------------------------------------------------------
    # Detect sample shape from data directory
    sample_shape = None
    if data_dir and os.path.isdir(data_dir):
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith('.npy'):
                    sample_shape = np.load(os.path.join(root, f)).shape
                    break
            if sample_shape:
                break

    if sample_shape is None:
        # Fallback: try config
        sample_shape = getattr(config.unet, "input_size", None)
    if sample_shape is None:
        # Last resort defaults per signal type
        _SHAPE_DEFAULTS = {"wifi": (64, 256), "mmwave": (16, 128), "ultrasound": (256, 256)}
        sample_shape = _SHAPE_DEFAULTS.get(cfg_signal_type, (64, 256))
        print(f"[Shape] Using default shape for {cfg_signal_type}: {sample_shape}")

    config.unet.in_channels = 3  # noisy sample + static offset + dynamic offset
    config.unet.out_channels = 1

    ca = getattr(config.unet, "use_cross_attn", False)
    dyn_t = getattr(config.unet, "use_dynamic_timestep", True)
    gate_state = "ENABLED" if config.unet.use_offset_gate else "DISABLED"
    gate_detail = f" (hidden_ch={config.unet.offset_gate_channels})" if config.unet.use_offset_gate else ""
    print(f'[Residual Gate] {gate_state}{gate_detail}  '
          f'[Cross-Attn] {"ON" if ca else "OFF"}  '
          f'[Dynamic->Timestep] {"ON" if dyn_t else "OFF"}')
    print(f"[OffsetMode] {cfg_offset_mode}")

    unet = UNetModel(config.unet, input_size=sample_shape[0])
    unet.load_state_dict(state_dict)
    print(f"[Model] U-Net loaded. Params: {sum(p.numel() for p in unet.parameters()):,}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    diffusion = GaussianDiffusion(unet, config.diffusion).to(device)
    diffusion.eval()
    print(f"[Device] {device}")

    # ------------------------------------------------------------------
    #  Path resolution
    # ------------------------------------------------------------------
    data_dir = _resolve_path(data_dir, "data_dir")

    # Base offset pool dir (for cross-class / unseen generation)
    # Auto-detect: use checkpoint offset_dir if available, else modality default
    _OFFSET_DEFAULTS = {"wifi": "offsets_WiFi", "mmwave": "offsets_mmWave", "ultrasound": "offsets_ultrasound"}
    base_offset_dir = args.base_offset_dir
    if base_offset_dir is None:
        ckpt_offset_dir = tc.offset_dir if hasattr(tc, "offset_dir") else None
        if ckpt_offset_dir and ckpt_offset_dir != "offsets" and os.path.exists(ckpt_offset_dir):
            base_offset_dir = ckpt_offset_dir
        else:
            base_offset_dir = _OFFSET_DEFAULTS.get(cfg_signal_type, "offsets_mmWave")
    if args.base_offset_dir:
        base_offset_dir = _resolve_path(base_offset_dir, "base_offset_dir")
    base_label = "user-specified" if args.base_offset_dir else "auto-detected"
    print(f"[Config] base_offset_dir = {base_offset_dir}  ({base_label})")

    _OUTPUT_DEFAULTS = {"wifi": "generated_WiFi", "mmwave": "generated_mmWave", "ultrasound": "generated_ultrasound"}
    if args.output_dir is None:
        args.output_dir = _OUTPUT_DEFAULTS.get(cfg_signal_type, "generated")

    offset_static_scale = data_std
    offset_dynamic_scale = data_std
    print(f"[OffsetNorm] Using unified data_std normalization for offsets: scale={data_std:.6f}")

    # ------------------------------------------------------------------
    #  Device / performance warnings
    # ------------------------------------------------------------------
    if args.sdedit_t is not None and args.sdedit_t > 0:
        effective_sampling_steps = args.sdedit_t
    else:
        effective_sampling_steps = config.diffusion.num_timesteps

    if device.type == "cpu":
        step_word = "step" if effective_sampling_steps == 1 else "steps"
        print(f"\n  [WARN] DDPM {effective_sampling_steps} {step_word} on CPU will be slow.")
        print(f"         Consider running on a machine with CUDA GPU.")
        print(f"         Runtime depends on sample shape and batch size ({args.sample_batch_size}).")

    # ------------------------------------------------------------------
    #  Parse class list (support comma-separated for multi-class)
    # ------------------------------------------------------------------
    if args.class_name:
        class_list = [c.strip() for c in args.class_name.split(",") if c.strip()]
    else:
        # Auto-discover all classes in data directory
        _cls_prefix = "" if cfg_signal_type == "ultrasound" else "class_"
        discovered = _discover_classes(data_dir, source, cfg_src_prefix, class_prefix=_cls_prefix, signal_type=cfg_signal_type)
        if discovered:
            if args.num_classes is not None:
                discovered = discovered[:args.num_classes]
                print(f"[Auto] Using first {len(discovered)} classes: {discovered}")
            else:
                print(f"[Auto] Discovered {len(discovered)} classes: {discovered}")
            class_list = discovered
        else:
            print(f"[ERROR] No classes found for source {source} under {data_dir}")
            sys.exit(1)

    # ------------------------------------------------------------------
    #  Per-class generation
    # ------------------------------------------------------------------
    rng = np.random.RandomState(args.seed)
    grand_total = 0
    class_summaries = []

    # Cross-class offset pool (built lazily when first needed)
    _offset_pools = {}

    def _get_offset_pool(target_class=None):
        key = target_class or "__all__"
        if key not in _offset_pools:
            classes = [target_class] if target_class else None
            label = target_class or "ALL base classes"
            print(f"\n[OffsetPool] Building offset pool for {label} from: {base_offset_dir}")
            _offset_pools[key] = OffsetPoolSampler(
                offset_dir=base_offset_dir,
                sources=[source],
                classes=classes,
                sample_shape=sample_shape,
                signal_type=cfg_signal_type,
                source_prefix=cfg_src_prefix,
            )
        return _offset_pools[key]

    for class_idx, target_class in enumerate(class_list):
        class_label = target_class or "all_classes"
        print(f"\n{'#'*55}")
        print(f"#  CLASS {class_idx+1}/{len(class_list)}: {class_label}")
        print(f"{'#'*55}")

        # ------------------------------------------------------------------
        #  Scan real samples for this class
        # ------------------------------------------------------------------
        all_samples = scan_samples(data_dir, source, target_class, cfg_src_prefix, cfg_signal_type)
        if not all_samples:
            print(f"  [SKIP] No sample_*.npy files found for class={target_class}")
            continue

        n_total = len(all_samples)
        print(f"\n[Data] {data_dir}/{source}: {n_total} real samples ({class_label})")

        # ------------------------------------------------------------------
        #  Shot / Eval split
        # ------------------------------------------------------------------
        eval_indices_from_file = None
        if args.shot_indices:
            shot_idxs = [int(x.strip()) for x in args.shot_indices.split(",")]
            shots = []
            for si in shot_idxs:
                found = [s for s in all_samples if s[2] == si]
                if found:
                    shots.append(found[0])
            shot_set = set(s[2] for s in shots)
            split_mode = "manual"
        else:
            # Prefer the split generated by extract_offset.py few-shot mode.
            split_from_file = None
            split_json = None
            if args.split_info_dir:
                canonical_split = Path(args.split_info_dir) / str(source) / target_class / "split_info.json"
                legacy_split = Path(args.split_info_dir) / target_class / "split_info.json"
                split_json = canonical_split if canonical_split.is_file() else legacy_split
            else:
                default_split_dir = Path(f"split_info_{cfg_signal_type}")
                canonical_split = default_split_dir / str(source) / target_class / "split_info.json"
                legacy_split = default_split_dir / target_class / "split_info.json"
                split_json = canonical_split if canonical_split.is_file() else legacy_split

            if split_json and split_json.is_file():
                with open(split_json, "r", encoding="utf-8") as f:
                    split_from_file = json.load(f)

            if split_from_file is not None:
                shot_idxs = split_from_file["shot_indices"]
                eval_indices_from_file = split_from_file.get("eval_indices")
                shots = []
                for si in shot_idxs:
                    found = [s for s in all_samples if s[2] == si]
                    if found:
                        shots.append(found[0])
                shot_set = set(s[2] for s in shots)
                split_mode = f"pre-defined ({split_json.parent})"
            elif n_total <= args.num_shots:
                shots = list(all_samples)
                shot_set = set(s[2] for s in shots)
                split_mode = "all samples"
            else:
                indices = list(range(n_total))
                chosen = rng.choice(indices, args.num_shots, replace=False)
                shots = [all_samples[i] for i in sorted(chosen)]
                shot_set = set(s[2] for s in shots)
                split_mode = f"pseudo-random (seed={args.seed})"

        if eval_indices_from_file:
            eval_set = set(eval_indices_from_file)
            eval_samples = [s for s in all_samples if s[2] in eval_set and s[2] not in shot_set]
        else:
            eval_samples = [s for s in all_samples if s[2] not in shot_set]
        n_eval = len(eval_samples)

        # Print split
        print(f"\n{'='*55}")
        print(f"  SHOT / EVAL SPLIT  |  {split_mode}")
        print(f"{'='*55}")
        print(f"  Source     : {source}")
        print(f"  Class      : {class_label}")
        print(f"  Total      : {n_total} real samples")
        print(f"  Shots      : {len(shots)}  ->  {len(shots) * args.samples_per_shot} generated")
        print(f"  Eval       : {n_eval}  (for downstream evaluation)")
        print(f"{'='*55}")
        print(f"\n  Shot samples:")
        for spath, cn, si in shots:
            print(f"    [SHOT] sample_{si:02d}  ({cn})")
        if n_eval > 0:
            print(f"\n  Eval samples ({n_eval} total, withheld):")
            for epath, cn, si in eval_samples[:min(10, n_eval)]:
                print(f"    [EVAL] sample_{si:02d}  ({cn})")
            if n_eval > 10:
                print(f"    ... and {n_eval - 10} more")

        # ------------------------------------------------------------------
        #  Prepare output directory
        # ------------------------------------------------------------------
        class_out_dir = os.path.join(args.output_dir, class_label)
        os.makedirs(class_out_dir, exist_ok=True)

        # Save split info
        split_info = {
            "class": class_label,
            "total_real": n_total,
            "num_shots": len(shots),
            "num_eval": n_eval,
            "shot_indices": [s[2] for s in shots],
            "eval_indices": [s[2] for s in eval_samples],
            "seed": args.seed,
            "split_mode": split_mode,
            "samples_per_shot": args.samples_per_shot,
            "data_mean": data_mean,
            "data_std": data_std,
            "offset_static_scale": offset_static_scale,
            "offset_dynamic_scale": offset_dynamic_scale,
            "offset_mode": cfg_offset_mode,
        }
        split_path = os.path.join(class_out_dir, "split_info.json")
        with open(split_path, "w") as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)

        # ------------------------------------------------------------------
        #  Generate
        # ------------------------------------------------------------------
        method_label = "DDPM" + (" + SDEdit t=" + str(args.sdedit_t) if args.sdedit_t else "")
        step_word = "step" if effective_sampling_steps == 1 else "steps"
        print(f"\n[Generating] {args.samples_per_shot} virtual samples per shot "
              f"({method_label}, {effective_sampling_steps} {step_word})")

        total_generated = 0

        for shot_idx, (sample_path, cls_name, sample_idx) in enumerate(shots):
            # When --per_class_offsets is set, use class-specific pool (Siamese-filtered)
            # Otherwise use all base-class offsets for cross-class transfer
            if args.per_class_offsets and target_class:
                pool = _get_offset_pool(target_class)
            else:
                pool = _get_offset_pool(None)
            n_pool = len(pool._offsets)
            print(f"\n  Shot {shot_idx+1}/{len(shots)}: sample_{sample_idx:02d} "
                  f"(class={cls_name}, {n_pool} pool offsets)")

            # Load anchor sample for SDEdit
            sdedit_t = getattr(args, "sdedit_t", None)
            if sdedit_t is not None and sdedit_t > 0:
                anchor_raw = np.load(str(sample_path)).astype(np.float32)
                anchor_t = torch.from_numpy(
                    (anchor_raw - data_mean) / (data_std + 1e-8)
                ).unsqueeze(0).unsqueeze(0).to(device)
            else:
                anchor_t = None

            n_per_batch = min(max(1, args.sample_batch_size), args.samples_per_shot)
            num_batches = (args.samples_per_shot + n_per_batch - 1) // n_per_batch

            gen_count = 0
            for bi in range(num_batches):
                curr_n = min(n_per_batch, args.samples_per_shot - gen_count)

                # Pick a fresh random offset for EACH sample from base pool
                s_list, d_list = [], []
                for _ in range(curr_n):
                    st, dt = pool.sample(n=1, source=source)
                    s_list.append(st.squeeze(0).squeeze(0).numpy())
                    d_list.append(dt.squeeze(0).squeeze(0).numpy())

                static_b = np.stack(s_list, axis=0)  # (B, H, W)
                dynamic_b = np.stack(d_list, axis=0)
                static_t = (torch.from_numpy(static_b) / (offset_static_scale + 1e-8)).unsqueeze(1)
                dynamic_t = (torch.from_numpy(dynamic_b) / (offset_dynamic_scale + 1e-8)).unsqueeze(1)
                offset_scale = getattr(args, "offset_scale", 1.0)
                static_t = static_t * offset_scale
                dynamic_t = dynamic_t * offset_scale
                static_t, dynamic_t = apply_offset_mode(static_t, dynamic_t, cfg_offset_mode)
                cond_batch = torch.cat([static_t, dynamic_t], dim=1).to(device)  # (B, 2, H, W)

                gen = generate_batch(
                    diffusion, cond_batch,
                    progress=args.progress,
                    cfg_scale=args.cfg_scale,
                    anchor=anchor_t.expand(curr_n, -1, -1, -1) if anchor_t is not None else None,
                    sdedit_t=sdedit_t,
                )

                if args.denormalize:
                    gen = gen * data_std + data_mean

                gen_np = gen.cpu().numpy()
                for i in range(curr_n):
                    fname = f"shot{shot_idx+1:02d}_s{sample_idx:02d}_g{total_generated:04d}.npy"
                    fpath = os.path.join(class_out_dir, fname)
                    np.save(fpath, gen_np[i, 0].astype(np.float32))
                    total_generated += 1
                    gen_count += 1

            print(f"    -> {gen_count} virtual samples generated")

        # Class summary
        print(f"\n  [Class {class_label}] {total_generated} samples -> {class_out_dir}")
        class_summaries.append((class_label, total_generated, n_eval))
        grand_total += total_generated

    # ------------------------------------------------------------------
    #  Grand Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  GENERATION COMPLETE")
    print(f"{'='*55}")
    for cls_label, cls_gen, cls_eval in class_summaries:
        print(f"  {cls_label:20s}: {cls_gen:4d} generated, {cls_eval:4d} eval remaining")
    print(f"  {'-'*40}")
    print(f"  {'TOTAL':20s}: {grand_total:4d} generated")
    print(f"  Output dir    : {args.output_dir}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
