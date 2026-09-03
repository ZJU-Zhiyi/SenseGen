"""
Configuration for the default SenseGen diffusion model.
Defaults are conservative enough for common 24 GB VRAM GPUs.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class UNetConfig:
    """U-Net backbone configuration with offset-gating support."""

    in_channels: int = 3
    out_channels: int = 1

    model_channels: int = 48
    channel_mult: Tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = ()
    dropout: float = 0.1

    time_emb_dim: Optional[int] = None
    num_groups: int = 32
    num_heads: int = 4
    num_head_channels: int = -1

    resblock_updown: bool = False
    use_scale_shift_norm: bool = False

    # Final diffusion defaults: residual offset gating + bottleneck cross-attention.
    use_offset_gate: bool = True
    offset_gate_channels: int = 64
    use_cross_attn: bool = True
    use_dynamic_timestep: bool = True


@dataclass
class DiffusionConfig:
    """Diffusion process configuration."""

    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "cosine"

    clip_denoised: bool = True
    cfg_scale: float = 1.0


@dataclass
class TrainingConfig:
    """Training hyperparameters with modality-aware defaults."""

    signal_type: str = "wifi"
    data_dir: str = "WiFi data_pad/base"
    offset_dir: str = "offsets_WiFi"
    sources: List = field(default_factory=lambda: ["volunteer_0"])
    channels: Optional[List] = None  # deprecated checkpoint/CLI compatibility
    volunteer: str = "volunteer_0"
    source_prefix: str = ""
    classes: Optional[List[str]] = None

    batch_size: int = 40
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    ema_decay: float = 0.9999
    grad_clip: float = 1.0
    cfg_drop_prob: float = 0.1
    offset_mode: str = "both"
    offset_scale_min: float = 1.0
    offset_scale_max: float = 1.0
    max_samples_per_epoch: Optional[int] = None

    num_epochs: int = 500
    lr_warmup_steps: int = 1000

    log_interval: int = 100
    save_interval: int = 50
    save_best: bool = False
    sample_interval: int = 25
    num_samples: int = 8

    output_dir: str = "checkpoints_wifi"
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42


@dataclass
class SenseGenConfig:
    """Top-level configuration."""

    unet: UNetConfig = field(default_factory=UNetConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
