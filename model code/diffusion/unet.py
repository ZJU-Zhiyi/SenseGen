'''
U-Net backbone for SenseGen diffusion model with Offset-Gated ResBlock (G1).

G1: Inject offset conditioning at every U-Net level via a lightweight
Conv->SiLU->Conv->Sigmoid gate that modulates feature activations after
each residual block's out_layers.

Designed for offset-conditioned sensing tensors.
Input shape auto-detected.
'''

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import UNetConfig
except ImportError:
    from config import UNetConfig


# ---------------------------------------------------------------------------
#  Timestep Embedding
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    '''Sinusoidal positional embedding for diffusion timesteps.'''

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(t.device)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


# ---------------------------------------------------------------------------
#  Normalization helper
# ---------------------------------------------------------------------------

def normalization(channels: int, num_groups: int = 32) -> nn.GroupNorm:
    '''GroupNorm with safe group count.'''
    groups = min(num_groups, channels)
    if channels % groups != 0:
        for g in range(groups, 0, -1):
            if channels % g == 0:
                groups = g
                break
    return nn.GroupNorm(num_groups=groups, num_channels=channels)


# ---------------------------------------------------------------------------
#  G1: Offset Gate
# ---------------------------------------------------------------------------

class OffsetGate(nn.Module):
    """Gated FiLM: offset controls both injection strength and feature modulation.

    Combines residual gating with FiLM modulation:
        feat = SharedTrunk(offset)
        gate = GateHead(feat)          -- spatial attention (B, 1, H, W)
        gamma, beta = FiLMHead(feat)   -- per-channel scale + shift (B, C, H, W)

        out = skip(x) + gate * (gamma * h + beta)
              ^^^^^^^^   ^^^^   ^^^^^^^^^^^^^^^^^
              identity   where   what to inject
              preserved  to inject

    gate controls spatial injection strength (which regions get modulated),
    FiLM controls fine-grained channel modulation (gamma: scale, beta: shift).
    """

    def __init__(self, offset_channels: int = 2, hidden_channels: int = 16,
                 out_channels: int = 48):
        super().__init__()
        # Shared feature trunk
        self.shared = nn.Sequential(
            nn.Conv2d(offset_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        # Spatial gate: (B, 1, H, W) in (0, 1)
        self.gate_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        # FiLM head: (B, 2*C, H, W) -> gamma, beta
        self.film_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels * 2, kernel_size=3, padding=1),
        )

    def forward(self, offset: torch.Tensor, h: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.shared(offset)

        # Spatial gate: where to inject
        gate = self.gate_head(feat)          # (B, 1, H, W) in (0, 1)

        # FiLM: what to inject
        film = self.film_head(feat)          # (B, 2*C, H, W)
        gamma, beta = film.chunk(2, dim=1)
        gamma = torch.tanh(gamma) * 0.5 + 1.0   # range [0.5, 1.5], centered at 1.0

        h_mod = gamma * h + beta
        return gate, h_mod

# ---------------------------------------------------------------------------
#  Residual Block with optional Offset-Gating
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    '''Residual block with time embedding, channel change support,
    and optional offset-gating (G1).

    Args:
        in_channels:  input feature channels
        out_channels: output feature channels
        emb_channels: time embedding dimension
        use_offset_gate: if True, add OffsetGate that modulates out_layers output
        offset_gate_channels: hidden channels in OffsetGate
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        dropout: float = 0.0,
        num_groups: int = 32,
        use_scale_shift_norm: bool = False,
        use_offset_gate: bool = False,
        offset_gate_channels: int = 16,
    ):
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(in_channels, num_groups),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        emb_mult = 2 if use_scale_shift_norm else 1
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, out_channels * emb_mult),
        )

        self.out_layers = nn.Sequential(
            normalization(out_channels, num_groups),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.skip_connection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        # Residual gating: gate on the transformed branch only
        if use_offset_gate:
            self.offset_gate = OffsetGate(
                offset_channels=2,
                hidden_channels=offset_gate_channels,
                out_channels=out_channels,
            )
        else:
            self.offset_gate = None

    def forward(self, x: torch.Tensor, emb: torch.Tensor,
                offset: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''Forward pass.

        Args:
            x:      (B, in_channels, H, W) feature map
            emb:    (B, emb_channels) time embedding
            offset: (B, 2, H, W) pooled offset conditioning, or None
        '''
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)[:, :, None, None]

        if self.use_scale_shift_norm:
            scale, shift = emb_out.chunk(2, dim=1)
            h = h * (1 + scale) + shift
        else:
            h = h + emb_out

        h = self.out_layers(h)

        # Gated FiLM: gate controls injection strength, FiLM modulates features
        if self.offset_gate is not None and offset is not None:
            gate, h = self.offset_gate(offset, h)
            h = gate * h  # spatial gate x FiLM-modulated features

        return self.skip_connection(x) + h


# ---------------------------------------------------------------------------
#  Attention Block
# ---------------------------------------------------------------------------

class AttentionBlock(nn.Module):
    '''Multi-head self-attention with residual connection.'''

    def __init__(self, channels: int, num_heads: int = 4, num_head_channels: int = -1):
        super().__init__()
        self.channels = channels

        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            self.num_heads = channels // num_head_channels
        assert channels % self.num_heads == 0, (
            f'channels {channels} not divisible by num_heads {self.num_heads}'
        )

        self.norm = normalization(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.reshape(B * self.num_heads, C // self.num_heads * 3, H, W) \
                       .split(C // self.num_heads, dim=1)
        scale = (C // self.num_heads) ** -0.5
        q = q.reshape(B * self.num_heads, C // self.num_heads, H * W).transpose(1, 2)
        k = k.reshape(B * self.num_heads, C // self.num_heads, H * W)
        v = v.reshape(B * self.num_heads, C // self.num_heads, H * W).transpose(1, 2)
        attn = torch.bmm(q, k) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return x + self.proj(out)


# ---------------------------------------------------------------------------
#  Downsample / Upsample
# ---------------------------------------------------------------------------


class OffsetCrossAttn(nn.Module):
    """Cross-attention: image features query offset features.

    Image features at each spatial position attend to all offset positions,
    learning which offset patterns are relevant for which image regions.

    out = h + attention(Q=image, K=offset, V=offset)
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0

        self.norm_img = normalization(channels)
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.kv_proj = nn.Conv2d(2, channels * 2, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, h: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        B, C, H, W = h.shape
        q = self.q_proj(self.norm_img(h))
        q = q.reshape(B * self.num_heads, self.head_dim, H * W).transpose(1, 2)

        kv = self.kv_proj(offset)
        k, v = kv.chunk(2, dim=1)
        k = k.reshape(B * self.num_heads, self.head_dim, H * W)
        v = v.reshape(B * self.num_heads, self.head_dim, H * W).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = torch.bmm(q, k) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return h + self.out_proj(out)

class Downsample(nn.Module):
    '''Strided convolution for 2x spatial downsampling.'''

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    '''Nearest-neighbour upsampling followed by convolution.'''

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# ---------------------------------------------------------------------------
#  U-Net Model with G1 Offset Routing
# ---------------------------------------------------------------------------

class UNetModel(nn.Module):
    '''Conditional U-Net for diffusion with G1 offset-gating support.

    Offsets are extracted from the input tensor (channels 1 and 2), pooled
    to match each U-Net resolution level, and passed to every ResBlock's
    OffsetGate for spatial-channel feature modulation.
    '''

    def __init__(
        self,
        config: UNetConfig,
        input_size: int = 64,
    ):
        super().__init__()

        # Pull config values
        in_channels = config.in_channels
        out_channels = config.out_channels
        model_channels = config.model_channels
        channel_mult = config.channel_mult
        num_res_blocks = config.num_res_blocks
        attention_resolutions = config.attention_resolutions
        dropout = config.dropout
        time_emb_dim = config.time_emb_dim or model_channels * 4
        num_groups = config.num_groups
        num_heads = config.num_heads
        num_head_channels = config.num_head_channels
        resblock_updown = config.resblock_updown
        use_scale_shift_norm = config.use_scale_shift_norm
        use_offset_gate = config.use_offset_gate
        offset_gate_channels = config.offset_gate_channels
        use_cross_attn = getattr(config, 'use_cross_attn', False)
        use_dynamic_timestep = getattr(config, 'use_dynamic_timestep', True)

        self.use_offset_gate = use_offset_gate
        self.use_dynamic_timestep = use_dynamic_timestep
        self.input_size = input_size

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # G3: dynamic offset -> time embedding modulation
        if use_offset_gate and use_dynamic_timestep:
            self.dyn_proj = nn.Sequential(
                nn.Linear(1, time_emb_dim // 4),
                nn.SiLU(),
                nn.Linear(time_emb_dim // 4, time_emb_dim),
            )

        # Input projection
        self.input_proj = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        # --- Build downsampling path ---
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        ds = 1
        skips = [ch]

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                layers: List[nn.Module] = [
                    ResBlock(ch, out_ch, time_emb_dim, dropout, num_groups,
                             use_scale_shift_norm,
                             use_offset_gate=use_offset_gate,
                             offset_gate_channels=offset_gate_channels)
                ]
                ch = out_ch
                resolution = self.input_size // ds
                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads, num_head_channels))
                self.down_blocks.append(nn.ModuleList(layers))
                skips.append(ch)

            if level != len(channel_mult) - 1:
                self.down_blocks.append(
                    nn.ModuleList([
                        ResBlock(ch, ch, time_emb_dim, dropout, num_groups,
                                 use_scale_shift_norm,
                                 use_offset_gate=use_offset_gate,
                                 offset_gate_channels=offset_gate_channels)
                        if resblock_updown else Downsample(ch)
                    ])
                )
                skips.append(ch)
                ds *= 2

        # --- Bottleneck ---
        resolution = self.input_size // ds
        mid_layers: List[nn.Module] = [
            ResBlock(ch, ch, time_emb_dim, dropout, num_groups,
                     use_scale_shift_norm,
                     use_offset_gate=use_offset_gate,
                     offset_gate_channels=offset_gate_channels),
        ]
        if resolution in attention_resolutions:
            mid_layers.append(AttentionBlock(ch, num_heads, num_head_channels))
        mid_layers.append(
            ResBlock(ch, ch, time_emb_dim, dropout, num_groups,
                     use_scale_shift_norm,
                     use_offset_gate=use_offset_gate,
                     offset_gate_channels=offset_gate_channels)
        )
        self.mid_block = nn.ModuleList(mid_layers)

        # Cross-attention at bottleneck (lowest resolution, most semantic)
        if use_cross_attn:
            self.cross_attn_mid = nn.ModuleList([
                OffsetCrossAttn(ch, num_heads=num_heads)
                for _ in range(2)
            ])
        else:
            self.cross_attn_mid = nn.ModuleList()

        # --- Build upsampling path ---
        self.up_blocks = nn.ModuleList()
        for level, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult
            for i in range(num_res_blocks + 1):
                skip_ch = skips.pop()
                in_ch = ch + skip_ch
                layers = [
                    ResBlock(in_ch, out_ch, time_emb_dim, dropout, num_groups,
                             use_scale_shift_norm,
                             use_offset_gate=use_offset_gate,
                             offset_gate_channels=offset_gate_channels)
                ]
                ch = out_ch
                resolution = self.input_size // ds
                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads, num_head_channels))
                self.up_blocks.append(nn.ModuleList(layers))

            if level != len(channel_mult) - 1:
                self.up_blocks.append(
                    nn.ModuleList([
                        ResBlock(ch, ch, time_emb_dim, dropout, num_groups,
                                 use_scale_shift_norm,
                                 use_offset_gate=use_offset_gate,
                                 offset_gate_channels=offset_gate_channels)
                        if resblock_updown else Upsample(ch)
                    ])
                )
                ds //= 2

        assert len(skips) == 0, f'Skip connection mismatch: {len(skips)} remaining'

        # Output projection
        self.out = nn.Sequential(
            normalization(ch, num_groups),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        '''Forward pass with G1 offset routing.

        Input:
            x: (B, 3, H, W) -- concatenated [noisy_sample, static_offset, dynamic_offset]
            t: (B,) -- diffusion timestep indices

        When use_offset_gate=True, channels [1:3] are extracted as offset,
        pooled through the U-Net hierarchy, and fed to every ResBlock's offset_gate.

        Returns:
            (B, 1, H, W) -- predicted noise
        '''
        emb = self.time_embed(t)

        # --- Extract and route offsets (G1) ---
        if self.use_offset_gate:
            # Input layout: channel 0 = noisy sample, channels 1-2 = offsets
            offset_current = x[:, 1:, :, :]          # (B, 2, H, W)
            offset_stack = [offset_current]           # stack for up-path retrieval

            # G3: dynamic offset pooled -> modulates time embedding
            if self.use_dynamic_timestep:
                dyn = x[:, 2:3, :, :]                     # (B, 1, H, W)
                dyn = F.adaptive_avg_pool2d(dyn, 1).flatten(1)  # (B, 1)
                emb = emb + self.dyn_proj(dyn)            # (B, time_emb_dim)
        else:
            offset_current = None
            offset_stack = []

        # Input projection
        h = self.input_proj(x)
        hs = [h]

        # --- Down path ---
        for block in self.down_blocks:
            for layer in block:
                if isinstance(layer, ResBlock):
                    h = layer(h, emb, offset=offset_current)
                elif isinstance(layer, AttentionBlock):
                    h = layer(h)
                else:  # Downsample
                    h = layer(h)
                    if self.use_offset_gate and offset_current is not None:
                        offset_current = F.avg_pool2d(offset_current, 2)
                        offset_stack.append(offset_current)
            hs.append(h)

        # --- Bottleneck ---
        for layer in self.mid_block:
            if isinstance(layer, ResBlock):
                h = layer(h, emb, offset=offset_current)
            elif isinstance(layer, AttentionBlock):
                h = layer(h)

        # Cross-attention: image features query offset features
        if self.use_offset_gate and offset_current is not None:
            for ca_layer in self.cross_attn_mid:
                h = ca_layer(h, offset_current)

        # --- Up path ---
        for block in self.up_blocks:
            for layer in block:
                if isinstance(layer, ResBlock):
                    skip = hs.pop()
                    h = torch.cat([h, skip], dim=1)
                    h = layer(h, emb, offset=offset_current)
                elif isinstance(layer, AttentionBlock):
                    h = layer(h)
                else:  # Upsample
                    h = layer(h)
                    if self.use_offset_gate and offset_stack:
                        offset_stack.pop()
                        offset_current = offset_stack[-1] if offset_stack else None

        assert len(hs) == 0

        return self.out(h)
