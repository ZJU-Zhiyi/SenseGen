"""SenseGen Diffusion Model with Offset-Gated ResBlock (G1).

G1: Offset-Gated ResBlock — injects offset conditioning at every U-Net level
via a lightweight Conv→SiLU→Conv→Sigmoid gate that modulates feature
activations after each residual block's out_layers.

Usage:
    from diffusion.config import SenseGenConfig, UNetConfig
    config = UNetConfig(use_offset_gate=True)
"""
