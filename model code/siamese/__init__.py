"""Siamese Network for Offset Similarity Filtering.

Provides:
  - OffsetEncoder: Shared CNN encoder for offset tensors (shape-agnostic).
  - OffsetSiamese: Siamese network for similarity scoring.
  - ContrastiveLoss: Margin-based contrastive loss.

Supports WiFi (64,256), mmWave (16,128), and Ultrasound (256,256) offsets.
"""

from .siamese_network import OffsetEncoder, OffsetSiamese, ContrastiveLoss

__all__ = ["OffsetEncoder", "OffsetSiamese", "ContrastiveLoss"]
