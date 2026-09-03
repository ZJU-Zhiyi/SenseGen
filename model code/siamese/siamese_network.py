#!/usr/bin/env python3
"""
Siamese Network for Offset Similarity Filtering — SenseGen Stage 2.5
=====================================================================
A Siamese CNN that learns to judge whether two offset pairs belong to the
same pose class (similar) or different pose classes (dissimilar).

Input:  two offset tensors, each of shape (2, H, W) — [static, dynamic]
        Works with WiFi (64,256), mmWave (16,128), and Ultrasound (256,256).
Output: similarity score in [0, 1]

Architecture:
  Shared CNN encoder (Siamese twin) → concatenate |emb1 - emb2|, emb1, emb2
  → FC projection head → sigmoid similarity score

Usage:
  from siamese_network import OffsetSiamese, OffsetEncoder
  model = OffsetSiamese()
  score = model(offset_a, offset_b)  # (B, 1)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OffsetEncoder(nn.Module):
    """Shared CNN encoder for offset tensors (2, H, W) → embedding vector.

    Architecture uses AdaptiveAvgPool2d at the bottleneck, making it compatible
    with any spatial input size (WiFi 64x256, mmWave 16x128, Ultrasound 256x256).
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 32,
        embedding_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(base_channels * 8, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode offset tensor to embedding.

        Args:
            x: (B, 2, H, W) offset tensor [static, dynamic].

        Returns:
            embedding: (B, embedding_dim).
        """
        x = self.conv1(x)        # (B, 32, H, W)
        x = self.conv2(x)        # (B, 64, H/2, W/2)
        x = self.conv3(x)        # (B, 128, H/4, W/4)
        x = self.conv4(x)        # (B, 256, H/4, W/4)
        x = self.dropout(x)
        x = self.pool(x)         # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)
        x = self.fc(x)           # (B, embedding_dim)
        return x


class OffsetSiamese(nn.Module):
    """Siamese network for offset similarity judgment.

    Two offset tensors pass through a shared encoder, then their embeddings
    are combined and projected to a similarity score.

    Shape-agnostic: compatible with WiFi (64,256), mmWave (16,128),
    and Ultrasound (256,256) offset tensors.

    Args:
        encoder: Shared OffsetEncoder (created internally if None).
        embedding_dim: Dimension of the encoder output embeddings.
        projection_dim: Hidden dimension for the projection MLP head.
        dropout: Dropout rate in the projection head.
    """

    def __init__(
        self,
        encoder = None,
        embedding_dim: int = 256,
        projection_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = encoder if encoder is not None else OffsetEncoder(embedding_dim=embedding_dim)

        # Projection head: (3 * embedding_dim) → projection_dim → 1
        # We use [emb1, emb2, |emb1 - emb2|] as the combined representation
        self.head = nn.Sequential(
            nn.Linear(embedding_dim * 3, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim // 2),
            nn.BatchNorm1d(projection_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(projection_dim // 2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single offset tensor (forward through encoder only)."""
        return self.encoder(x)

    def forward(
        self,
        offset_a: torch.Tensor,
        offset_b: torch.Tensor,
    ) -> torch.Tensor:
        """Compute similarity between two offset tensors.

        Args:
            offset_a: (B, 2, H, W) first offset batch.
            offset_b: (B, 2, H, W) second offset batch.

        Returns:
            similarity: (B, 1) in [0, 1]. 1 = same pose, 0 = different pose.
        """
        emb_a = self.encoder(offset_a)       # (B, E)
        emb_b = self.encoder(offset_b)       # (B, E)
        diff = torch.abs(emb_a - emb_b)       # (B, E)
        combined = torch.cat([emb_a, emb_b, diff], dim=1)  # (B, 3E)
        return self.head(combined)            # (B, 1)


# ---------------------------------------------------------------------------
#  Loss function
# ---------------------------------------------------------------------------

class ContrastiveLoss(nn.Module):
    """Contrastive loss for Siamese network training.

    Margin-based loss: pushes dissimilar pairs apart by at least `margin`.

    Args:
        margin: Minimum distance margin for dissimilar pairs.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        similarity: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            similarity: (B, 1) predicted similarity in [0, 1].
            label: (B, 1) ground truth: 1.0 = similar (same class), 0.0 = dissimilar.

        Returns:
            Scalar loss.
        """
        label = label.view_as(similarity)
        # Similar pairs: loss = (1 - similarity)^2
        loss_similar = label * (1.0 - similarity).pow(2)
        # Dissimilar pairs: loss = max(0, similarity - margin)^2
        margin_violation = torch.clamp(similarity - self.margin, min=0.0)
        loss_dissimilar = (1.0 - label) * margin_violation.pow(2)
        return (loss_similar + loss_dissimilar).mean()
