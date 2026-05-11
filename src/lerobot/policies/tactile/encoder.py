#!/usr/bin/env python
"""Shared tactile encoders for policies that consume 2D tactile maps."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class TactileCNN(nn.Module):
    """CNN backbone that maps a tactile frame to one feature vector."""

    def __init__(
        self,
        input_shape: tuple[int, int] = (12, 32),
        feature_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.feature_dim = feature_dim

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)

        feature_h = input_shape[0] // 8
        feature_w = input_shape[1] // 8
        if feature_h <= 0 or feature_w <= 0:
            raise ValueError(f"tactile input_shape is too small for CNN pooling: {input_shape}")

        conv_output_dim = 128 * feature_h * feature_w
        self.fc1 = nn.Linear(conv_output_dim, 512)
        self.fc2 = nn.Linear(512, feature_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class TactileAttentionCNN(nn.Module):
    """CNN tactile backbone with a simple spatial attention head."""

    def __init__(
        self,
        input_shape: tuple[int, int] = (12, 32),
        feature_dim: int = 256,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.feature_dim = feature_dim

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)

        self.attention = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, feature_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x * self.attention(x)
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        x = torch.cat([avg_pool, max_pool], dim=1)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class TactileTokenEncoder(nn.Module):
    """Encode tactile maps as one or more prefix tokens.

    Input shape is (B, H, W), output shape is (B, n_tokens, feature_dim).
    """

    def __init__(
        self,
        encoder_type: str,
        input_shape: tuple[int, int],
        feature_dim: int,
        n_tokens: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        if n_tokens < 1:
            raise ValueError(f"n_tokens must be >= 1, got {n_tokens}")

        self.n_tokens = n_tokens
        self.feature_dim = feature_dim

        if encoder_type == "cnn":
            self.backbone = TactileCNN(input_shape, feature_dim, dropout)
        elif encoder_type == "attention":
            self.backbone = TactileAttentionCNN(input_shape, feature_dim, dropout)
        else:
            raise ValueError(f"Unknown tactile encoder type: {encoder_type!r}. Choose 'cnn' or 'attention'.")

        self.token_proj = nn.Linear(feature_dim, n_tokens * feature_dim) if n_tokens > 1 else None

    def forward(self, x: Tensor) -> Tensor:
        feat = self.backbone(x)
        if self.n_tokens == 1:
            return feat.unsqueeze(1)
        feat = self.token_proj(feat)
        return feat.reshape(feat.size(0), self.n_tokens, self.feature_dim)
