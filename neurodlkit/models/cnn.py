from __future__ import annotations

import torch
from torch import nn


class Tiny3DCNN(nn.Module):
    """Small 3D CNN for fast volumetric classification or regression.

    It uses adaptive pooling, so it works with any reasonably sized 3D volume.
    """

    def __init__(self, in_channels: int = 1, num_outputs: int = 1, base_channels: int = 16, dropout: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(2),
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(2),
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(base_channels * 4, num_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class Simple3DCNN(nn.Module):
    """Deeper generic 3D CNN for classification or regression."""

    def __init__(
        self,
        in_channels: int = 1,
        num_outputs: int = 1,
        channels: tuple[int, ...] = (16, 32, 64, 128),
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        current = in_channels
        for out in channels:
            layers.extend(
                [
                    nn.Conv3d(current, out, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm3d(out),
                    nn.SiLU(inplace=True),
                    nn.Conv3d(out, out, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm3d(out),
                    nn.SiLU(inplace=True),
                    nn.MaxPool3d(2),
                ]
            )
            current = out
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool3d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(current, num_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class ConvAutoencoder3D(nn.Module):
    """3D convolutional autoencoder for reconstruction/anomaly-score experiments."""

    def __init__(self, in_channels: int = 1, latent_channels: int = 64, base_channels: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 2, latent_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(base_channels, in_channels, 4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        rec = self.decoder(self.encoder(x))
        return rec[..., : x.shape[-3], : x.shape[-2], : x.shape[-1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reconstruct(x)

    @torch.no_grad()
    def anomaly_map(self, x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        err = (self.reconstruct(x) - x).abs()
        if reduction == "none":
            return err
        if reduction == "sum":
            return err.sum(dim=1, keepdim=True)
        return err.mean(dim=1, keepdim=True)
