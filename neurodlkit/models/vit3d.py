from __future__ import annotations

import math
import torch
from torch import nn


def _triple(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, tuple):
        if len(value) != 3:
            raise ValueError("Expected a 3-tuple.")
        return value
    return (value, value, value)


class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int | tuple[int, int, int]):
        super().__init__()
        self.patch_size = _triple(patch_size)
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        x = self.proj(x)
        grid = x.shape[-3:]
        x = x.flatten(2).transpose(1, 2)
        return x, grid


class VisionTransformer3D(nn.Module):
    """3D Vision Transformer for MRI/NIfTI-like volumes.

    The architecture uses volumetric patch embedding, a learnable CLS token,
    positional embeddings, and a PyTorch Transformer encoder.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_outputs: int = 1,
        image_size: int | tuple[int, int, int] = (32, 32, 32),
        patch_size: int | tuple[int, int, int] = 8,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pool: str = "cls",
    ):
        super().__init__()
        image_size = _triple(image_size)
        patch_size = _triple(patch_size)
        if any(s % p != 0 for s, p in zip(image_size, patch_size)):
            raise ValueError("image_size must be divisible by patch_size in every dimension.")
        self.pool = pool
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, patch_size)
        num_patches = math.prod(s // p for s, p in zip(image_size, patch_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_outputs)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        if x.shape[1] != self.pos_embed.shape[1]:
            raise ValueError(
                f"Input produced {x.shape[1] - 1} patches but model was built for {self.pos_embed.shape[1] - 1}."
            )
        x = self.pos_drop(x + self.pos_embed)
        x = self.norm(self.encoder(x))
        return x[:, 0] if self.pool == "cls" else x[:, 1:].mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))
