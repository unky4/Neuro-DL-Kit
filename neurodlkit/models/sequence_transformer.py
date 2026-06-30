from __future__ import annotations

import math
import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class SequenceTransformer(nn.Module):
    """Transformer encoder for rfMRI/BOLD/time-series classification or regression.

    Expected input shape is ``(batch, time, features)``.
    """

    def __init__(
        self,
        input_dim: int,
        num_outputs: int = 1,
        d_model: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__()
        self.pooling = pooling
        self.input_projection = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if pooling == "cls" else None
        self.positional_encoding = SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_outputs)

    def forward_features(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_projection(x)
        if self.cls_token is not None:
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
            if padding_mask is not None:
                padding_mask = torch.cat(
                    [torch.zeros(x.shape[0], 1, dtype=torch.bool, device=x.device), padding_mask], dim=1
                )
        x = self.positional_encoding(x)
        x = self.norm(self.encoder(x, src_key_padding_mask=padding_mask))
        if self.pooling == "cls":
            return x[:, 0]
        if padding_mask is None:
            return x.mean(dim=1)
        valid = (~padding_mask).float().unsqueeze(-1)
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.forward_features(x, padding_mask))
