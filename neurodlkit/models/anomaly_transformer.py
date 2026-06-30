from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class AnomalyAttention(nn.Module):
    """Self-attention that returns both sequence output and attention weights."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        y = self.dropout(attn) @ v
        y = y.transpose(1, 2).reshape(b, t, d)
        return self.out(y), attn


class AnomalyTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = AnomalyAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y, attn = self.attn(self.norm1(x))
        x = x + self.dropout(y)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, attn


class AnomalyTransformer(nn.Module):
    """Transformer autoencoder for time-series reconstruction and anomaly scores."""

    def __init__(self, input_dim: int, d_model: int = 128, depth: int = 3, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([AnomalyTransformerBlock(d_model, num_heads, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        y = self.input_projection(x)
        attentions = []
        for block in self.blocks:
            y, attn = block(y)
            attentions.append(attn)
        rec = self.output_projection(self.norm(y))
        if return_attention:
            return rec, attentions
        return rec

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        err = (self(x) - x).pow(2)
        if reduction == "none":
            return err
        if reduction == "sum":
            return err.sum(dim=-1)
        return err.mean(dim=-1)
