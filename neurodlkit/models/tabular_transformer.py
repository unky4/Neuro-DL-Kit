from __future__ import annotations

import torch
from torch import nn


class TabTransformer(nn.Module):
    """Compact FT/TabTransformer-style model for mixed tabular features.

    ``x_num`` has shape ``(batch, n_numeric)``. ``x_cat`` has shape
    ``(batch, n_categorical)`` and stores integer category IDs.
    """

    def __init__(
        self,
        num_numeric: int,
        categorical_cardinalities: list[int] | tuple[int, ...] = (),
        num_outputs: int = 1,
        d_model: int = 64,
        depth: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_numeric = num_numeric
        self.categorical_cardinalities = tuple(categorical_cardinalities)
        self.numeric_tokenizer = nn.Linear(1, d_model) if num_numeric else None
        self.category_embeddings = nn.ModuleList([nn.Embedding(card, d_model) for card in self.categorical_cardinalities])
        n_tokens = num_numeric + len(self.categorical_cardinalities)
        if n_tokens == 0:
            raise ValueError("At least one numeric or categorical feature is required.")
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens + 1, d_model))
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
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, num_outputs))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _tokens(self, x_num: torch.Tensor | None = None, x_cat: torch.Tensor | None = None) -> torch.Tensor:
        tokens: list[torch.Tensor] = []
        if self.num_numeric:
            if x_num is None:
                raise ValueError("x_num is required because num_numeric > 0.")
            if x_num.ndim != 2 or x_num.shape[1] != self.num_numeric:
                raise ValueError(f"x_num must have shape (batch, {self.num_numeric}).")
            tokens.append(self.numeric_tokenizer(x_num.unsqueeze(-1)))
        if self.category_embeddings:
            if x_cat is None:
                raise ValueError("x_cat is required because categorical_cardinalities is not empty.")
            if x_cat.ndim != 2 or x_cat.shape[1] != len(self.category_embeddings):
                raise ValueError(f"x_cat must have shape (batch, {len(self.category_embeddings)}).")
            tokens.extend(emb(x_cat[:, i]).unsqueeze(1) for i, emb in enumerate(self.category_embeddings))
        return torch.cat(tokens, dim=1)

    def forward_features(self, x_num: torch.Tensor | None = None, x_cat: torch.Tensor | None = None) -> torch.Tensor:
        x = self._tokens(x_num, x_cat)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        return self.norm(self.encoder(x))[:, 0]

    def forward(self, x_num: torch.Tensor | None = None, x_cat: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.forward_features(x_num, x_cat))
