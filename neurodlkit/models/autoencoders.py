from __future__ import annotations

import torch
from torch import nn


class MLPAutoencoder(nn.Module):
    """Plain MLP autoencoder for vector/ROI features."""

    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reconstruct(x)

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        err = (self.reconstruct(x) - x).pow(2)
        if reduction == "none":
            return err
        if reduction == "sum":
            return err.sum(dim=-1)
        return err.mean(dim=-1)


class RegionAutoencoder(nn.Module):
    """Dictionary-based autoencoder for per-brain-region time-series.

    Input and output are dictionaries: ``{region_name: tensor}``.
    """

    def __init__(self, region_names: list[str] | tuple[str, ...], input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.region_names = tuple(region_names)
        self.encoder = nn.ModuleDict({name: nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()) for name in self.region_names})
        self.decoder = nn.ModuleDict({name: nn.Linear(hidden_dim, input_dim) for name in self.region_names})

    def encode(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: self.encoder[name](x[name]) for name in self.region_names}

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        return {name: self.decoder[name](z[name]) for name in self.region_names}

    def reconstruction_loss(self, x: dict[str, torch.Tensor], criterion: nn.Module | None = None) -> torch.Tensor:
        criterion = criterion or nn.MSELoss()
        rec = self(x)
        return torch.stack([criterion(rec[name], x[name]) for name in self.region_names]).mean()
