from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainResult:
    history: list[dict[str, float]] = field(default_factory=list)
    best_metric: float | None = None
    best_epoch: int | None = None


def _move_batch(batch, device: torch.device):
    x, y, *rest = batch
    if isinstance(x, dict):
        x = {k: v.to(device) for k, v in x.items()}
    else:
        x = x.to(device)
    return x, y.to(device), rest


def _forward(model: nn.Module, x):
    if isinstance(x, dict):
        return model(**x)
    return model(x)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: Callable, device: str | torch.device = "cpu") -> dict[str, float]:
    device = torch.device(device)
    model.eval()
    total_loss, n = 0.0, 0
    for batch in loader:
        x, y, _ = _move_batch(batch, device)
        pred = _forward(model, x)
        loss = criterion(pred, y)
        total_loss += loss.item() * y.shape[0]
        n += y.shape[0]
    return {"loss": total_loss / max(n, 1)}


class Trainer:
    """Minimal, reusable trainer for classification, regression, and reconstruction tasks."""

    def __init__(
        self,
        model: nn.Module,
        criterion: Callable,
        optimizer: torch.optim.Optimizer,
        device: str | torch.device | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        output_dir: str | Path = "outputs",
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss, n = 0.0, 0
        for batch in loader:
            x, y, _ = _move_batch(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            pred = _forward(self.model, x)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * y.shape[0]
            n += y.shape[0]
        if self.scheduler is not None:
            self.scheduler.step()
        return total_loss / max(n, 1)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None, epochs: int = 10, save_best: bool = True) -> TrainResult:
        result = TrainResult(best_metric=float("inf"))
        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            row = {"epoch": float(epoch), "train_loss": train_loss}
            metric = train_loss
            if val_loader is not None:
                val_stats = evaluate(self.model, val_loader, self.criterion, self.device)
                row["val_loss"] = val_stats["loss"]
                metric = val_stats["loss"]
            if save_best and metric < (result.best_metric or float("inf")):
                result.best_metric = metric
                result.best_epoch = epoch
                torch.save(self.model.state_dict(), self.output_dir / "best_model.pt")
            result.history.append(row)
            print(" | ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        return result

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        preds, targets = [], []
        for batch in loader:
            x, y, _ = _move_batch(batch, self.device)
            preds.append(_forward(self.model, x).detach().cpu())
            targets.append(y.detach().cpu())
        return torch.cat(preds), torch.cat(targets)
