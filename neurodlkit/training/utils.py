from __future__ import annotations

import random
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_weighted_sampler(labels) -> WeightedRandomSampler:
    labels = torch.as_tensor(labels).long().view(-1)
    classes, counts = labels.unique(return_counts=True)
    weights = torch.zeros_like(labels, dtype=torch.float32)
    for cls, count in zip(classes, counts):
        weights[labels == cls] = 1.0 / count.float()
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
