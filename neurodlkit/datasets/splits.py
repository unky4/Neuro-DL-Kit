from __future__ import annotations

import numpy as np
from torch.utils.data import Subset


def train_val_test_split(n: int, train: float = 0.7, val: float = 0.15, seed: int = 42):
    if not 0 < train < 1 or not 0 <= val < 1 or train + val >= 1:
        raise ValueError("Expected 0 < train < 1, 0 <= val < 1, and train + val < 1.")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    return idx[:n_train].tolist(), idx[n_train:n_train + n_val].tolist(), idx[n_train + n_val:].tolist()


def stratified_split(labels, train: float = 0.7, val: float = 0.15, seed: int = 42):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        n_train = int(round(len(idx) * train))
        n_val = int(round(len(idx) * val))
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])
    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
    return list(map(int, train_idx)), list(map(int, val_idx)), list(map(int, test_idx))


def subset_splits(dataset, labels=None, train: float = 0.7, val: float = 0.15, seed: int = 42):
    if labels is None:
        idx = train_val_test_split(len(dataset), train, val, seed)
    else:
        idx = stratified_split(labels, train, val, seed)
    return tuple(Subset(dataset, i) for i in idx)
