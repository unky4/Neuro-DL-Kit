from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class NPZImageDataset(Dataset):
    """Loads ``images`` and ``labels`` arrays from an ``.npz`` file.

    Expected image shape is ``(N, C, D, H, W)``. Labels are returned as float
    tensors with shape ``(1,)`` for binary/regression tasks.
    """

    def __init__(self, path: str | Path, image_key: str = "images", label_key: str = "labels"):
        data = np.load(Path(path), allow_pickle=False)
        self.images = data[image_key].astype("float32")
        self.labels = data[label_key].astype("float32")
        self.ids = data["ids"].astype(str) if "ids" in data else np.array([f"sample_{i:05d}" for i in range(len(self.labels))])
        if self.images.ndim != 5:
            raise ValueError("images must have shape (N, C, D, H, W).")
        if len(self.images) != len(self.labels):
            raise ValueError("images and labels must have the same length.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.images[idx]), torch.tensor([self.labels[idx]], dtype=torch.float32), str(self.ids[idx])


class NPZSequenceDataset(Dataset):
    """Loads ``sequences`` and ``labels`` from an ``.npz`` file.

    Expected sequence shape is ``(N, T, F)``.
    """

    def __init__(self, path: str | Path, sequence_key: str = "sequences", label_key: str = "labels"):
        data = np.load(Path(path), allow_pickle=False)
        self.sequences = data[sequence_key].astype("float32")
        self.labels = data[label_key].astype("float32")
        self.ids = data["ids"].astype(str) if "ids" in data else np.array([f"sample_{i:05d}" for i in range(len(self.labels))])
        if self.sequences.ndim != 3:
            raise ValueError("sequences must have shape (N, T, F).")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.sequences[idx]), torch.tensor([self.labels[idx]], dtype=torch.float32), str(self.ids[idx])


class NPZTabularDataset(Dataset):
    """Loads numeric/categorical tabular arrays and labels from an ``.npz`` file."""

    def __init__(self, path: str | Path, label_key: str = "labels"):
        data = np.load(Path(path), allow_pickle=False)
        self.x_num = data["x_num"].astype("float32") if "x_num" in data else None
        self.x_cat = data["x_cat"].astype("int64") if "x_cat" in data else None
        self.labels = data[label_key].astype("float32")
        self.ids = data["ids"].astype(str) if "ids" in data else np.array([f"sample_{i:05d}" for i in range(len(self.labels))])
        if self.x_num is None and self.x_cat is None:
            raise ValueError("NPZ must contain x_num or x_cat.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        item = {}
        if self.x_num is not None:
            item["x_num"] = torch.from_numpy(self.x_num[idx])
        if self.x_cat is not None:
            item["x_cat"] = torch.from_numpy(self.x_cat[idx])
        return item, torch.tensor([self.labels[idx]], dtype=torch.float32), str(self.ids[idx])
