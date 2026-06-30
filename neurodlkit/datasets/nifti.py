from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class NiftiDataset(Dataset):
    """Optional NIfTI loader for MRI volumes.

    Requires ``nibabel`` and a CSV with columns for image path, label, and optional id.
    """

    def __init__(self, csv_path: str | Path, image_col: str = "path", label_col: str = "label", id_col: str | None = None, root: str | Path | None = None):
        try:
            import pandas as pd
            import nibabel as nib  # noqa: F401
        except ImportError as exc:
            raise ImportError("NiftiDataset requires the optional 'medical' dependencies: pandas and nibabel.") from exc
        self._nib = __import__("nibabel")
        self.df = pd.read_csv(csv_path)
        self.image_col, self.label_col, self.id_col = image_col, label_col, id_col
        self.root = Path(root) if root else Path(csv_path).parent

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = Path(row[self.image_col])
        if not path.is_absolute():
            path = self.root / path
        arr = self._nib.load(str(path)).get_fdata().astype("float32")
        if arr.ndim == 3:
            arr = arr[None]
        elif arr.ndim == 4 and arr.shape[-1] <= 8:
            arr = np.moveaxis(arr, -1, 0)
        else:
            raise ValueError(f"Unsupported image shape {arr.shape} for {path}.")
        label = torch.tensor([float(row[self.label_col])], dtype=torch.float32)
        sample_id = str(row[self.id_col]) if self.id_col else path.name
        return torch.from_numpy(arr), label, sample_id
