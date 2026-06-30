from __future__ import annotations

from pathlib import Path
import numpy as np


def _ids(n: int) -> np.ndarray:
    return np.array([f"sample_{i:05d}" for i in range(n)])


def make_synthetic_image_dataset(
    output_path: str | Path,
    n_samples: int = 64,
    shape: tuple[int, int, int, int] = (1, 32, 32, 32),
    task: str = "classification",
    seed: int = 42,
) -> Path:
    rng = np.random.default_rng(seed)
    images = rng.normal(0, 1, size=(n_samples, *shape)).astype("float32")
    signal = images[:, :, : shape[1] // 2].mean(axis=(1, 2, 3, 4))
    labels = (signal > np.median(signal)).astype("float32") if task == "classification" else signal.astype("float32")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, images=images, labels=labels, ids=_ids(n_samples))
    return output_path


def make_synthetic_sequence_dataset(
    output_path: str | Path,
    n_samples: int = 64,
    timesteps: int = 80,
    features: int = 16,
    task: str = "classification",
    seed: int = 42,
) -> Path:
    rng = np.random.default_rng(seed)
    sequences = rng.normal(0, 1, size=(n_samples, timesteps, features)).astype("float32")
    signal = sequences[:, :, 0].mean(axis=1)
    labels = (signal > np.median(signal)).astype("float32") if task == "classification" else signal.astype("float32")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, sequences=sequences, labels=labels, ids=_ids(n_samples))
    return output_path


def make_synthetic_tabular_dataset(
    output_path: str | Path,
    n_samples: int = 128,
    num_numeric: int = 10,
    categorical_cardinalities: tuple[int, ...] = (4, 6),
    task: str = "classification",
    seed: int = 42,
) -> Path:
    rng = np.random.default_rng(seed)
    x_num = rng.normal(0, 1, size=(n_samples, num_numeric)).astype("float32")
    x_cat = np.stack([rng.integers(0, c, size=n_samples) for c in categorical_cardinalities], axis=1).astype("int64") if categorical_cardinalities else None
    signal = x_num[:, 0] + (x_cat[:, 0] / max(categorical_cardinalities[0] - 1, 1) if x_cat is not None else 0)
    labels = (signal > np.median(signal)).astype("float32") if task == "classification" else signal.astype("float32")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"x_num": x_num, "labels": labels, "ids": _ids(n_samples)}
    if x_cat is not None:
        kwargs["x_cat"] = x_cat
        kwargs["categorical_cardinalities"] = np.array(categorical_cardinalities)
    np.savez_compressed(output_path, **kwargs)
    return output_path
