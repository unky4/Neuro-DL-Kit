# Neuro-DL-Kit

Experimental PyTorch utilities and model prototypes for neuroimaging-style deep-learning workflows.

This repository contains research/prototyping code developed while exploring deep-learning approaches for biomedical imaging data. It is not a clinically validated tool and is not associated with a published deep-learning study.

## Features

- **3D CNNs**: `Tiny3DCNN` and `Simple3DCNN` for volumetric classification or regression.
- **3D Vision Transformer**: `VisionTransformer3D` with volumetric patch embeddings.
- **rfMRI/time-series Transformer**: `SequenceTransformer` for `(batch, time, features)` inputs.
- **Tabular Transformer**: `TabTransformer` for mixed numeric and categorical features.
- **Anomaly and reconstruction models**: `ConvAutoencoder3D`, `MLPAutoencoder`, `RegionAutoencoder`, and `AnomalyTransformer`.
- **Interpretability**: `GradCAM3D` for 3D CNN activation heatmaps.
- **Training utilities**: reusable trainer, binary classification metrics, regression metrics, weighted sampling, and deterministic seeding.
- **Dataset utilities**: `.npz` dataset loaders, reproducible synthetic data generation, train/validation/test splitting, and an optional NIfTI loader.
- **Example scripts**: command-line tools to generate random test data and run a one-epoch smoke-training job.

## Repository layout

```text
neurodlkit/
  datasets/          # NPZ, synthetic, split, and optional NIfTI loaders
  interpretability/  # Grad-CAM for 3D CNNs
  models/            # CNN, ViT, sequence, tabular, autoencoder, and anomaly models
  scripts/           # CLI scripts for example data generation and smoke training
  training/          # Trainer, metrics, sampler, and seed utilities
tests/               # CPU smoke tests
examples/            # Minimal usage examples
```

## Installation

### Recommended editable install

```bash
cd neurodlkit_repo
python -m pip install -e .
```

### Install from `requirements.txt`

A `requirements.txt` file is included for environments that expect one:

```bash
python -m pip install -r requirements.txt
```

The package also uses `pyproject.toml`, which is the modern Python packaging format. Keeping both files makes the project convenient for local development, CI systems, and simple notebook/server environments.

### Optional extras

Medical-imaging dependencies:

```bash
python -m pip install -e ".[medical]"
```

Development dependencies:

```bash
python -m pip install -e ".[dev]"
```

## Generate random example data

The generated data is intentionally synthetic. It is useful for checking that the package, dataloaders, models, and training loop run end to end.

```bash
neurodl-generate-example --output-dir data/example --n-samples 64
```

This creates:

```text
data/example/images3d.npz
data/example/sequences.npz
data/example/tabular.npz
```

## Smoke-train a model

Train a small 3D CNN for one epoch on the generated random 3D image dataset:

```bash
neurodl-train-example --data data/example/images3d.npz --model tiny3dcnn --epochs 1
```

Train the 3D ViT smoke example:

```bash
neurodl-train-example --data data/example/images3d.npz --model vit3d --epochs 1
```

## Python usage

```python
import torch
from torch.utils.data import DataLoader

from neurodlkit.datasets import NPZImageDataset
from neurodlkit.models import Tiny3DCNN
from neurodlkit.training import Trainer

train_ds = NPZImageDataset("data/example/images3d.npz")
loader = DataLoader(train_ds, batch_size=8, shuffle=True)

model = Tiny3DCNN(in_channels=1, num_outputs=1)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

trainer = Trainer(model, criterion, optimizer)
trainer.fit(loader, epochs=1)
```

## Model registry

Models can also be constructed by name:

```python
from neurodlkit.models import create_model

model = create_model("tiny3dcnn", in_channels=1, num_outputs=1)
```

Available registry names:

```text
tiny3dcnn
simple3dcnn
vit3d
sequence_transformer
tab_transformer
conv_autoencoder3d
mlp_autoencoder
region_autoencoder
anomaly_transformer
```

## Running tests

```bash
python -m pip install -e ".[dev]"
pytest
```

The tests are lightweight CPU smoke tests that verify model forward passes, dataset generation, and a minimal training run.

## GitHub setup

1. Create a new repository.
2. Copy the contents of this folder into the repository.
3. Run `python -m pip install -e ".[dev]"`.
4. Run `pytest`.
5. Commit the package files.

Generated datasets, training outputs, model checkpoints, and cache folders are ignored by `.gitignore`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
