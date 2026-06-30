from pathlib import Path
import torch
from torch.utils.data import DataLoader
from neurodlkit.datasets import NPZImageDataset, make_synthetic_image_dataset
from neurodlkit.models import Tiny3DCNN
from neurodlkit.training import Trainer, set_seed

set_seed(42)
data_path = Path("data/example/images3d.npz")
make_synthetic_image_dataset(data_path, n_samples=32)
dataset = NPZImageDataset(data_path)
loader = DataLoader(dataset, batch_size=8, shuffle=True)
model = Tiny3DCNN(in_channels=1, base_channels=8)
trainer = Trainer(model, torch.nn.BCEWithLogitsLoss(), torch.optim.AdamW(model.parameters(), lr=1e-3))
trainer.fit(loader, epochs=1)
