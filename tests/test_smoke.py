from pathlib import Path
import torch
from torch.utils.data import DataLoader

from neurodlkit.datasets import NPZImageDataset, make_synthetic_image_dataset
from neurodlkit.models import Tiny3DCNN, VisionTransformer3D, SequenceTransformer, TabTransformer, AnomalyTransformer
from neurodlkit.training import Trainer


def test_image_dataset_and_tiny_cnn(tmp_path: Path):
    path = make_synthetic_image_dataset(tmp_path / "images.npz", n_samples=12, shape=(1, 16, 16, 16))
    ds = NPZImageDataset(path)
    x, y, sid = ds[0]
    assert x.shape == (1, 16, 16, 16)
    assert y.shape == (1,)
    assert sid.startswith("sample_")
    model = Tiny3DCNN(in_channels=1, base_channels=4)
    out = model(x.unsqueeze(0))
    assert out.shape == (1, 1)


def test_vit3d_forward():
    model = VisionTransformer3D(in_channels=1, image_size=(16, 16, 16), patch_size=8, embed_dim=32, depth=1, num_heads=4)
    out = model(torch.randn(2, 1, 16, 16, 16))
    assert out.shape == (2, 1)


def test_sequence_and_tabular_forward():
    seq_model = SequenceTransformer(input_dim=5, d_model=32, depth=1, num_heads=4)
    assert seq_model(torch.randn(2, 10, 5)).shape == (2, 1)
    tab_model = TabTransformer(num_numeric=3, categorical_cardinalities=(4, 5), d_model=32, depth=1, num_heads=4)
    out = tab_model(torch.randn(2, 3), torch.tensor([[1, 2], [3, 4]]))
    assert out.shape == (2, 1)


def test_anomaly_transformer_forward():
    model = AnomalyTransformer(input_dim=4, d_model=32, depth=1, num_heads=4)
    x = torch.randn(2, 12, 4)
    rec, attn = model(x, return_attention=True)
    assert rec.shape == x.shape
    assert len(attn) == 1


def test_one_training_step(tmp_path: Path):
    path = make_synthetic_image_dataset(tmp_path / "images.npz", n_samples=16, shape=(1, 16, 16, 16))
    loader = DataLoader(NPZImageDataset(path), batch_size=4)
    model = Tiny3DCNN(in_channels=1, base_channels=4)
    trainer = Trainer(model, torch.nn.BCEWithLogitsLoss(), torch.optim.AdamW(model.parameters(), lr=1e-3), output_dir=tmp_path / "out")
    result = trainer.fit(loader, epochs=1)
    assert len(result.history) == 1
    assert (tmp_path / "out" / "best_model.pt").exists()
