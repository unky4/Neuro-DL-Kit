from __future__ import annotations

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from neurodlkit.datasets import NPZImageDataset, stratified_split, make_synthetic_image_dataset
from neurodlkit.models import Tiny3DCNN, VisionTransformer3D
from neurodlkit.training import Trainer, set_seed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Smoke-train a NeuroDLKit model on the generated random image dataset.")
    p.add_argument("--data", type=Path, default=Path("data/example/images3d.npz"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/example"))
    p.add_argument("--model", choices=["tiny3dcnn", "vit3d"], default="tiny3dcnn")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    if not args.data.exists():
        make_synthetic_image_dataset(args.data, n_samples=64, seed=args.seed)
    dataset = NPZImageDataset(args.data)
    labels = dataset.labels.astype(int)
    train_idx, val_idx, _ = stratified_split(labels, train=0.75, val=0.15, seed=args.seed)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size)
    if args.model == "vit3d":
        model = VisionTransformer3D(in_channels=dataset.images.shape[1], image_size=dataset.images.shape[2:], patch_size=8, embed_dim=64, depth=2, num_heads=4)
    else:
        model = Tiny3DCNN(in_channels=dataset.images.shape[1], base_channels=8)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model, criterion, optimizer, output_dir=args.output_dir)
    trainer.fit(train_loader, val_loader, epochs=args.epochs)


if __name__ == "__main__":
    main()
