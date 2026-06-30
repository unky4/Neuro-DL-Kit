from __future__ import annotations

import argparse
from pathlib import Path
from neurodlkit.datasets.synthetic import make_synthetic_image_dataset, make_synthetic_sequence_dataset, make_synthetic_tabular_dataset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate random test datasets for NeuroDLKit.")
    p.add_argument("--output-dir", type=Path, default=Path("data/example"))
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = make_synthetic_image_dataset(args.output_dir / "images3d.npz", args.n_samples, seed=args.seed, task=args.task)
    seq_path = make_synthetic_sequence_dataset(args.output_dir / "sequences.npz", args.n_samples, seed=args.seed, task=args.task)
    tab_path = make_synthetic_tabular_dataset(args.output_dir / "tabular.npz", args.n_samples, seed=args.seed, task=args.task)
    print(f"Wrote {image_path}")
    print(f"Wrote {seq_path}")
    print(f"Wrote {tab_path}")


if __name__ == "__main__":
    main()
