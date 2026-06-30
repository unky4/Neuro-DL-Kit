from __future__ import annotations

from .cnn import ConvAutoencoder3D, Simple3DCNN, Tiny3DCNN
from .vit3d import VisionTransformer3D
from .sequence_transformer import SequenceTransformer
from .tabular_transformer import TabTransformer
from .anomaly_transformer import AnomalyTransformer
from .autoencoders import MLPAutoencoder, RegionAutoencoder

_REGISTRY = {
    "tiny3dcnn": Tiny3DCNN,
    "simple3dcnn": Simple3DCNN,
    "conv_autoencoder3d": ConvAutoencoder3D,
    "vit3d": VisionTransformer3D,
    "sequence_transformer": SequenceTransformer,
    "tab_transformer": TabTransformer,
    "anomaly_transformer": AnomalyTransformer,
    "mlp_autoencoder": MLPAutoencoder,
    "region_autoencoder": RegionAutoencoder,
}


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def create_model(name: str, **kwargs):
    key = name.lower().replace("-", "_")
    if key not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {', '.join(available_models())}")
    return _REGISTRY[key](**kwargs)
