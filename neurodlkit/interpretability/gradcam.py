from __future__ import annotations

import numpy as np
import torch
from torch import nn


def normalize_cam(cam: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    flat = cam.flatten(1)
    min_v = flat.min(dim=1).values.view(-1, *([1] * (cam.ndim - 1)))
    max_v = flat.max(dim=1).values.view(-1, *([1] * (cam.ndim - 1)))
    return (cam - min_v) / (max_v - min_v + eps)


class GradCAM3D:
    """Framework-independent Grad-CAM for 3D CNNs.

    Example:
        ``cam = GradCAM3D(model, target_layer=model.features[-2]); heatmap = cam(x)``
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __call__(self, x: torch.Tensor, target_index: int | None = None) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if target_index is None:
            target = logits[:, 0].sum() if logits.shape[-1] == 1 else logits.max(dim=1).values.sum()
        else:
            target = logits[:, target_index].sum()
        target.backward(retain_graph=True)
        if self.activations is None or self.gradients is None:
            raise RuntimeError("No activations/gradients captured; check target_layer.")
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam.unsqueeze(1), size=x.shape[-3:], mode="trilinear", align_corners=False).squeeze(1)
        return normalize_cam(cam)

    @staticmethod
    def to_numpy(cam: torch.Tensor) -> np.ndarray:
        return cam.detach().cpu().numpy()
