from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class BinaryClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    confusion_matrix: tuple[int, int, int, int]

    @classmethod
    def from_logits(cls, logits, targets, threshold: float = 0.5):
        probs = torch.sigmoid(torch.as_tensor(logits)).detach().cpu().numpy().reshape(-1)
        return cls.from_probabilities(probs, targets, threshold)

    @classmethod
    def from_probabilities(cls, probs, targets, threshold: float = 0.5):
        probs = np.asarray(probs).reshape(-1)
        y = np.asarray(targets).reshape(-1).astype(int)
        pred = (probs >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        eps = 1e-12
        return cls(
            accuracy=(tp + tn) / max(tp + tn + fp + fn, 1),
            precision=tp / (tp + fp + eps),
            recall=tp / (tp + fn + eps),
            specificity=tn / (tn + fp + eps),
            f1=2 * tp / (2 * tp + fp + fn + eps),
            confusion_matrix=(tn, fp, fn, tp),
        )


def classification_metrics(logits, targets, threshold: float = 0.5) -> dict[str, float]:
    m = BinaryClassificationMetrics.from_logits(logits, targets, threshold)
    return {
        "accuracy": m.accuracy,
        "precision": m.precision,
        "recall": m.recall,
        "specificity": m.specificity,
        "f1": m.f1,
    }


def regression_metrics(preds, targets) -> dict[str, float]:
    p = np.asarray(preds).reshape(-1)
    y = np.asarray(targets).reshape(-1)
    err = p - y
    return {"mae": float(np.abs(err).mean()), "mse": float((err ** 2).mean()), "rmse": float(np.sqrt((err ** 2).mean()))}
