"""Evaluation metrics, chosen for a class-imbalanced clinical problem.

Plain accuracy is misleading here: predicting `nv` for everything already
scores ~67%. We therefore report balanced accuracy, macro F1, macro AUC and -
most importantly for triage - malignant sensitivity (recall over akiec/bcc/mel).
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from .. import CLASSES, MALIGNANT

MALIGNANT_IDX = [CLASSES.index(c) for c in CLASSES if c in MALIGNANT]


@torch.no_grad()
def predict(model, loader, device="cpu") -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    for xb, yb in loader:
        p = torch.softmax(model(xb.to(device)), dim=1)
        probs.append(p.cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def compute_metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    pred = probs.argmax(1)
    present = sorted(set(y.tolist()))
    try:
        auc = float(
            roc_auc_score(y, probs[:, present] / probs[:, present].sum(1, keepdims=True),
                          multi_class="ovr", average="macro", labels=present)
        )
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y, pred, labels=list(range(len(CLASSES))))
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.diag(cm) / np.clip(cm.sum(1), 1, None)
    mal_true = np.isin(y, MALIGNANT_IDX)
    mal_pred = np.isin(pred, MALIGNANT_IDX)
    return {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "macro_auc": auc,
        "malignant_sensitivity": float(mal_pred[mal_true].mean()) if mal_true.any() else float("nan"),
        "malignant_specificity": float((~mal_pred[~mal_true]).mean()) if (~mal_true).any() else float("nan"),
        "per_class_recall": {c: float(recall[i]) for i, c in enumerate(CLASSES)},
        "confusion_matrix": cm.tolist(),
        "num_samples": int(len(y)),
    }


def evaluate(model, loader, device="cpu") -> dict:
    probs, y = predict(model, loader, device)
    return compute_metrics(probs, y)
