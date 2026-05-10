"""
src/utils/metrics.py
────────────────────
Evaluation helpers: accuracy, weighted F1, per-class breakdown.
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str] = None,
) -> Dict:
    """
    Compute accuracy, weighted F1, per-class F1, and confusion matrix.

    Args:
        y_true       : ground-truth labels (list or array)
        y_pred       : predicted labels
        class_names  : optional human-readable class names

    Returns:
        dict with keys: accuracy, f1_weighted, f1_per_class,
                        confusion_matrix, classification_report
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1c = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm  = confusion_matrix(y_true, y_pred)

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )

    return {
        "accuracy":               float(acc),
        "f1_weighted":            float(f1w),
        "f1_per_class":           f1c.tolist(),
        "confusion_matrix":       cm.tolist(),
        "classification_report":  report,
    }


class RunningMetrics:
    """
    Accumulates predictions and ground-truth labels across batches,
    then computes metrics once at epoch end.

    Usage:
        metrics = RunningMetrics()
        for batch in loader:
            logits = model(...)
            metrics.update(logits, labels)
        result = metrics.compute(class_names)
        metrics.reset()
    """

    def __init__(self):
        self._preds: List[int] = []
        self._trues: List[int] = []
        self._loss_sum: float = 0.0
        self._n_batches: int = 0

    def reset(self):
        self._preds.clear()
        self._trues.clear()
        self._loss_sum = 0.0
        self._n_batches = 0

    def update(self, logits, labels, loss: float = None):
        """
        Args:
            logits : (B, num_classes) raw model output (torch.Tensor)
            labels : (B,) ground-truth class indices
            loss   : scalar batch loss (optional, for tracking)
        """
        import torch
        preds = logits.argmax(dim=1).detach().cpu().tolist()
        trues = labels.detach().cpu().tolist()
        self._preds.extend(preds)
        self._trues.extend(trues)
        if loss is not None:
            self._loss_sum += float(loss)
            self._n_batches += 1

    def compute(self, class_names: List[str] = None) -> Dict:
        m = compute_metrics(self._trues, self._preds, class_names)
        if self._n_batches > 0:
            m["avg_loss"] = self._loss_sum / self._n_batches
        return m

    @property
    def num_samples(self):
        return len(self._trues)
