"""Evaluation metrics for binding site prediction.

Computes threshold-independent metrics (AUC-ROC, AUPRC) and
threshold-dependent metrics (MCC, precision, recall, specificity).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass
class MetricsResult:
    """Container for computed evaluation metrics.

    Attributes:
        auc_roc: Area Under the ROC Curve.
        auprc: Area Under the Precision-Recall Curve.
        mcc: Matthews Correlation Coefficient (at threshold=0.5).
        precision: Precision at threshold=0.5.
        recall: Recall (sensitivity) at threshold=0.5.
        specificity: Specificity at threshold=0.5.
    """

    auc_roc: float
    auprc: float
    mcc: float
    accuracy: float
    precision: float
    recall: float
    specificity: float

    def __str__(self) -> str:
        return (
            f"AUC-ROC: {self.auc_roc:.4f} | "
            f"AUPRC: {self.auprc:.4f} | "
            f"MCC: {self.mcc:.4f} | "
            f"ACC: {self.accuracy:.4f} | "
            f"Precision: {self.precision:.4f} | "
            f"Recall: {self.recall:.4f} | "
            f"Specificity: {self.specificity:.4f}"
        )


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> MetricsResult:
    """Compute comprehensive evaluation metrics.

    Args:
        predictions: Predicted probabilities, shape (N,).
        labels: Ground truth binary labels, shape (N,).
        threshold: Classification threshold for binary metrics.

    Returns:
        MetricsResult with all computed metrics.

    Raises:
        ValueError: If inputs are empty or labels contain only one class.
    """
    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)

    if len(predictions) == 0:
        raise ValueError("Cannot compute metrics on empty predictions.")

    # Threshold-independent metrics.
    auc_roc = roc_auc_score(labels, predictions)
    precisions, recalls, _ = precision_recall_curve(labels, predictions)
    auprc = auc(recalls, precisions)

    # Threshold-dependent metrics.
    binary_preds = (predictions > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, binary_preds).ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Matthews Correlation Coefficient.
    denom = np.sqrt(
        float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    )
    mcc = float(tp * tn - fn * fp) / denom if denom > 0 else 0.0

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return MetricsResult(
        auc_roc=float(auc_roc),
        auprc=float(auprc),
        mcc=mcc,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        specificity=specificity,
    )
