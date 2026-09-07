"""
src/metrics.py
==============
Compute a full panel of classification metrics from out-of-fold predictions.

Everything so far reported AUC (a threshold-independent ranking metric).
This adds accuracy and related threshold-based metrics, computed from the
pooled out-of-fold predictions any run produces (cv_y_true, cv_y_score),
so no re-running of the models is required where those arrays were saved.

Important note on class imbalance
---------------------------------
The groups are imbalanced (more MDD than HC epochs). Plain accuracy is
therefore misleading -- a trivial model predicting the majority class scores
above 50% while learning nothing. BALANCED ACCURACY (the mean of sensitivity
and specificity) is the honest headline metric, alongside AUC. Both are
reported here.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    precision_score, f1_score, confusion_matrix,
)


def compute_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    """
    Full metrics panel from probabilities.

    Parameters
    ----------
    y_true : array of {0=HC, 1=MDD}
    y_score : predicted P(MDD)
    threshold : decision threshold for binary metrics (default 0.5)

    Returns
    -------
    dict of metric name -> value
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")  # true positive rate (MDD)
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")  # true negative rate (HC)

    return {
        "auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) == 2 else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity_MDD": float(sensitivity),
        "specificity_HC": float(specificity),
        "precision_MDD": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1_MDD": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
        "n_MDD_epochs": int((y_true == 1).sum()),
        "n_HC_epochs": int((y_true == 0).sum()),
    }


def print_metrics(metrics: dict, title: str = "") -> None:
    """Pretty-print a metrics dict."""
    if title:
        print(f"\n── {title} ──")
    print(f"  AUC:                {metrics['auc']:.3f}")
    print(f"  Accuracy:           {metrics['accuracy']:.3f}")
    print(f"  Balanced accuracy:  {metrics['balanced_accuracy']:.3f}   <- honest headline (imbalanced classes)")
    print(f"  Sensitivity (MDD):  {metrics['sensitivity_MDD']:.3f}")
    print(f"  Specificity (HC):   {metrics['specificity_HC']:.3f}")
    print(f"  Precision (MDD):    {metrics['precision_MDD']:.3f}")
    print(f"  F1 (MDD):           {metrics['f1_MDD']:.3f}")
    print(f"  (epochs: {metrics['n_MDD_epochs']} MDD, {metrics['n_HC_epochs']} HC)")