"""
src/figures.py
==============
Minimal interpretive figures for Objective 2:
  - Adjacency (wPLI) matrix heatmap
  - Example harmonic topographies on the scalp
"""


from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

import config


def plot_adjacency_matrix(
    wpli_matrix: np.ndarray, ch_names: list[str], subject_id: str, save: bool = True
):
    """Plot the wPLI adjacency matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(wpli_matrix, cmap="viridis")
    ax.set_xticks(range(len(ch_names)))
    ax.set_yticks(range(len(ch_names)))
    ax.set_xticklabels(ch_names, rotation=90, fontsize=7)
    ax.set_yticklabels(ch_names, fontsize=7)
    ax.set_title(f"{subject_id} — alpha-band wPLI adjacency matrix")
    fig.colorbar(im, ax=ax, label="wPLI")
    fig.tight_layout()

    if save:
        out_path = config.FIGURES_DIR / f"{subject_id}_adjacency_matrix.png"
        fig.savefig(out_path, dpi=150)
        print(f"  Saved: {out_path}")

    return fig


def plot_harmonic_topographies(
    eigenvectors: np.ndarray,
    ch_names: list[str],
    subject_id: str,
    modes_to_plot: list[int] = (0, 1, 2, -3, -2, -1),
    save: bool = True,
):
    """
    Plot example harmonic topographies (low and high modes) on a scalp layout.

    Parameters
    ----------
    eigenvectors : ndarray, shape (n_channels, n_channels)
    ch_names : list of str
    subject_id : str
    modes_to_plot : list of int
        Mode indices to display (negative indices index from the high-frequency end)
    """
    info = mne.create_info(ch_names=ch_names, sfreq=config.SFREQ, ch_types="eeg")
    info.set_montage("standard_1020", on_missing="warn")

    n_modes = len(modes_to_plot)
    fig, axes = plt.subplots(1, n_modes, figsize=(3 * n_modes, 3.5))

    for ax, mode_idx in zip(axes, modes_to_plot):
        mne.viz.plot_topomap(
            eigenvectors[:, mode_idx], info, axes=ax, show=False, cmap="RdBu_r"
        )
        label = f"mode {mode_idx}" if mode_idx >= 0 else f"mode {eigenvectors.shape[1] + mode_idx}"
        ax.set_title(label, fontsize=10)

    fig.suptitle(f"{subject_id} — example graph harmonic topographies")
    fig.tight_layout()

    if save:
        out_path = config.FIGURES_DIR / f"{subject_id}_harmonic_topographies.png"
        fig.savefig(out_path, dpi=150)
        print(f"  Saved: {out_path}")

    return fig


def make_interpretive_figures(results: dict, subject_ids: list[str] | None = None):
    """
    Generate the minimal interpretive figure set for one or more subjects.

    Parameters
    ----------
    results : dict
        Output of src.objective2.run_objective2 (the `results` dict, not the
        feature table)
    subject_ids : list of str, optional
        Defaults to the first subject in `results` if not specified
    """
    if subject_ids is None:
        subject_ids = [next(iter(results))]

    for sid in subject_ids:
        if sid not in results:
            print(f"  ✗ {sid}: not found in results")
            continue
        r = results[sid]
        plot_adjacency_matrix(r["wpli"], r["ch_names"], sid)
        plot_harmonic_topographies(r["eigenvectors"], r["ch_names"], sid)

  
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix


def plot_roc_curve(y_true, y_score, save=True):
    """ROC curve from pooled out-of-fold predictions."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — pooled out-of-fold predictions")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    fig.tight_layout()
    if save:
        out_path = config.FIGURES_DIR / "objective3_roc_curve.png"
        fig.savefig(out_path, dpi=150)
        print(f"  Saved: {out_path}")
    return fig


def plot_per_fold_auc(fold_results, save=True):
    """Bar chart of AUC per outer fold, showing fold-to-fold variability."""
    folds = [f"Fold {r['fold']}" for r in fold_results]
    aucs = [r["auc"] for r in fold_results]
    valid_aucs = [a for a in aucs if not np.isnan(a)]
    mean_auc = np.mean(valid_aucs) if valid_aucs else np.nan

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["lightgray" if np.isnan(a) else "steelblue" for a in aucs]
    plotted = [0 if np.isnan(a) else a for a in aucs]
    bars = ax.bar(folds, plotted, color=colors, edgecolor="black")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (0.5)")
    ax.axhline(mean_auc, color="crimson", linewidth=1.5, label=f"Mean AUC = {mean_auc:.3f}")

    for bar, a in zip(bars, aucs):
        label = "nan" if np.isnan(a) else f"{a:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2, 0.02, label,
                ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-fold AUC (outer CV)")
    ax.legend()
    fig.tight_layout()
    if save:
        out_path = config.FIGURES_DIR / "objective3_per_fold_auc.png"
        fig.savefig(out_path, dpi=150)
        print(f"  Saved: {out_path}")
    return fig


def plot_confusion_matrix(y_true, y_score, threshold=0.5, save=True):
    """Confusion matrix at a given probability threshold."""
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["HC", "MDD"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (threshold = {threshold})")
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=14)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if save:
        out_path = config.FIGURES_DIR / "objective3_confusion_matrix.png"
        fig.savefig(out_path, dpi=150)
        print(f"  Saved: {out_path}")
    return fig


def plot_feature_importance(coefs, feature_names, save=True):
    """Bar chart of ElasticNet coefficients."""
    order = np.argsort(np.abs(coefs))[::-1]
    coefs_sorted = np.array(coefs)[order]
    names_sorted = np.array(feature_names)[order]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["crimson" if c < 0 else "steelblue" for c in coefs_sorted]
    ax.barh(range(len(coefs_sorted)), coefs_sorted, color=colors, edgecolor="black")
    ax.set_yticks(range(len(coefs_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("ElasticNet coefficient")
    ax.set_title("Feature importance (blue = MDD-associated, red = HC-associated)")
    fig.tight_layout()
    if save:
        out_path = config.FIGURES_DIR / "objective3_feature_importance.png"
        fig.savefig(out_path, dpi=150)
        print(f"  Saved: {out_path}")
    return fig


def plot_permutation_distribution(permuted_aucs, true_auc, p_value, save=True):
    """
    Plot the null distribution of permuted AUCs with the true (unshuffled)
    AUC marked, for permutation test reporting.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(permuted_aucs, bins=30, color="lightgray", edgecolor="black", alpha=0.8)
    ax.axvline(true_auc, color="crimson", linewidth=2,
               label=f"True AUC = {true_auc:.3f}\n(p = {p_value:.3f})")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (AUC = 0.5)")

    ax.set_xlabel("AUC")
    ax.set_ylabel("Count (permutations)")
    ax.set_title("Permutation test: null distribution vs. true AUC")
    ax.legend()
    fig.tight_layout()

    if save:
        out_path = config.FIGURES_DIR / "objective3_permutation_distribution.png"
        fig.savefig(out_path, dpi=150)
        print(f"  Saved: {out_path}")

    return fig