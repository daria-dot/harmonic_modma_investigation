"""
src/graph_spectral_density.py
=============================
Graph spectral density (GSD) features: a compact, literature-grounded summary
of how EEG signal energy is distributed across the graph-frequency spectrum.

Motivation
----------
The existing graph-harmonic analysis uses each mode's power as a separate
feature (~20+ features). This module instead summarises the *shape* of the
energy distribution across the graph spectrum with a few interpretable numbers
per epoch. This "connectome spectral density" framing follows the graph signal
processing literature (Rué-Queralt et al.; Glomb et al.; Atasoy et al.; Van De
Ville and colleagues), where network harmonics are ordered by smoothness and
the concentration of signal energy in the smoothest (low graph-frequency) modes
indexes structure-function coupling / global integration.

Pre-specified hypothesis
------------------------
Depression is associated with altered structure-function coupling -- a shift in
how EEG energy distributes between integrated (low-graph-frequency) and
segregated (high-graph-frequency) network modes. GSD features may therefore
discriminate MDD from HC. (Reported regardless of outcome.)

Note on adaptation
------------------
The strongest literature versions build the graph from *structural* (dMRI)
connectomes. Here the graph is the *functional* wPLI connectivity already used
for the harmonic analysis, so this is a functional graph-spectral-density
variant. This is stated explicitly rather than glossed over.

Features (per epoch)
--------------------
  gsd_energy_low  : fraction of total signal energy in the lowest-k harmonics
                    (broadcasting / integration index)
  gsd_energy_high : fraction in the highest-k harmonics (segregation index)
  gsd_centroid    : graph-spectral centroid -- mean graph frequency (eigenvalue)
                    weighted by per-mode energy; "where the energy sits on the
                    graph-frequency axis"
  gsd_spread      : graph-spectral spread -- weighted standard deviation of the
                    graph frequency; how dispersed the energy is across the
                    spectrum
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import objective2


def compute_gsd_features(eigenvalues, per_epoch_power, k_low=10, k_high=10):
    """
    Compute graph spectral density features from an eigenvalue spectrum and
    per-epoch per-mode power.

    Parameters
    ----------
    eigenvalues : ndarray (n_modes,)
        Graph Laplacian eigenvalues, ascending (graph frequencies).
    per_epoch_power : ndarray (n_epochs, n_modes)
        Signal energy each epoch places in each harmonic mode
        (objective2 'per_epoch_power').
    k_low, k_high : int
        Number of lowest / highest modes for the energy-ratio features.

    Returns
    -------
    X : ndarray (n_epochs, 4)
    names : list of str
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    P = np.asarray(per_epoch_power, dtype=float)
    n_epochs, n_modes = P.shape

    k_low = min(k_low, n_modes)
    k_high = min(k_high, n_modes)

    total = P.sum(axis=1, keepdims=True)
    total_safe = np.where(total > 0, total, 1e-30)
    norm = P / total_safe  # per-epoch energy distribution over modes (sums to 1)

    energy_low = norm[:, :k_low].sum(axis=1)
    energy_high = norm[:, -k_high:].sum(axis=1)

    centroid = (norm * eigenvalues[None, :]).sum(axis=1)
    spread = np.sqrt(
        (norm * (eigenvalues[None, :] - centroid[:, None]) ** 2).sum(axis=1)
    )

    X = np.column_stack([energy_low, energy_high, centroid, spread])
    # replace any non-finite (from degenerate epochs) with column medians
    for j in range(X.shape[1]):
        col = X[:, j]
        fin = np.isfinite(col)
        if not fin.all():
            col[~fin] = np.median(col[fin]) if fin.any() else 0.0
            X[:, j] = col

    names = ["gsd_energy_low", "gsd_energy_high", "gsd_centroid", "gsd_spread"]
    return X, names


def build_gsd_matrix(epochs_dict, labels_df, band=None, k_low=10, k_high=10):
    """
    Build a per-epoch GSD feature matrix across all HC/MDD subjects.

    For each subject: compute wPLI connectivity (in `band` if given), derive the
    graph harmonics, project epochs onto them, then summarise the per-epoch
    energy distribution with the four GSD features.

    Parameters
    ----------
    epochs_dict : dict {subject_id: mne.Epochs}
    labels_df : pd.DataFrame  (columns include subject, label)
    band : tuple (fmin, fmax) or None
        Connectivity band. None uses objective2's default (alpha).
    k_low, k_high : int

    Returns
    -------
    X, y, groups, feature_names, subjects_used
    """
    label_map = dict(zip(labels_df["subject"], labels_df["label"]))
    fmin, fmax = (band if band is not None else (None, None))

    X_parts, y_parts, g_parts, subs = [], [], [], []
    feature_names = None

    for sid, epochs in epochs_dict.items():
        label = label_map.get(sid)
        if label not in ("HC", "MDD") or len(epochs) < 2:
            continue

        wpli, ch_names = objective2.compute_wpli(epochs, fmin=fmin, fmax=fmax)
        eigenvalues, eigenvectors = objective2.compute_graph_harmonics(wpli)
        feats = objective2.project_and_extract_features(epochs, eigenvectors, ch_names)
        per_epoch_power = feats["per_epoch_power"]  # (n_epochs, n_modes)

        Xe, names = compute_gsd_features(
            eigenvalues, per_epoch_power, k_low=k_low, k_high=k_high
        )
        if feature_names is None:
            feature_names = names

        X_parts.append(Xe)
        y_parts.append(np.full(Xe.shape[0], 1 if label == "MDD" else 0))
        g_parts.append(np.full(Xe.shape[0], sid, dtype=object))
        subs.append(sid)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    groups = np.concatenate(g_parts)
    return X, y, groups, feature_names, subs