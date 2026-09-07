"""
src/objective2.py
==================
Sensor-space graph harmonic features.

Pipeline:
  1. Compute alpha-band wPLI connectivity between electrode pairs (per subject)
  2. Build the electrode adjacency matrix from wPLI values
  3. Compute the normalised graph Laplacian and its eigendecomposition
     (the "sensor harmonics")
  4. Project each epoch's EEG signal onto the harmonics
  5. Extract per-mode features: mean power, variance, low/high mode ratio

Requires: pip install mne-connectivity
"""


from __future__ import annotations
import mne
import numpy as np
import pandas as pd
from mne_connectivity import spectral_connectivity_epochs

import config

mne.set_log_level("WARNING")


# ── Step 1: wPLI connectivity ────────────────────────────────────────────────────


def compute_wpli(epochs: mne.Epochs, fmin: float = None, fmax: float = None):
    """
    Compute alpha-band wPLI connectivity matrix from an Epochs object.

    Parameters
    ----------
    epochs : mne.Epochs
        Eyes-closed resting-state epochs (only EEG channels are used)
    fmin, fmax : float, optional
        Frequency band of interest (Hz). Defaults to config.ALPHA_BAND.

    Returns
    -------
    wpli_matrix : ndarray, shape (n_channels, n_channels)
        Symmetric wPLI connectivity matrix (zero diagonal)
    ch_names : list of str
    """
    if fmin is None:
        fmin = config.ALPHA_BAND[0]
    if fmax is None:
        fmax = config.ALPHA_BAND[1]

    eeg_epochs = epochs.copy().pick("eeg")
    ch_names = eeg_epochs.ch_names

    con = spectral_connectivity_epochs(
        eeg_epochs,
        method="wpli",
        mode="multitaper",
        sfreq=eeg_epochs.info["sfreq"],
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        verbose=False,
    )

    wpli_matrix = con.get_data(output="dense")[:, :, 0]
    wpli_matrix = wpli_matrix + wpli_matrix.T  # symmetrise (lower-triangular output)

    return wpli_matrix, ch_names


# ── Step 2 + 3: adjacency -> normalised graph Laplacian -> harmonics ──────────────


def compute_graph_harmonics(adjacency: np.ndarray):
    """
    Compute the normalised graph Laplacian and its eigendecomposition.

    Parameters
    ----------
    adjacency : ndarray, shape (n_channels, n_channels)
        Symmetric, non-negative adjacency/connectivity matrix

    Returns
    -------
    eigenvalues : ndarray, shape (n_channels,)
        Ascending order (mode 0 = smoothest / lowest "graph frequency")
    eigenvectors : ndarray, shape (n_channels, n_channels)
        Columns are the graph harmonics, ordered to match eigenvalues
    """
    W = adjacency.copy()
    np.fill_diagonal(W, 0)

    degree = W.sum(axis=1)
    degree_safe = np.where(degree > 0, degree, 1e-10)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree_safe))

    identity = np.eye(W.shape[0])
    laplacian_norm = identity - d_inv_sqrt @ W @ d_inv_sqrt

    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_norm)
    order = np.argsort(eigenvalues)

    return eigenvalues[order], eigenvectors[:, order]


# ── Step 4 + 5: project epochs onto harmonics, extract features ──────────────────


def project_and_extract_features(
    epochs: mne.Epochs,
    eigenvectors: np.ndarray,
    ch_names: list[str],
    n_low: int = None,
    n_high: int = None,
):
    """
    Project each epoch's EEG signal onto the graph harmonics and extract
    per-mode features.

    Parameters
    ----------
    epochs : mne.Epochs
    eigenvectors : ndarray, shape (n_channels, n_channels)
        Graph harmonics (columns), ascending eigenvalue order
    ch_names : list of str
        Channel order matching eigenvector rows
    n_low, n_high : int, optional
        Number of lowest/highest modes used for the low/high ratio feature.
        Defaults to config.N_LOW_MODES / config.N_HIGH_MODES.

    Returns
    -------
    dict with keys: mode_power, mode_variance, low_high_ratio, per_epoch_power
    """
    if n_low is None:
        n_low = config.N_LOW_MODES
    if n_high is None:
        n_high = config.N_HIGH_MODES

    eeg_epochs = epochs.copy().pick(ch_names)
    data = eeg_epochs.get_data()  # (n_epochs, n_channels, n_times)

    # Project: signal_modes[epoch, mode, time] = eigenvectors.T @ data[epoch, :, time]
    signal_modes = np.einsum("mc,ect->emt", eigenvectors.T, data)

    power_per_epoch_mode = np.mean(signal_modes**2, axis=2)  # (n_epochs, n_modes)

    mode_power = power_per_epoch_mode.mean(axis=0)
    mode_variance = power_per_epoch_mode.var(axis=0)

    low_power = power_per_epoch_mode[:, :n_low].mean(axis=1)
    high_power = power_per_epoch_mode[:, -n_high:].mean(axis=1)
    high_power_safe = np.where(high_power > 0, high_power, 1e-20)
    low_high_ratio = (low_power / high_power_safe).mean()

    # Per-epoch derived features (for use as classifier inputs, not just aggregates)
    # For each epoch: variance of power across modes, and low-mode/high-mode power ratio
    per_epoch_variance = power_per_epoch_mode.var(axis=1, keepdims=True)  # (n_epochs, 1)

    low_per_epoch = power_per_epoch_mode[:, :n_low].mean(axis=1)
    high_per_epoch = power_per_epoch_mode[:, -n_high:].mean(axis=1)
    high_safe = np.where(high_per_epoch > 0, high_per_epoch, 1e-20)
    per_epoch_ratio = (low_per_epoch / high_safe).reshape(-1, 1)  # (n_epochs, 1)

    return {
        "mode_power": mode_power,
        "mode_variance": mode_variance,
        "low_high_ratio": low_high_ratio,
        "per_epoch_power": power_per_epoch_mode,
        "per_epoch_variance": per_epoch_variance,
        "per_epoch_ratio": per_epoch_ratio,
    }


# ── Main: run for all subjects ─────────────────────────────────────────────────


def run_objective2(epochs_dict: dict, labels_df: pd.DataFrame):
    """
    Run the full Objective 2 pipeline for all subjects with epochs.

    Parameters
    ----------
    epochs_dict : dict {subject_id: mne.Epochs}
    labels_df : pd.DataFrame
        Output of src.labels.get_labels(), columns: subject, phq9, label

    Returns
    -------
    results : dict {subject_id: {wpli, eigenvalues, eigenvectors, ch_names, features}}
    feature_table : pd.DataFrame
        One row per subject, ready for Objective 3 classification
    """
    results = {}
    rows = []

    for sid, epochs in epochs_dict.items():
        if len(epochs) < 2:
            print(f"  ✗ {sid}: too few epochs ({len(epochs)}), skipping")
            continue

        print(f"\n── {sid} ──────────────────────────────────────")
        print(f"  Computing alpha-band wPLI ({len(epochs)} epochs)...")
        wpli_matrix, ch_names = compute_wpli(epochs)

        print("  Computing graph harmonics...")
        eigenvalues, eigenvectors = compute_graph_harmonics(wpli_matrix)

        print("  Projecting epochs onto harmonics & extracting features...")
        features = project_and_extract_features(epochs, eigenvectors, ch_names)

        results[sid] = {
            "wpli": wpli_matrix,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "ch_names": ch_names,
            "features": features,
        }

        row = {"subject": sid}
        for i, p in enumerate(features["mode_power"]):
            row[f"mode{i}_power"] = p
        for i, v in enumerate(features["mode_variance"]):
            row[f"mode{i}_var"] = v
        row["low_high_ratio"] = features["low_high_ratio"]
        rows.append(row)

        print(f"  ✓ {sid} done")

    feature_table = pd.DataFrame(rows)

    label_map = dict(zip(labels_df["subject"], labels_df["label"]))
    feature_table["label"] = feature_table["subject"].map(label_map)

    return results, feature_table
