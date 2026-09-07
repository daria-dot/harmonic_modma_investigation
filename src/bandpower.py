"""
src/bandpower.py
================
Traditional band-power feature extraction, as a comparison baseline against
the graph-harmonic features. For each epoch, computes the power in each
frequency band at each channel via Welch's method (PSD integrated over band).

This is the conventional EEG feature for MDD classification (cf. the standard
delta/theta/alpha/beta power markers), providing a benchmark against which the
graph-harmonic approach can be compared using the identical classification and
validation pipeline.
"""

from __future__ import annotations

import numpy as np
import mne

mne.set_log_level("WARNING")


def compute_band_power_epochs(epochs, bands):
    """
    Compute per-epoch, per-channel band power for each frequency band.

    Parameters
    ----------
    epochs : mne.Epochs
        Eyes-closed resting epochs (EEG channels used)
    bands : dict {band_name: (fmin, fmax)}

    Returns
    -------
    X : ndarray, shape (n_epochs, n_channels * n_bands)
        Per-epoch band-power features. Column order is
        [ch0_band0, ch0_band1, ..., ch1_band0, ...] following np.stack.
    feature_names : list of str
        e.g. ["Oz_theta", "Oz_alpha", "Oz_beta", "PO3_theta", ...]
    """
    eeg = epochs.copy().pick("eeg")
    ch_names = eeg.ch_names
    sfreq = eeg.info["sfreq"]

    # PSD via multitaper (robust for short epochs); one PSD per epoch per channel
    spectrum = eeg.compute_psd(method="multitaper", fmin=1.0, fmax=40.0, verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)  # (n_epochs, n_channels, n_freqs)

    band_names = list(bands.keys())
    n_epochs, n_channels, _ = psds.shape

    # Integrate PSD over each band -> band power
    feature_cols = []
    feature_names = []
    for ch_idx, ch in enumerate(ch_names):
        for band_name in band_names:
            fmin, fmax = bands[band_name]
            mask = (freqs >= fmin) & (freqs <= fmax)
            # Mean power in band for this channel, per epoch
            band_power = psds[:, ch_idx, mask].mean(axis=1)  # (n_epochs,)
            feature_cols.append(band_power)
            feature_names.append(f"{ch}_{band_name}")

    X = np.stack(feature_cols, axis=1)  # (n_epochs, n_channels*n_bands)
    return X, feature_names


def build_bandpower_feature_matrix(epochs_dict, labels_df, bands):
    """
    Build a per-epoch band-power feature matrix across all subjects,
    mirroring objective3.build_epoch_feature_matrix so the same classifier
    and CV pipeline can be applied.

    Returns
    -------
    X, y, groups, feature_names, subjects_used
    """
    label_map = dict(zip(labels_df["subject"], labels_df["label"]))

    X_parts, y_parts, group_parts = [], [], []
    subjects_used = []
    feature_names = None

    for sid, epochs in epochs_dict.items():
        label = label_map.get(sid)
        if label not in ("HC", "MDD"):
            continue
        if len(epochs) < 2:
            continue

        Xb, fnames = compute_band_power_epochs(epochs, bands)
        if feature_names is None:
            feature_names = fnames

        X_parts.append(Xb)
        y_parts.append(np.full(Xb.shape[0], 1 if label == "MDD" else 0))
        group_parts.append(np.full(Xb.shape[0], sid, dtype=object))
        subjects_used.append(sid)

    if not X_parts:
        raise ValueError("No HC/MDD subjects found.")

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    groups = np.concatenate(group_parts)
    return X, y, groups, feature_names, subjects_used


def build_alpha_beta_ratio_matrix(epochs_dict, labels_df):
    """
    Per-epoch alpha/beta power ratio at each channel.
    One feature per channel (n_channels features).
    """
    import numpy as np
    bands = {"alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
    label_map = dict(zip(labels_df["subject"], labels_df["label"]))

    X_parts, y_parts, group_parts, subjects_used = [], [], [], []
    feature_names = None

    for sid, epochs in epochs_dict.items():
        label = label_map.get(sid)
        if label not in ("HC", "MDD") or len(epochs) < 2:
            continue

        Xb, fnames = compute_band_power_epochs(epochs, bands)
        # fnames alternate ch_alpha, ch_beta per channel
        alpha_idx = [i for i, n in enumerate(fnames) if n.endswith("_alpha")]
        beta_idx = [i for i, n in enumerate(fnames) if n.endswith("_beta")]

        alpha_pow = Xb[:, alpha_idx]
        beta_pow = Xb[:, beta_idx]
        beta_safe = np.where(beta_pow > 0, beta_pow, 1e-20)
        ratio = alpha_pow / beta_safe   # (n_epochs, n_channels)

        if feature_names is None:
            ch_names = [fnames[i].replace("_alpha", "") for i in alpha_idx]
            feature_names = [f"{ch}_alpha_beta_ratio" for ch in ch_names]

        X_parts.append(ratio)
        y_parts.append(np.full(ratio.shape[0], 1 if label == "MDD" else 0))
        group_parts.append(np.full(ratio.shape[0], sid, dtype=object))
        subjects_used.append(sid)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    groups = np.concatenate(group_parts)
    return X, y, groups, feature_names, subjects_used