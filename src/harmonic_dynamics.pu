"""
src/harmonic_dynamics.py
========================
Two harmonic features capturing information the static per-mode powers discard:

  1. Harmonic energy entropy (diversity)
     Shannon entropy of the per-epoch energy distribution across harmonic modes,
     normalised to [0, 1]. High = energy spread across many modes; low = energy
     concentrated in few. Pre-specified hypothesis: reduced diversity in MDD.

  2. Temporal mode switching (dynamics)
     Within each epoch, at every time sample, identify the dominant harmonic mode
     (the one carrying the most instantaneous power), then measure:
       - switch_rate: fraction of consecutive samples where the dominant mode
         changes (how fast the brain reconfigures across the harmonic basis)
       - mean_dwell: average run length the dominant mode persists before
         switching (in samples)
     Pre-specified hypothesis: slower switching / longer dwell in MDD
     (reduced metastability). This is the one genuinely dynamic feature in the
     project -- all prior features are static (epoch-averaged) summaries.

Both are reported regardless of outcome.

Note: the time-resolved mode projection is recomputed here because
objective2.project_and_extract_features returns only epoch-averaged summaries,
not the per-timepoint mode signal.
"""

from __future__ import annotations

import numpy as np

from src import objective2


# ── Feature 1: harmonic energy entropy ─────────────────────────────────────────

def compute_energy_entropy(per_epoch_power):
    """
    Shannon entropy of each epoch's energy distribution across modes,
    normalised to [0, 1] by log(n_modes).

    Parameters
    ----------
    per_epoch_power : ndarray (n_epochs, n_modes)

    Returns
    -------
    entropy : ndarray (n_epochs,)
    """
    P = np.asarray(per_epoch_power, dtype=float)
    total = P.sum(axis=1, keepdims=True)
    total_safe = np.where(total > 0, total, 1e-30)
    Pn = P / total_safe
    Pn_safe = np.where(Pn > 0, Pn, 1e-30)
    ent = -(Pn * np.log(Pn_safe)).sum(axis=1)
    return ent / np.log(P.shape[1])


# ── Feature 2: temporal mode switching ─────────────────────────────────────────

def compute_mode_switching(signal_modes, n_top_modes=20):
    """
    Switching rate and mean dwell time of the dominant harmonic mode over time,
    per epoch.

    Parameters
    ----------
    signal_modes : ndarray (n_epochs, n_modes, n_times)
        Time-resolved projection of the signal onto the harmonic modes.
    n_top_modes : int
        Restrict the "dominant mode" competition to the lowest n_top_modes
        (high modes are noise-dominated).

    Returns
    -------
    switch_rate : ndarray (n_epochs,)
    mean_dwell : ndarray (n_epochs,)
    """
    S = np.asarray(signal_modes, dtype=float)
    n_epochs, n_modes, n_times = S.shape
    k = min(n_top_modes, n_modes)

    power = S[:, :k, :] ** 2
    dominant = power.argmax(axis=1)  # (n_epochs, n_times)

    switch_rate = np.empty(n_epochs)
    mean_dwell = np.empty(n_epochs)

    for e in range(n_epochs):
        seq = dominant[e]
        if n_times < 2:
            switch_rate[e] = 0.0
            mean_dwell[e] = float(n_times)
            continue
        switches = int(np.sum(seq[1:] != seq[:-1]))
        switch_rate[e] = switches / (n_times - 1)

        # mean run length (dwell)
        run_lengths = []
        cur = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i - 1]:
                cur += 1
            else:
                run_lengths.append(cur)
                cur = 1
        run_lengths.append(cur)
        mean_dwell[e] = float(np.mean(run_lengths))

    return switch_rate, mean_dwell


# ── Per-subject extraction (recomputes time-resolved projection) ───────────────

def _project_timecourse(epochs, eigenvectors, ch_names):
    """Return signal_modes (n_epochs, n_modes, n_times) and per_epoch_power."""
    eeg = epochs.copy().pick(ch_names)
    data = eeg.get_data()  # (n_epochs, n_channels, n_times)
    # signal_modes[e, m, t] = sum_c eigenvectors[c, m] * data[e, c, t]
    signal_modes = np.einsum("mc,ect->emt", eigenvectors.T, data)
    per_epoch_power = np.mean(signal_modes ** 2, axis=2)  # (n_epochs, n_modes)
    return signal_modes, per_epoch_power


def build_dynamics_matrix(epochs_dict, labels_df, band=None, n_top_modes=20):
    """
    Build a per-epoch feature matrix of the three dynamics features
    [energy_entropy, switch_rate, mean_dwell] across all HC/MDD subjects.

    Returns
    -------
    X, y, groups, feature_names, subjects_used
    """
    label_map = dict(zip(labels_df["subject"], labels_df["label"]))
    fmin, fmax = (band if band is not None else (None, None))

    X_parts, y_parts, g_parts, subs = [], [], [], []
    feature_names = ["harm_energy_entropy", "harm_switch_rate", "harm_mean_dwell"]

    for sid, epochs in epochs_dict.items():
        label = label_map.get(sid)
        if label not in ("HC", "MDD") or len(epochs) < 2:
            continue

        wpli, ch_names = objective2.compute_wpli(epochs, fmin=fmin, fmax=fmax)
        _, eigenvectors = objective2.compute_graph_harmonics(wpli)

        signal_modes, per_epoch_power = _project_timecourse(epochs, eigenvectors, ch_names)

        ent = compute_energy_entropy(per_epoch_power)
        sw, dw = compute_mode_switching(signal_modes, n_top_modes=n_top_modes)

        Xe = np.column_stack([ent, sw, dw])
        # clean any non-finite
        for j in range(Xe.shape[1]):
            col = Xe[:, j]; fin = np.isfinite(col)
            if not fin.all():
                col[~fin] = np.median(col[fin]) if fin.any() else 0.0
                Xe[:, j] = col

        X_parts.append(Xe)
        y_parts.append(np.full(Xe.shape[0], 1 if label == "MDD" else 0))
        g_parts.append(np.full(Xe.shape[0], sid, dtype=object))
        subs.append(sid)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    groups = np.concatenate(g_parts)
    return X, y, groups, feature_names, subs