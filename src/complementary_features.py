"""
src/complementary_features.py
=============================
Features designed to be COMPLEMENTARY to the graph-harmonic approach --
capturing information the harmonics structurally discard.

Rationale
---------
Graph harmonics encode the *spatial* structure of band-limited connectivity,
collapsing each epoch to average mode power. They are largely blind to:
  - temporal dynamics (how the signal evolves within an epoch)
  - the aperiodic (1/f) spectral background
  - hemispheric lateralisation

Three complementary feature families address these orthogonal axes:

  1. Temporal complexity (per channel): sample entropy and spectral entropy.
     Captures signal irregularity/dynamics the harmonics ignore. Reduced
     complexity is a reported MDD marker.

  2. Aperiodic 1/f slope (per channel): the exponent of the aperiodic
     component of the power spectrum (via FOOOF/specparam). Reflects
     excitation/inhibition balance; increasingly studied in depression.

  3. Frontal alpha asymmetry: right-minus-left alpha log-power over frontal
     electrode pairs (F4-F3, F8-F7). The most-cited EEG marker in depression.

All are computed per epoch then, for classification, can be averaged per
subject or used per epoch to match the existing pipeline.
"""

from __future__ import annotations

import numpy as np
import mne

try:
    import antropy
    _HAVE_ANTROPY = True
except ImportError:
    _HAVE_ANTROPY = False

mne.set_log_level("WARNING")

ALPHA_BAND = (8.0, 13.0)

# Frontal electrode pairs for asymmetry (right, left). Present in the
# Neuroelectrics 19-channel montage.
FRONTAL_PAIRS = [("F4", "F3"), ("F8", "F7")]


# ── 1. Temporal complexity ─────────────────────────────────────────────────────

def compute_complexity_epochs(epochs):
    """
    Per-epoch, per-channel sample entropy and spectral entropy.

    Returns
    -------
    X : ndarray (n_epochs, n_channels * 2)
        [ch0_sampen, ch0_specen, ch1_sampen, ...]
    feature_names : list of str
    """
    if not _HAVE_ANTROPY:
        raise ImportError("antropy required: pip install antropy")

    eeg = epochs.copy().pick("eeg")
    data = eeg.get_data()  # (n_epochs, n_channels, n_times)
    ch_names = eeg.ch_names
    sfreq = eeg.info["sfreq"]

    n_epochs, n_channels, _ = data.shape
    cols, names = [], []

    for ch_idx, ch in enumerate(ch_names):
        sampen = np.empty(n_epochs)
        specen = np.empty(n_epochs)
        for e in range(n_epochs):
            sig = data[e, ch_idx]
            # Sample entropy: irregularity of the time series
            sampen[e] = antropy.sample_entropy(sig)
            # Spectral entropy: flatness of the power spectrum
            specen[e] = antropy.spectral_entropy(
                sig, sf=sfreq, method="welch", normalize=True
            )
        cols.append(sampen); names.append(f"{ch}_sampen")
        cols.append(specen); names.append(f"{ch}_specen")

    X = np.column_stack(cols)
    # Replace any non-finite (rare, from flat segments) with column medians
    X = _clean_nonfinite(X)
    return X, names


# ── 2. Aperiodic 1/f slope ──────────────────────────────────────────────────────

def compute_aperiodic_slope_epochs(epochs, fmin=2.0, fmax=40.0):
    """
    Per-epoch, per-channel aperiodic exponent (1/f slope) via FOOOF.

    Returns
    -------
    X : ndarray (n_epochs, n_channels)
    feature_names : list of str
    """
    from fooof import FOOOF

    eeg = epochs.copy().pick("eeg")
    ch_names = eeg.ch_names
    sfreq = eeg.info["sfreq"]

    # Compute per-epoch PSDs
    spectrum = eeg.compute_psd(method="welch", fmin=fmin, fmax=fmax, verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)  # (n_epochs, n_ch, n_freqs)

    n_epochs, n_channels, _ = psds.shape
    exponents = np.empty((n_epochs, n_channels))

    for e in range(n_epochs):
        for c in range(n_channels):
            fm = FOOOF(max_n_peaks=6, verbose=False)
            try:
                fm.fit(freqs, psds[e, c], [fmin, fmax])
                exponents[e, c] = fm.get_params("aperiodic_params", "exponent")
            except Exception:
                exponents[e, c] = np.nan

    exponents = _clean_nonfinite(exponents)
    names = [f"{ch}_1f_slope" for ch in ch_names]
    return exponents, names


# ── 3. Frontal alpha asymmetry ──────────────────────────────────────────────────

def compute_frontal_asymmetry_epochs(epochs, pairs=None):
    """
    Per-epoch frontal alpha asymmetry: log(right alpha) - log(left alpha)
    for each frontal pair. Positive = relatively less left-frontal alpha.

    Returns
    -------
    X : ndarray (n_epochs, n_pairs)
    feature_names : list of str
    """
    if pairs is None:
        pairs = FRONTAL_PAIRS

    eeg = epochs.copy().pick("eeg")
    ch_names = eeg.ch_names
    sfreq = eeg.info["sfreq"]

    spectrum = eeg.compute_psd(
        method="welch", fmin=ALPHA_BAND[0], fmax=ALPHA_BAND[1], verbose=False
    )
    psds, freqs = spectrum.get_data(return_freqs=True)  # (n_epochs, n_ch, n_freqs)
    alpha_power = psds.mean(axis=2)  # (n_epochs, n_ch), mean over alpha bins

    cols, names = [], []
    for right, left in pairs:
        if right not in ch_names or left not in ch_names:
            continue
        ri, li = ch_names.index(right), ch_names.index(left)
        # log-power asymmetry, standard in the asymmetry literature
        asym = np.log(alpha_power[:, ri] + 1e-30) - np.log(alpha_power[:, li] + 1e-30)
        cols.append(asym)
        names.append(f"asym_{right}_{left}")

    X = np.column_stack(cols) if cols else np.empty((len(alpha_power), 0))
    return X, names


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _clean_nonfinite(X):
    """Replace NaN/inf with column medians (computed over finite values)."""
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        finite = np.isfinite(col)
        if finite.all():
            continue
        med = np.median(col[finite]) if finite.any() else 0.0
        col[~finite] = med
        X[:, j] = col
    return X


# ── Assemble per-subject matrices ──────────────────────────────────────────────

def build_complementary_matrix(epochs_dict, labels_df, feature_fn):
    """
    Apply a per-epoch feature function across all HC/MDD subjects and stack
    into (X, y, groups) form for the existing classification pipeline.

    Parameters
    ----------
    epochs_dict : dict {subject_id: mne.Epochs}
    labels_df : pd.DataFrame
    feature_fn : callable(epochs) -> (X_epochs, feature_names)
        One of compute_complexity_epochs / compute_aperiodic_slope_epochs /
        compute_frontal_asymmetry_epochs

    Returns
    -------
    X, y, groups, feature_names, subjects_used
    """
    label_map = dict(zip(labels_df["subject"], labels_df["label"]))

    X_parts, y_parts, g_parts, subjects_used = [], [], [], []
    feature_names = None

    for sid, epochs in epochs_dict.items():
        label = label_map.get(sid)
        if label not in ("HC", "MDD") or len(epochs) < 2:
            continue

        Xe, names = feature_fn(epochs)
        if feature_names is None:
            feature_names = names
        if Xe.shape[1] == 0:
            continue

        X_parts.append(Xe)
        y_parts.append(np.full(Xe.shape[0], 1 if label == "MDD" else 0))
        g_parts.append(np.full(Xe.shape[0], sid, dtype=object))
        subjects_used.append(sid)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    groups = np.concatenate(g_parts)
    return X, y, groups, feature_names, subjects_used