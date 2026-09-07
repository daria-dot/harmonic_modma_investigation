"""
src/filter_comparison.py
=========================
Compares the current FIR (firwin) preprocessing filters against an
equivalent IIR (Butterworth) design, to evaluate whether a lower-latency
filtering approach could be substituted without materially changing
downstream results.

Motivation (per lecturer feedback): the current FIR filters are long
(1651 and 3301 samples -- 3.3s and 6.6s respectively), which is fine for
offline, non-causal, zero-phase filtering but would introduce unacceptable
latency in a real-time/online BCI system. IIR filters achieve similar
frequency selectivity with far shorter effective length, at the cost of a
non-linear phase response (unless applied non-causally via filtfilt, which
this script also does for a fair offline comparison).

Usage:
    python run_filter_comparison.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import signal

import config

mne.set_log_level("WARNING")


def design_iir_bandpass(order: int = 4):
    """Design a Butterworth bandpass filter matching config.BANDPASS_LOW/HIGH."""
    nyquist = config.SFREQ / 2
    low = config.BANDPASS_LOW / nyquist
    high = config.BANDPASS_HIGH / nyquist
    sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
    return sos


def design_iir_notch(order: int = 2, width_hz: float = 2.0):
    """Design a Butterworth bandstop (notch) filter matching config.NOTCH_FREQ."""
    nyquist = config.SFREQ / 2
    low = (config.NOTCH_FREQ - width_hz / 2) / nyquist
    high = (config.NOTCH_FREQ + width_hz / 2) / nyquist
    sos = signal.butter(order, [low, high], btype="bandstop", output="sos")
    return sos


def apply_fir_pipeline(raw: mne.io.RawArray) -> mne.io.RawArray:
    """Apply the project's current FIR bandpass + notch filtering."""
    raw = raw.copy()
    raw.filter(
        l_freq=config.BANDPASS_LOW,
        h_freq=config.BANDPASS_HIGH,
        picks="eeg",
        fir_design="firwin",
        verbose=False,
    )
    raw.notch_filter(freqs=config.NOTCH_FREQ, picks="eeg", verbose=False)
    return raw


def apply_iir_pipeline(
    raw: mne.io.RawArray, bandpass_order: int = 4, notch_order: int = 2
) -> mne.io.RawArray:
    """
    Apply an equivalent IIR (Butterworth) bandpass + notch filter, using
    zero-phase (filtfilt-style, via scipy.signal.sosfiltfilt) application
    for a fair offline comparison against the FIR zero-phase result.

    Note: this still removes phase distortion (since it's applied
    non-causally here, matching the FIR approach), but the *effective
    filter length* / settling time is far shorter than the FIR kernel,
    which is the property that matters for real-time feasibility -- a
    causal (forward-only) IIR version would have low latency directly.
    """
    raw = raw.copy()
    bp_sos = design_iir_bandpass(order=bandpass_order)
    notch_sos = design_iir_notch(order=notch_order)

    data = raw.get_data(picks="eeg")
    data = signal.sosfiltfilt(bp_sos, data, axis=-1)
    data = signal.sosfiltfilt(notch_sos, data, axis=-1)

    eeg_idx = mne.pick_types(raw.info, eeg=True)
    raw._data[eeg_idx, :] = data

    return raw


def compare_filters(raw: mne.io.RawArray, subject_id: str = "subject"):
    """
    Run both FIR and IIR filtering pipelines on the same raw data and
    compare them quantitatively and visually.

    Returns
    -------
    dict with keys: raw_fir, raw_iir, correlation_per_channel, max_abs_diff_uv
    """
    raw_fir = apply_fir_pipeline(raw)
    raw_iir = apply_iir_pipeline(raw)

    data_fir = raw_fir.get_data(picks="eeg")
    data_iir = raw_iir.get_data(picks="eeg")

    correlations = np.array(
        [np.corrcoef(data_fir[i], data_iir[i])[0, 1] for i in range(data_fir.shape[0])]
    )
    max_abs_diff_uv = np.max(np.abs(data_fir - data_iir)) * 1e6  # V -> uV

    print(f"\n── FIR vs IIR comparison: {subject_id} ──")
    print(f"  Per-channel correlation: min={correlations.min():.4f}, "
          f"mean={correlations.mean():.4f}, max={correlations.max():.4f}")
    print(f"  Max absolute difference: {max_abs_diff_uv:.2f} µV")

    return {
        "raw_fir": raw_fir,
        "raw_iir": raw_iir,
        "correlation_per_channel": correlations,
        "max_abs_diff_uv": max_abs_diff_uv,
    }


def plot_filter_comparison(
    raw_fir, raw_iir, subject_id,
    raw_unfiltered=None, channel=None,
    start_s=0.0, duration_s=5.0, save=True,
):
    if channel is None:
        channel = config.EEG_CHANNELS[0]

    sfreq = raw_fir.info["sfreq"]
    start = int(start_s * sfreq)
    n_samples = int(duration_s * sfreq)
    sl = slice(start, start + n_samples)
    t = np.arange(n_samples) / sfreq + start_s

    fir_segment = raw_fir.get_data(picks=[channel])[0, sl] * 1e6
    iir_segment = raw_iir.get_data(picks=[channel])[0, sl] * 1e6

    fig, ax = plt.subplots(figsize=(11, 4.5))

    if raw_unfiltered is not None:
        raw_segment = raw_unfiltered.get_data(picks=[channel])[0, sl] * 1e6
        raw_segment = raw_segment - raw_segment.mean()
        ax.plot(t, raw_segment, label="Raw (unfiltered, mean-centred)",
                linewidth=0.8, color="lightgray", zorder=1)

    ax.plot(t, fir_segment, label="FIR (firwin)", linewidth=1.3, zorder=3)
    ax.plot(t, iir_segment, label="IIR (Butterworth)", linewidth=1.3,
            linestyle="--", zorder=2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"{subject_id} — {channel}: raw vs FIR vs IIR ({start_s:.0f}-{start_s+duration_s:.0f}s)")
    ax.legend()
    fig.tight_layout()

    if save:
        out_path = config.FIGURES_DIR / f"{subject_id}_filter_comparison_{channel}.png"
        fig.savefig(out_path, dpi=150)
        print(f"  Saved: {out_path}")

    return fig
def plot_frequency_response(bandpass_order: int = 4, notch_order: int = 2, save: bool = True):
    """Plot the frequency response of the IIR filters alongside the FIR design."""
    sfreq = config.SFREQ

    bp_sos = design_iir_bandpass(order=bandpass_order)
    w_iir_bp, h_iir_bp = signal.sosfreqz(bp_sos, worN=2048, fs=sfreq)

    notch_sos = design_iir_notch(order=notch_order)
    w_iir_notch, h_iir_notch = signal.sosfreqz(notch_sos, worN=2048, fs=sfreq)

    fir_coeffs = mne.filter.create_filter(
        data=np.zeros((1, sfreq * 10)),
        sfreq=sfreq,
        l_freq=config.BANDPASS_LOW,
        h_freq=config.BANDPASS_HIGH,
        method="fir",
        fir_design="firwin",
        verbose=False,
    )
    w_fir, h_fir = signal.freqz(fir_coeffs, worN=2048, fs=sfreq)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(w_fir, 20 * np.log10(np.abs(h_fir) + 1e-12), label="FIR (firwin)")
    axes[0].plot(w_iir_bp, 20 * np.log10(np.abs(h_iir_bp) + 1e-12), label=f"IIR (Butterworth, order {bandpass_order})")
    axes[0].set_xlim(0, 60)
    axes[0].set_ylim(-80, 5)
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].set_title(f"Bandpass {config.BANDPASS_LOW}-{config.BANDPASS_HIGH} Hz")
    axes[0].legend()
    axes[0].axvspan(config.BANDPASS_LOW, config.BANDPASS_HIGH, alpha=0.1, color="green")

    axes[1].plot(w_iir_notch, 20 * np.log10(np.abs(h_iir_notch) + 1e-12),
                 label=f"IIR notch (Butterworth, order {notch_order})", color="C1")
    axes[1].set_xlim(40, 60)
    axes[1].set_ylim(-80, 5)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude (dB)")
    axes[1].set_title(f"Notch at {config.NOTCH_FREQ} Hz")
    axes[1].legend()
    axes[1].axvline(config.NOTCH_FREQ, color="red", linestyle=":", alpha=0.5)

    fig.tight_layout()

    if save:
        out_path = config.FIGURES_DIR / "fir_vs_iir_frequency_response.png"
        fig.savefig(out_path, dpi=150)
        print(f"  Saved: {out_path}")

    return fig