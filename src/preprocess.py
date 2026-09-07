"""
src/preprocess.py
==================
Objective 1: preprocessing pipeline.

Steps:
  1. Bandpass filter (zero-phase FIR, firwin design)
  2. Notch filter at mains frequency (zero-phase FIR, firwin design)
  3. Bad channel detection (variance z-score) + spherical spline interpolation
  4. Average reference
"""


from __future__ import annotations
import mne
import numpy as np

import config

mne.set_log_level("WARNING")


def preprocess(raw: mne.io.RawArray, subject_id: str):
    """
    Run the full Objective 1 preprocessing pipeline on a single subject.

    Parameters
    ----------
    raw : mne.io.RawArray
        Loaded raw EEG/ECG data (not modified in place — a copy is returned)
    subject_id : str
        Used only for logging

    Returns
    -------
    raw_clean : mne.io.RawArray
    bad_channels : list of str
    """
    raw = raw.copy()

    # 1. Bandpass filter — zero-phase FIR (firwin), removes drift + high-freq noise
    raw.filter(
        l_freq=config.BANDPASS_LOW,
        h_freq=config.BANDPASS_HIGH,
        picks="eeg",
        fir_design="firwin",
        verbose=False,
    )

    # 2. Notch filter — zero-phase FIR (firwin), removes mains interference
    # (hardware line filter was OFF during recording, confirmed via .info files)
    raw.notch_filter(freqs=config.NOTCH_FREQ, picks="eeg", verbose=False)

    # 3. Bad channel detection (variance z-score) + interpolation
    eeg_data = raw.get_data(picks="eeg")
    ch_var = np.var(eeg_data, axis=1)
    z_scores = (ch_var - ch_var.mean()) / ch_var.std()
    bad_channels = [
        config.EEG_CHANNELS[i]
        for i, z in enumerate(z_scores)
        if abs(z) > config.BAD_CH_ZSCORE
    ]

    if bad_channels:
        print(f"    Bad channels: {bad_channels}")
        raw.info["bads"] = bad_channels
        raw.interpolate_bads(reset_bads=True, verbose=False)
    else:
        print("    No bad channels")

    # 4. Average reference (applied after interpolation)
    raw.set_eeg_reference("average", projection=False, verbose=False)

    return raw, bad_channels


def preprocess_all(loaded: dict) -> tuple[dict, dict]:
    """
    Run preprocessing on all loaded subjects.

    Parameters
    ----------
    loaded : dict {subject_id: (raw, df)}
        Output of src.load.load_all_subjects

    Returns
    -------
    preprocessed : dict {subject_id: mne.io.RawArray}
    bad_channel_report : dict {subject_id: list of str}
    """
    print(f"Preprocessing {len(loaded)} subjects...\n")
    preprocessed = {}
    bad_channel_report = {}

    for sid, (raw, _df) in loaded.items():
        print(f"  {sid}...")
        raw_clean, bad_chs = preprocess(raw, sid)
        preprocessed[sid] = raw_clean
        bad_channel_report[sid] = bad_chs

    print(f"\nDone. {len(preprocessed)}/{len(loaded)} subjects preprocessed.")
    return preprocessed, bad_channel_report


def print_bad_channel_report(bad_channel_report: dict) -> None:
    """Print a summary of bad channels across all subjects."""
    print(f'\n{"Subject":<10} {"Bad Channels":<30} {"N Interpolated"}')
    print("-" * 55)
    for sid, bads in bad_channel_report.items():
        bads_str = ", ".join(bads) if bads else "none"
        print(f"{sid:<10} {bads_str:<30} {len(bads)}")


if __name__ == "__main__":
    from load import load_all_subjects

    loaded = load_all_subjects()
    preprocessed, bad_channel_report = preprocess_all(loaded)
    print_bad_channel_report(bad_channel_report)
