"""
src/modma_pipeline.py
=====================
Preprocessing and epoching for the MODMA 128-channel resting-state dataset.

Mirrors the Objective 1 pipeline applied to the Neuroelectrics data, with
three necessary adaptations:

  1. Channel-agnostic bad-channel detection. The original preprocess.py
     indexes into a hardcoded 19-channel list; here channel names are read
     from the Raw object so the same logic works for 128 channels.

  2. No trigger channel. MODMA resting recordings are continuous with no
     events, so epochs are fixed-length across the whole recording rather
     than locked to an eyes-closed trigger. (Consequence: the eyes-open vs
     eyes-closed contrast is NOT possible on MODMA.)

  3. Different sampling rate (250 Hz vs 500 Hz) and units (uV vs nV),
     both handled in modma_load.py and via explicit parameters here.

Mains frequency in China is 50 Hz, so the same notch frequency applies.
"""

from __future__ import annotations

import gc
from pathlib import Path

import mne
import numpy as np
import pandas as pd

import modma_load as ml

mne.set_log_level("WARNING")


# ── Defaults (kept explicit rather than importing the Neuroelectrics config) ────

BANDPASS_LOW = 1.0            # Hz
BANDPASS_HIGH = 40.0          # Hz
NOTCH_FREQ = 50.0             # Hz (mains)
BAD_CH_ZSCORE = 3.0           # SD threshold for variance-based bad channels
EPOCH_LEN_S = 2.0             # seconds
REJECT_THRESHOLD_EEG = 150e-6  # volts, peak-to-peak

# Electrodes on the outer rim of the GSN-HydroCel net sit on the face/neck and
# are dominated by muscle and eye artefact. Excluding them is standard practice
# for EGI 128 resting analyses.
OUTER_RIM_CHANNELS = [
    "E127", "E126", "E17", "E21", "E14", "E25", "E8", "E128", "E125",
    "E43", "E48", "E119", "E120", "E49", "E113", "E56", "E63", "E68",
    "E73", "E81", "E88", "E94", "E99", "E107",
]


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess_modma(
    raw: mne.io.RawArray,
    subject_id: str,
    bandpass_low: float = BANDPASS_LOW,
    bandpass_high: float = BANDPASS_HIGH,
    notch_freq: float = NOTCH_FREQ,
    bad_ch_zscore: float = BAD_CH_ZSCORE,
    drop_outer_rim: bool = True,
):
    """
    Preprocess one MODMA recording.

    Steps (matching the Neuroelectrics pipeline):
      1. Optionally drop outer-rim (face/neck) electrodes
      2. Band-pass filter (zero-phase FIR, firwin)
      3. Notch filter at mains frequency (zero-phase FIR, firwin)
      4. Variance-based bad-channel detection + spherical spline interpolation
      5. Average reference

    Returns
    -------
    raw_clean : mne.io.RawArray
    bad_channels : list of str
    """
    raw = raw.copy()

    # 1. Drop outer-rim electrodes (heavy muscle/ocular contamination)
    if drop_outer_rim:
        to_drop = [ch for ch in OUTER_RIM_CHANNELS if ch in raw.ch_names]
        if to_drop:
            raw.drop_channels(to_drop)

    # 2. Band-pass (zero-phase FIR)
    raw.filter(
        l_freq=bandpass_low,
        h_freq=bandpass_high,
        picks="eeg",
        fir_design="firwin",
        verbose=False,
    )

    # 3. Notch at mains
    raw.notch_filter(freqs=notch_freq, picks="eeg", verbose=False)

    # 4. Bad-channel detection -- channel names read from raw, not hardcoded
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    eeg_ch_names = [raw.ch_names[i] for i in eeg_picks]

    eeg_data = raw.get_data(picks="eeg")
    ch_var = np.var(eeg_data, axis=1)
    # Guard against zero-variance (flat) channels breaking the z-score
    if ch_var.std() == 0:
        bad_channels = []
    else:
        z_scores = (ch_var - ch_var.mean()) / ch_var.std()
        bad_channels = [
            eeg_ch_names[i] for i, z in enumerate(z_scores) if abs(z) > bad_ch_zscore
        ]

    if bad_channels:
        raw.info["bads"] = bad_channels
        raw.interpolate_bads(reset_bads=True, verbose=False)

    # 5. Average reference
    raw.set_eeg_reference("average", projection=False, verbose=False)

    return raw, bad_channels


# ── Epoching ───────────────────────────────────────────────────────────────────

def extract_modma_epochs(
    raw: mne.io.RawArray,
    epoch_len_s: float = EPOCH_LEN_S,
    reject_threshold: float = REJECT_THRESHOLD_EEG,
    crop_start_s: float = 5.0,
    crop_end_s: float | None = None,
):
    """
    Segment a continuous MODMA recording into fixed-length epochs.

    Unlike the Neuroelectrics data there are no triggers, so the whole
    recording is segmented. The first few seconds are dropped by default,
    as amplifier settling and participant movement at recording onset are
    a common source of large artefacts.

    Parameters
    ----------
    crop_start_s : float
        Seconds to discard from the start of the recording.
    crop_end_s : float or None
        If given, crop the recording to end at this time (seconds).

    Returns
    -------
    epochs : mne.Epochs or None
    """
    tmax = raw.times[-1] if crop_end_s is None else min(crop_end_s, raw.times[-1])
    if tmax <= crop_start_s:
        return None

    raw_cropped = raw.copy().crop(tmin=crop_start_s, tmax=tmax)

    epochs = mne.make_fixed_length_epochs(
        raw_cropped, duration=epoch_len_s, overlap=0, preload=True, verbose=False
    )

    n_before = len(epochs)
    epochs.drop_bad(reject={"eeg": reject_threshold}, verbose=False)
    n_after = len(epochs)

    return epochs, n_before, n_after


# ── Full run (memory-safe, one subject at a time) ──────────────────────────────

def run_modma_objective1(
    data_dir: Path,
    labels_df: pd.DataFrame,
    epoch_len_s: float = EPOCH_LEN_S,
    reject_threshold: float = REJECT_THRESHOLD_EEG,
    drop_outer_rim: bool = True,
    subjects: list[str] | None = None,
):
    """
    Load, preprocess, and epoch all MODMA subjects one at a time.

    Processing one subject per iteration and freeing the raw object before
    the next keeps peak memory to roughly a single recording, which matters
    because 53 x ~74 MB recordings will not fit in a standard Colab session.

    Parameters
    ----------
    data_dir : Path
        Folder of MODMA .mat files
    labels_df : pd.DataFrame
        Output of modma_load.load_modma_labels
    subjects : list of str, optional
        Restrict to these subject IDs (useful for testing on a few subjects)

    Returns
    -------
    epochs_dict : dict {subject_id: mne.Epochs}
    report_df : pd.DataFrame  (per-subject QC summary)
    """
    file_map = ml.discover_modma_subjects(data_dir)
    label_map = dict(zip(labels_df["subject"], labels_df["label"]))

    target_ids = subjects if subjects is not None else sorted(file_map)

    epochs_dict = {}
    rows = []

    print(f"Processing {len(target_ids)} MODMA subjects one at a time...\n")

    for sid in target_ids:
        if sid not in file_map:
            print(f"  ! {sid}: no .mat file found, skipping")
            continue

        label = label_map.get(sid, "UNKNOWN")
        print(f"  {sid} ({label})...", end=" ")

        try:
            raw = ml.load_modma_subject(file_map[sid])
            raw_clean, bad_chs = preprocess_modma(
                raw, sid, drop_outer_rim=drop_outer_rim
            )

            result = extract_modma_epochs(
                raw_clean,
                epoch_len_s=epoch_len_s,
                reject_threshold=reject_threshold,
            )
            if result is None:
                print("no epochs")
                continue

            epochs, n_before, n_after = result
            if n_after > 0:
                epochs_dict[sid] = epochs

            print(f"{n_after}/{n_before} epochs kept, {len(bad_chs)} bad ch")

            rows.append({
                "subject": sid,
                "label": label,
                "n_bad_channels": len(bad_chs),
                "bad_channels": ", ".join(bad_chs) if bad_chs else "none",
                "n_epochs_before": n_before,
                "n_epochs_kept": n_after,
                "pct_rejected": 100 * (1 - n_after / n_before) if n_before else np.nan,
            })

        except Exception as exc:  # keep going if one file is malformed
            print(f"FAILED: {exc}")
            rows.append({
                "subject": sid, "label": label, "n_bad_channels": np.nan,
                "bad_channels": f"ERROR: {exc}", "n_epochs_before": 0,
                "n_epochs_kept": 0, "pct_rejected": np.nan,
            })

        finally:
            # Free the large objects before loading the next subject
            raw = None
            raw_clean = None
            gc.collect()

    report_df = pd.DataFrame(rows)
    print(f"\nDone. {len(epochs_dict)}/{len(target_ids)} subjects with usable epochs.")
    return epochs_dict, report_df