"""
src/modma_load.py
=================
Loader for the MODMA 128-channel resting-state EEG dataset
(EEG_128channels_resting_lanzhou_2015, Lanzhou University).

Format notes (confirmed by direct inspection):
  - One .mat file per subject, ~74 MB each
  - Each contains three keys:
      * <dynamic name>  : (129, n_samples) float array -- 128 EEG + Cz reference
      * samplingRate    : (1, 1) -- 250 Hz
      * Impedances_0    : (129, 1) -- electrode impedances in kOhm
  - The data key name is dynamic (encodes subject ID and date), so it is
    located by elimination rather than hardcoded.
  - Units are microvolts; MNE expects volts, so data is scaled by 1e-6.
  - EGI GSN-HydroCel-129 montage (E1..E128 + Cz).
  - Continuous 5-minute resting recording with NO event/trigger channel,
    so epochs are fixed-length rather than trigger-locked (unlike the
    Neuroelectrics dataset).

Subject IDs in the spreadsheet are integers (e.g. 2010002) but filenames use
an 8-digit zero-padded form (e.g. 02010002rest...), so IDs are normalised.
"""

from __future__ import annotations

import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy.io

mne.set_log_level("WARNING")


# ── Constants ──────────────────────────────────────────────────────────────────

MODMA_SFREQ = 250.0          # Hz, confirmed from samplingRate field
MODMA_N_CHANNELS = 129       # 128 EEG + Cz reference
MODMA_MONTAGE = "GSN-HydroCel-129"
MODMA_UNIT_SCALE = 1e-6      # microvolts -> volts

# Keys present in every .mat that are NOT the EEG data array
_NON_DATA_KEYS = {"samplingRate", "Impedances_0"}


# ── Subject ID handling ────────────────────────────────────────────────────────

def normalise_subject_id(raw_id) -> str:
    """
    Normalise a MODMA subject ID to the 8-digit zero-padded string used in
    filenames.

    Examples
    --------
    2010002   -> '02010002'
    '2010002' -> '02010002'
    '02010002'-> '02010002'
    """
    digits = re.sub(r"\D", "", str(raw_id))
    return digits.zfill(8)


def subject_id_from_filename(filename: str) -> str | None:
    """
    Extract the 8-digit subject ID from a MODMA filename.

    Filenames look like:
        '02010002rest 20150416 1017..mat'
        '02010021 20150805 1730.mat.mat'
        '02010008_rest 20150619 1653.mat'

    The subject ID is the leading run of digits.
    """
    match = re.match(r"(\d{8})", filename)
    return match.group(1) if match else None


# ── Labels ─────────────────────────────────────────────────────────────────────

def load_modma_labels(xlsx_path: Path) -> pd.DataFrame:
    """
    Load the MODMA subject information spreadsheet.

    Parameters
    ----------
    xlsx_path : Path
        Path to subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx

    Returns
    -------
    pd.DataFrame with columns:
        subject   : str, 8-digit zero-padded ID matching filenames
        label     : str, 'MDD' or 'HC' (taken directly from the 'type' column)
        phq9      : int
        age       : int
        gender    : str
    """
    df = pd.read_excel(xlsx_path)

    # Drop the trailing annotation columns (abbreviation notes, not data)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    out = pd.DataFrame({
        "subject": df["subject id"].apply(normalise_subject_id),
        "label": df["type"].astype(str).str.strip().str.upper(),
        "phq9": df["PHQ-9"],
        "age": df["age"],
        "gender": df["gender"].astype(str).str.strip(),
    })

    # Sanity: labels should only be MDD or HC
    unexpected = set(out["label"]) - {"MDD", "HC"}
    if unexpected:
        raise ValueError(f"Unexpected labels in spreadsheet: {unexpected}")

    return out


# ── Discovery ──────────────────────────────────────────────────────────────────

def discover_modma_subjects(data_dir: Path) -> dict[str, Path]:
    """
    Find all MODMA .mat recordings and map subject ID -> file path.

    Parameters
    ----------
    data_dir : Path
        Folder containing the per-subject .mat files

    Returns
    -------
    dict {subject_id: Path}
    """
    data_dir = Path(data_dir)
    mapping: dict[str, Path] = {}

    for f in sorted(data_dir.iterdir()):
        if not f.is_file() or not f.name.endswith(".mat"):
            continue
        sid = subject_id_from_filename(f.name)
        if sid is None:
            print(f"  ! Skipping (no subject ID in name): {f.name}")
            continue
        if sid in mapping:
            print(f"  ! Duplicate subject {sid}: {f.name} (keeping first)")
            continue
        mapping[sid] = f

    return mapping


# ── Loading ────────────────────────────────────────────────────────────────────

def _find_data_key(mat: dict) -> str:
    """
    Locate the EEG data array key.

    The key name is dynamic (encodes subject ID and date), and some
    recordings contain extra keys such as 'DIN_1' (EGI digital input /
    event markers). The data array is therefore identified by shape:
    it is the only 2D array with MODMA_N_CHANNELS rows.
    """
    candidates = []
    for k, v in mat.items():
        if k.startswith("__") or k in _NON_DATA_KEYS:
            continue
        arr = np.asarray(v)
        if arr.ndim == 2 and arr.shape[0] == MODMA_N_CHANNELS:
            candidates.append(k)

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        available = [k for k in mat if not k.startswith("__")]
        raise ValueError(
            f"No {MODMA_N_CHANNELS}-channel data array found. Keys: {available}"
        )
    raise ValueError(f"Ambiguous data keys (all {MODMA_N_CHANNELS} rows): {candidates}")


def load_modma_subject(mat_path: Path, drop_reference: bool = True):
    """
    Load one MODMA .mat recording into an MNE RawArray.

    Parameters
    ----------
    mat_path : Path
    drop_reference : bool
        If True, drop the Cz reference channel (channel 129), keeping the
        128 EEG channels. Recommended, since an average reference is applied
        downstream and the recording reference carries no independent signal.

    Returns
    -------
    raw : mne.io.RawArray
    """
    mat_path = Path(mat_path)
    mat = scipy.io.loadmat(mat_path)

    data_key = _find_data_key(mat)
    data = np.asarray(mat[data_key], dtype=np.float64)

    if data.shape[0] != MODMA_N_CHANNELS:
        raise ValueError(
            f"{mat_path.name}: expected {MODMA_N_CHANNELS} channels, "
            f"got shape {data.shape}"
        )

    sfreq = float(np.asarray(mat["samplingRate"]).ravel()[0])

    # Channel names must match the GSN-HydroCel-129 montage
    ch_names = [f"E{i}" for i in range(1, 129)] + ["Cz"]
    ch_types = ["eeg"] * MODMA_N_CHANNELS

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data * MODMA_UNIT_SCALE, info, verbose=False)
    raw.set_montage(MODMA_MONTAGE, on_missing="warn")

    if drop_reference:
        raw.drop_channels(["Cz"])

    return raw


def load_modma_impedances(mat_path: Path) -> np.ndarray:
    """Return the (129,) impedance vector for a recording, in kOhm."""
    mat = scipy.io.loadmat(Path(mat_path))
    return np.asarray(mat["Impedances_0"]).ravel()