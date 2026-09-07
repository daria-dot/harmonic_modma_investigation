"""
src/load.py
===========
Discover subjects and load Neuroelectrics .easy files into MNE RawArray objects.
"""


from __future__ import annotations
from pathlib import Path

import mne
import numpy as np
import pandas as pd

import config

mne.set_log_level("WARNING")


def discover_subjects(base_dir: Path = config.BASE_DIR) -> list[str]:
    """
    Auto-discover all subject folders containing a .easy file.

    Parameters
    ----------
    base_dir : Path
        Path to the MDD EEG data folder

    Returns
    -------
    list of str
        Subject folder names, sorted alphabetically, excluding config.EXCLUDE_FOLDERS
    """
    base_dir = Path(base_dir)
    subjects = []

    if not base_dir.exists():
        print(f"✗ Base directory not found: {base_dir}")
        return subjects

    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name in config.EXCLUDE_FOLDERS:
            continue
        if any(folder.glob("*.easy")):
            subjects.append(folder.name)

    return subjects


def load_subject(subject_id: str, base_dir: Path = config.BASE_DIR):
    """
    Load a single subject's .easy file into an MNE RawArray.

    Parameters
    ----------
    subject_id : str
        Folder name, e.g. 'S05'
    base_dir : Path
        Path to the MDD EEG data folder

    Returns
    -------
    raw : mne.io.RawArray or None
    df : pd.DataFrame or None
        Raw dataframe (kept around so trigger columns can be used downstream
        for epoch extraction without re-reading the file from disk)
    """
    subj_path = Path(base_dir) / subject_id
    easy_files = list(subj_path.glob("*.easy"))

    if not easy_files:
        print(f"  ✗ {subject_id}: no .easy file")
        return None, None

    easy_path = easy_files[0]
    size_mb = easy_path.stat().st_size / 1024**2
    print(f"  Loading {subject_id} ({size_mb:.1f} MB)...")

    df = pd.read_csv(easy_path, sep=r"\s+", header=None, names=config.COL_NAMES)
    print(f"    {len(df):,} samples  |  {len(df) / config.SFREQ / 60:.1f} min")

    ch_names = ["ECG"] + config.EEG_CHANNELS
    ch_types = ["ecg"] + ["eeg"] * len(config.EEG_CHANNELS)
    data = df[ch_names].values.T * 1e-9  # nV -> V

    info = mne.create_info(ch_names=ch_names, sfreq=config.SFREQ, ch_types=ch_types)
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage("standard_1020", on_missing="warn")

    return raw, df


def load_all_subjects(
    subjects: list[str] | None = None, base_dir: Path = config.BASE_DIR
) -> dict:
    """
    Load all (or a specified subset of) subjects.

    Parameters
    ----------
    subjects : list of str, optional
        Subset of subject IDs to load. Defaults to auto-discovered subjects.
    base_dir : Path

    Returns
    -------
    dict {subject_id: (raw, df)}
    """
    if subjects is None:
        subjects = discover_subjects(base_dir)

    print(f"Loading {len(subjects)} subjects from {base_dir}\n")
    loaded = {}

    for sid in subjects:
        raw, df = load_subject(sid, base_dir)
        if raw is not None:
            loaded[sid] = (raw, df)

    print(f"\nLoaded {len(loaded)}/{len(subjects)} subjects successfully.")
    return loaded


def inspect(loaded: dict) -> None:
    """Print a summary table of all loaded subjects."""
    print(f'\n{"Subject":<10} {"Channels":<10} {"Samples":<12} {"Duration (min)":<16} {"Sfreq"}')
    print("-" * 58)
    for sid, (raw, _df) in loaded.items():
        n_ch = len(raw.ch_names)
        n_samp = raw.n_times
        dur = raw.n_times / raw.info["sfreq"] / 60
        sfreq = raw.info["sfreq"]
        print(f"{sid:<10} {n_ch:<10} {n_samp:<12,} {dur:<16.1f} {sfreq}")


if __name__ == "__main__":
    loaded = load_all_subjects()
    inspect(loaded)
