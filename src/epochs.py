"""
src/epochs.py
=============
Eyes-closed resting-state epoch extraction.

Each occurrence of the eyes-closed trigger code marks the start of a
60-second eyes-closed block. Each block is subdivided into short fixed-length
windows (EPOCH_LEN_S) BEFORE artefact rejection is applied, so a single
artefact only drops that small window rather than the whole 60s block.
"""


from __future__ import annotations
import mne
import pandas as pd

import config

mne.set_log_level("WARNING")


def extract_ec_epochs(
    raw: mne.io.RawArray,
    df: pd.DataFrame,
    epoch_len_s: float = config.EPOCH_LEN_S,
    reject_threshold: float = config.REJECT_THRESHOLD_EEG,
):
    """
    Extract eyes-closed resting-state epochs from a preprocessed Raw object.

    Parameters
    ----------
    raw : mne.io.RawArray
        Preprocessed raw data (output of src.preprocess.preprocess)
    df : pd.DataFrame
        Original raw dataframe for this subject (needed for trigger column
        and Unix timestamps, since raw doesn't retain these)
    epoch_len_s : float
        Length of each analysis epoch in seconds
    reject_threshold : float
        Peak-to-peak rejection threshold in Volts

    Returns
    -------
    epochs : mne.Epochs or None
    """
    trig_rows = df[df["trigger"] == config.EC_TRIGGER].index.tolist()
    if not trig_rows:
        print("    ✗ No eyes-closed triggers found")
        return None

    t0 = df["timestamp"].iloc[0]
    onset_times = (df.loc[trig_rows, "timestamp"] - t0) / 1000  # ms -> s

    all_epochs = []
    for onset in onset_times.values:
        raw_block = raw.copy().crop(tmin=onset, tmax=onset + config.EC_BLOCK_S)
        block_epochs = mne.make_fixed_length_epochs(
            raw_block, duration=epoch_len_s, overlap=0, preload=True, verbose=False
        )
        all_epochs.append(block_epochs)

    if not all_epochs:
        return None

    epochs = mne.concatenate_epochs(all_epochs, verbose=False)

    n_before = len(epochs)
    epochs.drop_bad(reject={"eeg": reject_threshold}, verbose=False)
    n_after = len(epochs)
    print(
        f"    {n_before} {epoch_len_s:.0f}s windows found -> "
        f"{n_after} retained after {reject_threshold * 1e6:.0f}µV rejection"
    )

    return epochs


def extract_all_epochs(preprocessed: dict, raw_dfs: dict) -> dict:
    """
    Extract eyes-closed epochs for all preprocessed subjects.

    Parameters
    ----------
    preprocessed : dict {subject_id: mne.io.RawArray}
    raw_dfs : dict {subject_id: pd.DataFrame}
        The original dataframes from src.load.load_all_subjects (for triggers)

    Returns
    -------
    epochs_dict : dict {subject_id: mne.Epochs}
    """
    epochs_dict = {}
    for sid, raw_clean in preprocessed.items():
        df = raw_dfs.get(sid)
        if df is None:
            print(f"  ✗ {sid}: no raw dataframe available, skipping epoch extraction")
            continue
        print(f"  {sid}...")
        epochs = extract_ec_epochs(raw_clean, df)
        if epochs is not None:
            epochs_dict[sid] = epochs
    return epochs_dict
