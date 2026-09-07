"""
config.py
=========
Central configuration for the MDD EEG analysis pipeline.

Edit BASE_DIR to point at your local 'MDD EEG' data folder.
Everything else (channel names, filter settings, trigger codes, PHQ9 scores)
lives here so the rest of the codebase never hardcodes these values.
"""


from __future__ import annotations
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────────



import os
from pathlib import Path

if os.path.exists("/content/drive"):
    BASE_DIR = Path("/content/drive/MyDrive/mdd_eeg/MDD EEG")
else:
    BASE_DIR = Path("/Users/mustachelover/Downloads/MDD Zipped files/Datasets and Matlab codes /MDD EEG")
# Where intermediate/derived outputs get cached (created automatically)
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

# Folders to skip when auto-discovering subjects (pilot/test data, not real subjects)
EXCLUDE_FOLDERS = {"TEST"}


# ── Recording / hardware parameters ──────────────────────────────────────────────

SFREQ = 500  # Hz, Enobio/NIC recording sample rate

EEG_CHANNELS = [
    "Oz", "PO3", "PO4", "P7", "P3", "Pz", "P4", "P8",
    "CP1", "CP2", "FC1", "Fz", "C3", "FC2", "F7", "F3",
    "C4", "F4", "F8",
]

# Column layout of the .easy file, confirmed by direct inspection of raw rows:
# sample_idx | ECG | 19 EEG channels | ACC_x | ACC_y | ACC_z | trigger | unix_timestamp_ms
COL_NAMES = (
    ["sample_idx", "ECG"]
    + EEG_CHANNELS
    + ["ACC_x", "ACC_y", "trigger", "timestamp"]
)


# ── Preprocessing parameters (Objective 1) ───────────────────────────────────────

BANDPASS_LOW = 1.0     # Hz
BANDPASS_HIGH = 40.0    # Hz
NOTCH_FREQ = 50.0      # Hz, UK mains
BAD_CH_ZSCORE = 3.0    # std threshold for bad channel detection (variance-based)


# ── Epoching parameters (Objective 2) ────────────────────────────────────────────

EPOCH_LEN_S = 2.0       # seconds per analysis epoch
EC_BLOCK_S = 60.0       # duration of each eyes-closed resting block (s)
REJECT_THRESHOLD_EEG = 150e-6  # peak-to-peak rejection threshold, Volts (150 µV)

# Trigger codes, confirmed from restdata.m (provided by supervisor)
TRIGGER_CODES = {
    "rest_open": 1,
    "rest_closed": 2,
    "breathe_open": 3,
    "breathe_closed": 4,
    "rvp_correct": 6,
    "rvp_incorrect": 7,
}
EC_TRIGGER = TRIGGER_CODES["rest_closed"]


# ── Graph harmonic feature parameters (Objective 2) ──────────────────────────────

ALPHA_BAND = (8.0, 13.0)  # Hz
N_LOW_MODES = 5            # number of lowest-frequency harmonics used in low/high ratio
N_HIGH_MODES = 5           # number of highest-frequency harmonics used in low/high ratio

# Frequency bands for multi-band analysis (Objective 4)
BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

# ── Labels (Objective 3) ──────────────────────────────────────────────────────────
# PHQ9 scores per subject. Update this dict (or load from the Excel sheet via
# src/labels.py) as more subjects become available.
#
# Group assignment per protocol: PHQ9 < 5 -> HC, PHQ9 > 11 -> MDD, else EXCLUDE.

_PHQ9_RAW = {
    "S001": 0, "S005": 1, "S017": 1, "S002": 2, "S004": 2, "S013": 2,
    "S007": 3, "S22": 3, "S003": 4, "S016": 4, "S21": 6, "S24": 7,
    "S25": 8, "S019": 9, "S018": 10, "S012": 13, "S015": 13, "S27": 13,
    "S008": 14, "S009": 14, "S011": 14, "S014": 14, "S006": 15, "S26": 15,
    "S010": 17, "S020": 17, "S23": 26,
}
def _normalise_subject_id(raw_id: str) -> str:
    """Normalise PHQ9 sheet IDs (S001, S22, S010...) to folder-name format (S01, S22, S10...)."""
    num = int(raw_id.lstrip("Ss"))
    return f"S{num:02d}"


PHQ9_SCORES = {_normalise_subject_id(k): v for k, v in _PHQ9_RAW.items()}

PHQ9_HC_MAX = 5     # PHQ9 < this -> healthy control
PHQ9_MDD_MIN = 11    # PHQ9 > this -> MDD


def assign_label(phq9: Optional[float]) -> str:
    """Assign MDD / HC / EXCLUDE / UNKNOWN label from a PHQ9 score."""
    if phq9 is None:
        return "UNKNOWN"
    if phq9 < PHQ9_HC_MAX:
        return "HC"
    if phq9 > PHQ9_MDD_MIN:
        return "MDD"
    return "EXCLUDE"


# Ensure output directories exist on import
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Classification parameters (Objective 3) ──────────────────────────────────────

RANDOM_STATE = 42        # for reproducible CV splits and permutation tests
N_OUTER_SPLITS = 5      # outer CV folds (small, given ~7 subjects)
N_INNER_SPLITS = 3      # inner CV folds (hyperparameter tuning)
N_PERMUTATIONS = 200   # permutation test iterations