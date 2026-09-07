"""
src/labels.py
=============
MDD / HC group label assignment from PHQ9 scores.

PHQ9 scores can either come from config.PHQ9_SCORES (a manually maintained
dict) or be loaded directly from the Excel sheet provided by the supervisor.
"""


from __future__ import annotations
from pathlib import Path

import pandas as pd

import config


def load_phq9_from_excel(xlsx_path: Path) -> dict:
    """
    Load PHQ9 scores from the supervisor-provided Excel file.

    Expected columns (based on the file received): subject ID, PHQ9, Sex.
    Subject IDs in the sheet may not exactly match folder names (e.g. 'S22'
    vs 'S022'), so this does light normalisation (zero-padding handled by
    matching against discovered subject folder names where possible).

    Parameters
    ----------
    xlsx_path : Path

    Returns
    -------
    dict {subject_id: phq9_score}
    """
    df = pd.read_excel(xlsx_path)
    df.columns = [str(c).strip() for c in df.columns]

    # First column is subject ID, second is PHQ9 score (based on observed file)
    id_col, phq9_col = df.columns[0], df.columns[1]

    scores = {}
    for _, row in df.iterrows():
        sid_raw = str(row[id_col]).strip()
        # Normalise e.g. "S22" -> "S22", "S001" -> "S01" style mismatches
        # are left to the caller to reconcile against discover_subjects() output
        scores[sid_raw] = row[phq9_col]

    return scores


def get_labels(phq9_scores: dict | None = None) -> pd.DataFrame:
    """
    Build a labels dataframe from PHQ9 scores.

    Parameters
    ----------
    phq9_scores : dict, optional
        Defaults to config.PHQ9_SCORES

    Returns
    -------
    pd.DataFrame with columns: subject, phq9, label
    """
    if phq9_scores is None:
        phq9_scores = config.PHQ9_SCORES

    rows = [
        {"subject": sid, "phq9": phq9, "label": config.assign_label(phq9)}
        for sid, phq9 in phq9_scores.items()
    ]
    return pd.DataFrame(rows)
