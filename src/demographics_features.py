"""
Add age and sex as input features to the classifier, per supervisor request.
Provides a three-way comparison so the result is interpretable:
  1. EEG features alone
  2. Demographics alone (age + sex)   -- baseline
  3. EEG + demographics                -- does EEG add anything beyond demographics?

Reporting all three prevents the confound of presenting an EEG+demographics
number as if the EEG improved; it shows explicitly what demographics contribute.
"""
import numpy as np


def build_demographic_features(groups, labels_df):
    """
    Build per-epoch age and sex feature columns aligned to `groups`.

    sex encoded as 0/1 (F=0, M=1); age used as-is (standardised later by the
    pipeline's scaler).

    Returns
    -------
    demo : ndarray (n_epochs, 2)  columns [age, sex]
    names : list[str]
    """
    age_map = dict(zip(labels_df["subject"], labels_df["age"]))
    sex_map = dict(zip(labels_df["subject"], labels_df["gender"]))

    age_col = np.array([age_map.get(g, np.nan) for g in groups], dtype=float)
    sex_col = np.array([1.0 if sex_map.get(g) == "M" else 0.0 for g in groups], dtype=float)

    demo = np.column_stack([age_col, sex_col])
    # fill any missing age with the median
    if np.isnan(demo[:, 0]).any():
        med = np.nanmedian(demo[:, 0])
        demo[np.isnan(demo[:, 0]), 0] = med
    return demo, ["age", "sex"]