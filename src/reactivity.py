"""
src/reactivity.py
=================
Eyes-closed (EC) vs eyes-open (EO) reactivity analysis for the Neuroelectrics
dataset.

Motivation
----------
All prior analyses used a single static condition (eyes-closed resting). A
reported MDD marker, however, is *alpha reactivity* -- the CHANGE in activity
between eyes-closed and eyes-open, rather than either condition alone. Healthy
individuals typically show strong alpha suppression when opening the eyes;
this suppression is reported to be blunted in depression. Such a signal lives
in the EC-minus-EO difference and is invisible to any single-condition feature.

This module:
  1. Extracts eyes-open epochs (trigger 1) alongside the existing eyes-closed
     epochs (trigger 2), from the same preprocessed recordings.
  2. Computes graph-harmonic mode power per condition, per subject.
  3. Derives per-subject reactivity features:
       - EC power, EO power (each condition)
       - EC - EO (absolute reactivity)
       - (EC - EO) / (EC + EO) (normalised reactivity, robust to amplitude)

Because reactivity is a per-SUBJECT contrast (one graph and one difference per
subject), this analysis is naturally per-subject rather than per-epoch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import objective2

EO_TRIGGER = config.TRIGGER_CODES["rest_open"]   # 1
EC_TRIGGER = config.TRIGGER_CODES["rest_closed"] # 2


# ── Per-condition epoch extraction ─────────────────────────────────────────────

def extract_condition_epochs(raw, df, trigger, epoch_len_s=None, reject_threshold=None):
    """
    Extract fixed-length epochs for one resting condition (by trigger code).

    Mirrors src.epochs.extract_ec_epochs but with the trigger as a parameter,
    so it works for both eyes-open (1) and eyes-closed (2).
    """
    import mne
    if epoch_len_s is None:
        epoch_len_s = config.EPOCH_LEN_S
    if reject_threshold is None:
        reject_threshold = config.REJECT_THRESHOLD_EEG

    trig_rows = df[df["trigger"] == trigger].index.tolist()
    if not trig_rows:
        return None

    t0 = df["timestamp"].iloc[0]
    onset_times = (df.loc[trig_rows, "timestamp"] - t0) / 1000

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
    epochs.drop_bad(reject={"eeg": reject_threshold}, verbose=False)
    return epochs if len(epochs) > 0 else None


# ── Per-subject reactivity features ────────────────────────────────────────────

def compute_reactivity_features(ec_epochs, eo_epochs, n_modes=None):
    """
    For one subject, compute graph-harmonic mode power in each condition and
    derive reactivity features.

    The graph harmonics are built from the EYES-CLOSED connectivity (the
    canonical resting graph); both conditions are then projected onto the SAME
    harmonic basis so their mode powers are directly comparable.

    Parameters
    ----------
    ec_epochs, eo_epochs : mne.Epochs
    n_modes : int or None
        If given, keep only the lowest n_modes (reduces overfitting downstream).

    Returns
    -------
    dict with per-mode arrays: ec_power, eo_power, reactivity_diff,
    reactivity_norm
    """
    # Build the harmonic basis from eyes-closed connectivity
    wpli, ch_names = objective2.compute_wpli(ec_epochs)
    _, eigenvectors = objective2.compute_graph_harmonics(wpli)

    # Project each condition onto the same basis; average mode power over epochs
    ec_feats = objective2.project_and_extract_features(ec_epochs, eigenvectors, ch_names)
    eo_feats = objective2.project_and_extract_features(eo_epochs, eigenvectors, ch_names)

    ec_power = ec_feats["mode_power"]   # mean power per mode across EC epochs
    eo_power = eo_feats["mode_power"]

    if n_modes is not None:
        ec_power = ec_power[:n_modes]
        eo_power = eo_power[:n_modes]

    diff = ec_power - eo_power
    denom = ec_power + eo_power
    denom_safe = np.where(denom > 0, denom, 1e-30)
    norm = diff / denom_safe

    return {
        "ec_power": ec_power,
        "eo_power": eo_power,
        "reactivity_diff": diff,
        "reactivity_norm": norm,
    }


# ── Assemble per-subject reactivity table ──────────────────────────────────────

def build_reactivity_table(preprocessed, raw_dfs, labels_df, n_modes=None):
    """
    Build a per-subject feature table of reactivity features.

    Parameters
    ----------
    preprocessed : dict {subject_id: mne.io.RawArray}
        Preprocessed raws (from Objective 1). Needed because both conditions
        must be re-epoched from the same recording.
    raw_dfs : dict {subject_id: pd.DataFrame}
        Original dataframes (for trigger timestamps).
    labels_df : pd.DataFrame
    n_modes : int or None

    Returns
    -------
    feature_matrices : dict of {feature_type: (X, y, subjects)}
        One entry per reactivity feature type, each a per-subject matrix
        ready for classification. Also includes 'ec_only' and 'eo_only'
        for comparison.
    """
    label_map = dict(zip(labels_df["subject"], labels_df["label"]))

    rows = {"ec_power": [], "eo_power": [], "reactivity_diff": [], "reactivity_norm": []}
    y_list, subjects_used = [], []

    for sid, raw in preprocessed.items():
        label = label_map.get(sid)
        if label not in ("HC", "MDD"):
            continue
        df = raw_dfs.get(sid)
        if df is None:
            continue

        ec = extract_condition_epochs(raw, df, EC_TRIGGER)
        eo = extract_condition_epochs(raw, df, EO_TRIGGER)
        if ec is None or eo is None or len(ec) < 2 or len(eo) < 2:
            print(f"  {sid}: insufficient epochs (EC={0 if ec is None else len(ec)}, "
                  f"EO={0 if eo is None else len(eo)}), skipping")
            continue

        feats = compute_reactivity_features(ec, eo, n_modes=n_modes)
        for key in rows:
            rows[key].append(feats[key])
        y_list.append(1 if label == "MDD" else 0)
        subjects_used.append(sid)
        print(f"  {sid} ({label}): EC={len(ec)} EO={len(eo)} epochs")

    y = np.array(y_list)
    feature_matrices = {}
    for key in rows:
        feature_matrices[key] = (np.vstack(rows[key]), y, subjects_used)

    return feature_matrices