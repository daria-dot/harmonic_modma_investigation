"""
src/objective3.py
==================
Objective 3: ElasticNet logistic classifier with nested, subject-grouped
cross-validation, AUC reporting, and permutation testing.

Design 
------------
Features are extracted PER EPOCH (each 2s eyes-closed window is one sample),
not averaged per subject, to maximise sample size for such a small cohort.
This creates a leakage risk: epochs from the same subject are highly
correlated (same brain, same electrodes, same session), so if epochs from
one subject end up split across train/test, the classifier can learn
subject-specific signal rather than genuine MDD-vs-HC signal, inflating AUC.

To prevent this, BOTH the outer and inner cross-validation loops use
StratifiedGroupKFold, grouping by subject ID. This guarantees all epochs
from a given subject stay entirely within one fold (train or test, never
split), while still balancing HC/MDD labels across folds as evenly as
possible.

With only ~7 usable subjects, k=3 is used for both loops by default — small
enough to keep folds non-trivial, but coarse, so results should be read as
indicative of a small pilot study rather than a validated biomarker.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config


# ── Build per-epoch feature matrix ────────────────────────────────────────────


def build_epoch_feature_matrix(results: dict, labels_df: pd.DataFrame):
    """
    Build a per-epoch feature matrix (X, y, groups) from Objective 2 results.

    Parameters
    ----------
    results : dict {subject_id: {..., 'features': {'per_epoch_power': ...}}}
        Output of src.objective2.run_objective2 (the `results` dict)
    labels_df : pd.DataFrame
        Output of src.labels.get_labels(), columns: subject, phq9, label

    Returns
    -------
    X : ndarray, shape (n_total_epochs, n_modes)
        Per-epoch mode power features
    y : ndarray, shape (n_total_epochs,)
        Binary label, 1 = MDD, 0 = HC
    groups : ndarray, shape (n_total_epochs,)
        Subject ID for each epoch (for grouped CV)
    feature_names : list of str
    subjects_used : list of str
        Subjects actually included (HC/MDD only, EXCLUDE/UNKNOWN dropped)
    """
    label_map = dict(zip(labels_df["subject"], labels_df["label"]))

    X_parts, y_parts, group_parts = [], [], []
    subjects_used = []

    for sid, r in results.items():
        label = label_map.get(sid)
        if label not in ("HC", "MDD"):
            print(f"  Skipping {sid}: label is '{label}', not HC/MDD")
            continue

        feats = r["features"]
        per_epoch_power = feats["per_epoch_power"]  # (n_epochs, n_modes)

        # Stack power + variance + ratio horizontally
        parts = [per_epoch_power]
        if "per_epoch_variance" in feats:
            parts.append(feats["per_epoch_variance"])
        if "per_epoch_ratio" in feats:
            parts.append(feats["per_epoch_ratio"])
        subject_X = np.hstack(parts)

        n_epochs = subject_X.shape[0]
        X_parts.append(subject_X)
        y_parts.append(np.full(n_epochs, 1 if label == "MDD" else 0))
        group_parts.append(np.full(n_epochs, sid, dtype=object))
        subjects_used.append(sid)

    if not X_parts:
        raise ValueError("No subjects with HC/MDD labels found in results.")

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    groups = np.concatenate(group_parts)

    n_total_features = X.shape[1]
    feature_names = [f"mode{i}_power" for i in range(19)]
    if n_total_features > 19:
        feature_names += ["mode_variance", "low_high_ratio"][:n_total_features - 19]
    return X, y, groups, feature_names, subjects_used


# ── Nested cross-validation ───────────────────────────────────────────────────


def make_pipeline():
    """ElasticNet logistic regression pipeline with standardisation."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=0.5,        # default; overridden by the grid search
                    max_iter=20000,
                    tol=1e-3,
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ]
    )
    

def default_param_grid():
    """Hyperparameter grid for the inner CV loop."""
    return {
        "clf__C": [0.01, 0.1, 1.0],
        "clf__l1_ratio": [0.1, 0.9],
    }


def nested_cv_auc(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_outer_splits: int = 3,
    n_inner_splits: int = 3,
    param_grid: dict = None,
    random_state: int = None,
):
    """
    Run nested, subject-grouped, stratified cross-validation.

    Outer loop: gives an unbiased estimate of generalisation performance.
    Inner loop (via GridSearchCV): tunes ElasticNet C and l1_ratio.

    Both loops use StratifiedGroupKFold so epochs from the same subject
    never appear in both train and test within a split.

    Returns
    -------
    dict with keys:
        fold_results, y_true_all, y_score_all, mean_auc, std_auc, overall_auc
    """
    if param_grid is None:
        param_grid = default_param_grid()
    if random_state is None:
        random_state = config.RANDOM_STATE

    outer_cv = StratifiedGroupKFold(
        n_splits=n_outer_splits, shuffle=True, random_state=random_state
    )

    fold_results = []
    y_true_all = []
    y_score_all = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y, groups=groups)
    ):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        test_subjects = sorted(set(groups[test_idx]))

        inner_cv = StratifiedGroupKFold(
            n_splits=n_inner_splits, shuffle=True, random_state=random_state
        )

        # Materialise the splits as a list so GridSearchCV's parallel workers
        # can pickle them (a generator from .split() cannot be pickled)
        inner_splits = list(inner_cv.split(X_train, y_train, groups=groups_train))

        pipeline = make_pipeline()
        search = GridSearchCV(
            pipeline,
            param_grid,
            scoring="roc_auc",
            cv=inner_splits,
            n_jobs=1,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_score = best_model.predict_proba(X_test)[:, 1]

        if len(set(y_test)) == 2:
            fold_auc = roc_auc_score(y_test, y_score)
        else:
            fold_auc = float("nan")

        fold_results.append(
            {
                "fold": fold_idx,
                "test_subjects": test_subjects,
                "n_test_epochs": len(test_idx),
                "auc": fold_auc,
                "best_params": search.best_params_,
            }
        )

        y_true_all.append(y_test)
        y_score_all.append(y_score)

        print(
            f"  Fold {fold_idx}: test subjects={test_subjects}, "
            f"n_epochs={len(test_idx)}, AUC={fold_auc:.3f}, "
            f"best_params={search.best_params_}"
        )

    y_true_all = np.concatenate(y_true_all)
    y_score_all = np.concatenate(y_score_all)

    fold_aucs = [f["auc"] for f in fold_results if not np.isnan(f["auc"])]
    mean_auc = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
    std_auc = float(np.std(fold_aucs)) if fold_aucs else float("nan")
    overall_auc = roc_auc_score(y_true_all, y_score_all)

    return {
        "fold_results": fold_results,
        "y_true_all": y_true_all,
        "y_score_all": y_score_all,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "overall_auc": overall_auc,
    }


# ── Permutation testing ───────────────────────────────────────────────────────


def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_permutations: int = 1000,
    n_outer_splits: int = 3,
    n_inner_splits: int = 3,
    random_state: int = None,
    verbose_every: int = 100,
):
    """
    Permutation test for the nested CV pipeline.

    Labels are shuffled AT THE SUBJECT LEVEL (not per-epoch) before each
    permutation run, since shuffling per-epoch would break the grouping
    structure and leak information between train/test in a way that doesn't
    correspond to a valid null hypothesis.

    Returns
    -------
    dict with keys: true_auc, permuted_aucs, p_value
    """
    if random_state is None:
        random_state = config.RANDOM_STATE

    rng = np.random.RandomState(random_state)

    unique_groups = np.unique(groups)
    group_to_label = {}
    for g in unique_groups:
        labels_for_group = y[groups == g]
        assert len(set(labels_for_group)) == 1, f"Subject {g} has inconsistent labels"
        group_to_label[g] = labels_for_group[0]

    print("Running true (unshuffled) nested CV...")
    true_result = nested_cv_auc(
        X, y, groups,
        n_outer_splits=n_outer_splits,
        n_inner_splits=n_inner_splits,
        random_state=random_state,
    )
    true_auc = true_result["overall_auc"]
    print(f"True overall AUC: {true_auc:.3f}\n")

    permuted_aucs = np.zeros(n_permutations)

    print(f"Running {n_permutations} permutations...")
    for i in range(n_permutations):
        shuffled_subjects = rng.permutation(unique_groups)
        shuffled_label_map = dict(zip(unique_groups, [group_to_label[g] for g in shuffled_subjects]))
        y_perm = np.array([shuffled_label_map[g] for g in groups])

        result = nested_cv_auc(
            X, y_perm, groups,
            n_outer_splits=n_outer_splits,
            n_inner_splits=n_inner_splits,
            random_state=random_state,
        )
        permuted_aucs[i] = result["overall_auc"]

        if (i + 1) % verbose_every == 0:
            print(f"  {i + 1}/{n_permutations} permutations done")

    p_value = float(np.mean(permuted_aucs >= true_auc))

    # Refit a single model on all data for feature-importance inspection
    # (separate from the CV estimate of performance — this is for interpretation only)
    
    full_cv = StratifiedGroupKFold(n_splits=n_inner_splits, shuffle=True,
                                    random_state=config.RANDOM_STATE)
    full_search = GridSearchCV(
        make_pipeline(), default_param_grid(), scoring="roc_auc",
        cv=list(full_cv.split(X, y, groups=groups)), n_jobs=-1,
    )
    full_search.fit(X, y)
    final_coefs = full_search.best_estimator_.named_steps["clf"].coef_.ravel()

    return {
        "true_auc": true_auc,
        "permuted_aucs": permuted_aucs,
        "p_value": p_value,
        "final_coefs": final_coefs,
        "true_fold_results": true_result["fold_results"],
        "true_y_true": true_result["y_true_all"],
        "true_y_score": true_result["y_score_all"],
    }


# ── Main entrypoint ────────────────────────────────────────────────────────────


def run_objective3(
    results: dict,
    labels_df: pd.DataFrame,
    n_outer_splits: int = 3,
    n_inner_splits: int = 3,
    n_permutations: int = 1000,
):
    """
    Run the full Objective 3 pipeline: build features, nested CV, permutation test.

    Returns
    -------
    dict with keys: permutation_result, X, y, groups, feature_names, subjects_used
    """
    print("Building per-epoch feature matrix...")
    X, y, groups, feature_names, subjects_used = build_epoch_feature_matrix(
        results, labels_df
    )
    print(f"  {X.shape[0]} total epochs from {len(subjects_used)} subjects "
          f"({sum(y == 1)} MDD epochs, {sum(y == 0)} HC epochs)")
    print(f"  Subjects used: {subjects_used}\n")

    print("Running permutation test (includes the true nested CV run)...")
    perm_result = permutation_test(
        X, y, groups,
        n_permutations=n_permutations,
        n_outer_splits=n_outer_splits,
        n_inner_splits=n_inner_splits,
    )

    print(f"\n══ OBJECTIVE 3 RESULTS ══════════════════════════════════")
    print(f"Overall AUC (pooled out-of-fold predictions): {perm_result['true_auc']:.3f}")
    print(f"Permutation p-value ({n_permutations} permutations): {perm_result['p_value']:.4f}")

    return {
        "permutation_result": perm_result,
        "X": X, "y": y, "groups": groups,
        "feature_names": feature_names,
        "subjects_used": subjects_used,
        "final_coefs": perm_result["final_coefs"],
        "cv_y_true": perm_result["true_y_true"],
        "cv_y_score": perm_result["true_y_score"],
        "fold_results": perm_result["true_fold_results"],
    }