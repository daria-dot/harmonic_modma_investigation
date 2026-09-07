"""
src/svm_classifier.py
=====================
Standalone RBF-kernel SVM classifier with subject-grouped nested cross-
validation and permutation testing.

This mirrors the evaluation framework of objective3.permutation_test exactly
(StratifiedGroupKFold on both loops, pooled out-of-fold AUC, subject-level
label permutation) but swaps the linear ElasticNet for a non-linear RBF-SVM.

Purpose: the ElasticNet baseline is linear and cannot capture non-linear or
interaction effects. An RBF-SVM can. Running it on identical features and
folds tests whether the near-chance linear results are due to model choice
(missed non-linearity) rather than an absence of separable signal.

Kept fully separate from objective3.py so the validated linear pipeline is
untouched.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config


def make_svm_pipeline():
    """Standardisation + RBF-kernel SVM."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            probability=True,           # required for AUC
            class_weight="balanced",    # handle MDD/HC imbalance
            random_state=config.RANDOM_STATE,
        )),
    ])


def svm_param_grid():
    """C (regularisation) and gamma (RBF kernel width)."""
    return {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__gamma": ["scale", 0.01, 0.1],
    }


def svm_nested_cv_auc(X, y, groups, n_outer_splits=5, n_inner_splits=3,
                      random_state=None, verbose=True):
    """Subject-grouped nested CV with an RBF-SVM. Returns overall pooled AUC."""
    if random_state is None:
        random_state = config.RANDOM_STATE

    outer = StratifiedGroupKFold(n_splits=n_outer_splits, shuffle=True,
                                 random_state=random_state)
    y_true_all, y_score_all, fold_results = [], [], []

    for fi, (tr, te) in enumerate(outer.split(X, y, groups=groups)):
        inner = StratifiedGroupKFold(n_splits=n_inner_splits, shuffle=True,
                                     random_state=random_state)
        inner_splits = list(inner.split(X[tr], y[tr], groups=groups[tr]))

        search = GridSearchCV(make_svm_pipeline(), svm_param_grid(),
                              scoring="roc_auc", cv=inner_splits, n_jobs=1)
        search.fit(X[tr], y[tr])

        y_score = search.best_estimator_.predict_proba(X[te])[:, 1]
        y_test = y[te]

        fold_auc = roc_auc_score(y_test, y_score) if len(set(y_test)) == 2 else float("nan")
        fold_results.append({"fold": fi, "auc": fold_auc,
                             "test_subjects": sorted(set(groups[te])),
                             "best_params": search.best_params_})
        y_true_all.append(y_test); y_score_all.append(y_score)

        if verbose:
            print(f"  Fold {fi}: test={sorted(set(groups[te]))}, "
                  f"AUC={fold_auc:.3f}, best={search.best_params_}")

    y_true_all = np.concatenate(y_true_all)
    y_score_all = np.concatenate(y_score_all)
    overall = roc_auc_score(y_true_all, y_score_all)

    return {"overall_auc": overall, "fold_results": fold_results,
            "y_true_all": y_true_all, "y_score_all": y_score_all}


def run_svm_permutation(X, y, groups, n_permutations=100,
                        n_outer_splits=5, n_inner_splits=3,
                        random_state=None, verbose_every=10):
    """
    Full SVM permutation test. Labels shuffled at the subject level.

    Returns dict: true_auc, permuted_aucs, p_value, true_fold_results,
    true_y_true, true_y_score
    """
    if random_state is None:
        random_state = config.RANDOM_STATE

    unique_groups = np.unique(groups)
    group_to_label = {}
    for g in unique_groups:
        labs = y[groups == g]
        assert len(set(labs)) == 1, f"Subject {g} has inconsistent labels"
        group_to_label[g] = labs[0]

    print("Running true (unshuffled) SVM nested CV...")
    true_res = svm_nested_cv_auc(X, y, groups, n_outer_splits, n_inner_splits,
                                 random_state, verbose=True)
    true_auc = true_res["overall_auc"]
    print(f"True overall AUC: {true_auc:.3f}\n")

    permuted = np.zeros(n_permutations)
    print(f"Running {n_permutations} permutations...")
    for i in range(n_permutations):
        rng = np.random.RandomState(random_state + i)
        shuffled = rng.permutation(unique_groups)
        smap = dict(zip(unique_groups, [group_to_label[g] for g in shuffled]))
        y_perm = np.array([smap[g] for g in groups])

        res = svm_nested_cv_auc(X, y_perm, groups, n_outer_splits,
                                n_inner_splits, random_state, verbose=False)
        permuted[i] = res["overall_auc"]

        if (i + 1) % verbose_every == 0:
            running_p = float(np.mean(permuted[:i+1] >= true_auc))
            print(f"  {i+1}/{n_permutations} done | running p = {running_p:.3f}")

    p_value = float(np.mean(permuted >= true_auc))
    print(f"\nSVM: AUC = {true_auc:.3f}, p = {p_value:.4f}")

    return {"true_auc": true_auc, "permuted_aucs": permuted, "p_value": p_value,
            "true_fold_results": true_res["fold_results"],
            "true_y_true": true_res["y_true_all"],
            "true_y_score": true_res["y_score_all"]}