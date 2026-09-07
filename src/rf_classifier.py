"""
src/rf_classifier.py
====================
Standalone Random Forest classifier with subject-grouped nested cross-
validation and permutation testing.

This is the third model class in the comparison, alongside the linear
ElasticNet and the non-linear RBF-SVM. A Random Forest is an ensemble of
decision trees, each trained on a bootstrap sample of the data and a random
subset of features. It captures non-linear relationships and feature
interactions (which the linear ElasticNet cannot), while its ensembling and
built-in feature subsampling make it more robust to overfitting on small
samples than a single flexible model such as an RBF-SVM.

It also yields interpretable feature importances.

Evaluation framework is identical to the ElasticNet and SVM analyses
(StratifiedGroupKFold on both loops, pooled out-of-fold AUC, subject-level
label permutation), so results are directly comparable.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config


def make_rf_pipeline():
    """
    Random Forest pipeline. StandardScaler is not strictly required for trees
    (they are scale-invariant) but is included for pipeline consistency and
    harmlessness.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",      # handle MDD/HC imbalance
            random_state=config.RANDOM_STATE,
            n_jobs=1,
        )),
    ])


def rf_param_grid():
    """
    Modest grid tuned for small-sample robustness: shallower trees and larger
    leaf sizes regularise the forest, reducing overfitting.
    """
    return {
        "clf__max_depth": [3, 5, None],
        "clf__min_samples_leaf": [1, 5, 20],
    }


def rf_nested_cv_auc(X, y, groups, n_outer_splits=5, n_inner_splits=3,
                     random_state=None, verbose=True):
    """Subject-grouped nested CV with a Random Forest. Returns pooled AUC."""
    if random_state is None:
        random_state = config.RANDOM_STATE

    outer = StratifiedGroupKFold(n_splits=n_outer_splits, shuffle=True,
                                 random_state=random_state)
    y_true_all, y_score_all, fold_results = [], [], []

    for fi, (tr, te) in enumerate(outer.split(X, y, groups=groups)):
        inner = StratifiedGroupKFold(n_splits=n_inner_splits, shuffle=True,
                                     random_state=random_state)
        inner_splits = list(inner.split(X[tr], y[tr], groups=groups[tr]))

        search = GridSearchCV(make_rf_pipeline(), rf_param_grid(),
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
            print(f"  Fold {fi}: AUC={fold_auc:.3f}, best={search.best_params_}")

    y_true_all = np.concatenate(y_true_all)
    y_score_all = np.concatenate(y_score_all)
    overall = roc_auc_score(y_true_all, y_score_all)

    return {"overall_auc": overall, "fold_results": fold_results,
            "y_true_all": y_true_all, "y_score_all": y_score_all}


def run_rf_permutation(X, y, groups, n_permutations=100,
                       n_outer_splits=5, n_inner_splits=3,
                       random_state=None, verbose_every=10):
    """Full RF permutation test. Labels shuffled at the subject level."""
    if random_state is None:
        random_state = config.RANDOM_STATE

    unique_groups = np.unique(groups)
    group_to_label = {}
    for g in unique_groups:
        labs = y[groups == g]
        assert len(set(labs)) == 1, f"Subject {g} has inconsistent labels"
        group_to_label[g] = labs[0]

    print("Running true (unshuffled) Random Forest nested CV...")
    true_res = rf_nested_cv_auc(X, y, groups, n_outer_splits, n_inner_splits,
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

        res = rf_nested_cv_auc(X, y_perm, groups, n_outer_splits,
                               n_inner_splits, random_state, verbose=False)
        permuted[i] = res["overall_auc"]

        if (i + 1) % verbose_every == 0:
            running_p = float(np.mean(permuted[:i+1] >= true_auc))
            print(f"  {i+1}/{n_permutations} done | running p = {running_p:.3f}")

    p_value = float(np.mean(permuted >= true_auc))
    print(f"\nRandom Forest: AUC = {true_auc:.3f}, p = {p_value:.4f}")

    return {"true_auc": true_auc, "permuted_aucs": permuted, "p_value": p_value,
            "true_fold_results": true_res["fold_results"],
            "true_y_true": true_res["y_true_all"],
            "true_y_score": true_res["y_score_all"]}