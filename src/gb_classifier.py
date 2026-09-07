"""
src/gb_classifier.py
====================
Standalone gradient-boosting classifier with subject-grouped nested cross-
validation and permutation testing.

Fourth model class in the comparison (after linear ElasticNet, kernel RBF-SVM,
and Random Forest). Gradient boosting builds decision trees sequentially, each
correcting the residual errors of the ensemble so far. This can capture weak
non-linear interactions that parallel-tree methods (Random Forest) average
away, at the cost of a higher tendency to overfit on small samples -- so on a
53-subject dataset a near-chance or below-chance result would be unsurprising.

Uses HistGradientBoostingClassifier (histogram-based, substantially faster than
the classic GradientBoostingClassifier), which matters for the permutation test
runtime. Evaluation framework is identical to the other three models.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config


def make_gb_pipeline():
    """Histogram-based gradient boosting. Scaling is unnecessary for trees but
    kept for pipeline consistency and harmlessness."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            random_state=config.RANDOM_STATE,
            early_stopping=True,        # guards against overfitting via internal validation
            validation_fraction=0.15,
        )),
    ])


def gb_param_grid():
    """Small grid emphasising regularisation for small-sample robustness:
    shallow trees, slower learning rates, larger leaves."""
    return {
        "clf__max_depth": [2, 3],
        "clf__learning_rate": [0.03, 0.1],
        "clf__min_samples_leaf": [20, 50],
    }


def gb_nested_cv_auc(X, y, groups, n_outer_splits=5, n_inner_splits=3,
                     random_state=None, verbose=True):
    """Subject-grouped nested CV with gradient boosting. Returns pooled AUC."""
    if random_state is None:
        random_state = config.RANDOM_STATE

    outer = StratifiedGroupKFold(n_splits=n_outer_splits, shuffle=True,
                                 random_state=random_state)
    y_true_all, y_score_all, fold_results = [], [], []

    for fi, (tr, te) in enumerate(outer.split(X, y, groups=groups)):
        inner = StratifiedGroupKFold(n_splits=n_inner_splits, shuffle=True,
                                     random_state=random_state)
        inner_splits = list(inner.split(X[tr], y[tr], groups=groups[tr]))

        search = GridSearchCV(make_gb_pipeline(), gb_param_grid(),
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
    return {"overall_auc": roc_auc_score(y_true_all, y_score_all),
            "fold_results": fold_results,
            "y_true_all": y_true_all, "y_score_all": y_score_all}


def run_gb_permutation(X, y, groups, n_permutations=50,
                       n_outer_splits=5, n_inner_splits=3,
                       random_state=None, verbose_every=5):
    """Full gradient-boosting permutation test. Labels shuffled per subject."""
    if random_state is None:
        random_state = config.RANDOM_STATE

    unique_groups = np.unique(groups)
    group_to_label = {}
    for g in unique_groups:
        labs = y[groups == g]
        assert len(set(labs)) == 1, f"Subject {g} inconsistent labels"
        group_to_label[g] = labs[0]

    print("Running true (unshuffled) gradient-boosting nested CV...")
    true_res = gb_nested_cv_auc(X, y, groups, n_outer_splits, n_inner_splits,
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

        res = gb_nested_cv_auc(X, y_perm, groups, n_outer_splits,
                               n_inner_splits, random_state, verbose=False)
        permuted[i] = res["overall_auc"]

        if (i + 1) % verbose_every == 0:
            running_p = float(np.mean(permuted[:i+1] >= true_auc))
            print(f"  {i+1}/{n_permutations} done | running p = {running_p:.3f}")

    p_value = float(np.mean(permuted >= true_auc))
    print(f"\nGradient boosting: AUC = {true_auc:.3f}, p = {p_value:.4f}")

    return {"true_auc": true_auc, "permuted_aucs": permuted, "p_value": p_value,
            "true_fold_results": true_res["fold_results"],
            "true_y_true": true_res["y_true_all"],
            "true_y_score": true_res["y_score_all"]}