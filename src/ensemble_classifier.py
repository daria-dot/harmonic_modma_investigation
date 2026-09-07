"""
src/ensemble_classifier.py
==========================
Feature-selecting stacking ensemble, evaluated with subject-grouped nested CV
and permutation testing.

This addresses the "pick the best features and models and combine them" goal in
a leakage-free way. The critical design point: feature selection and all model
fitting happen INSIDE each training fold, never by inspecting test-set scores
across the project. The ensemble is therefore a single, honestly-evaluated
method, not a post-hoc combination of analyses that happened to score well
(which would be selection bias).

Pipeline (all refit per fold):
  1. StandardScaler
  2. SelectKBest  -- picks the k most informative features on the TRAINING fold
  3. StackingClassifier -- ElasticNet + RandomForest + GradientBoosting base
     learners, combined by a logistic-regression meta-learner

Because selection and stacking are inside the sklearn Pipeline, they are refit
on each training fold and applied unchanged to the held-out fold, so there is
no data leakage.
"""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                              StackingClassifier)
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

import config


def make_ensemble_pipeline(k_features=10):
    base_learners = [
        ("enet", LogisticRegression(penalty="elasticnet", solver="saga",
                 l1_ratio=0.5, C=0.1, max_iter=5000, tol=1e-3)),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=3,
                class_weight="balanced", random_state=config.RANDOM_STATE)),
        ("gb", HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05,
                random_state=config.RANDOM_STATE)),
    ]
    stack = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3, n_jobs=1,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k_features)),
        ("ensemble", stack),
    ])


def ensemble_param_grid(max_k):
    ks = [k for k in (5, 10, 20) if k <= max_k]
    if not ks:
        ks = [max_k]
    return {"select__k": ks}


def ensemble_nested_cv_auc(X, y, groups, n_outer_splits=5, n_inner_splits=3,
                           random_state=None, verbose=True):
    if random_state is None:
        random_state = config.RANDOM_STATE
    outer = StratifiedGroupKFold(n_splits=n_outer_splits, shuffle=True,
                                 random_state=random_state)
    yt_all, ys_all, folds = [], [], []
    for fi, (tr, te) in enumerate(outer.split(X, y, groups=groups)):
        inner = StratifiedGroupKFold(n_splits=n_inner_splits, shuffle=True,
                                     random_state=random_state)
        isplits = list(inner.split(X[tr], y[tr], groups=groups[tr]))
        gs = GridSearchCV(make_ensemble_pipeline(), ensemble_param_grid(X.shape[1]),
                          scoring="roc_auc", cv=isplits, n_jobs=1, error_score=0.5)
        gs.fit(X[tr], y[tr])
        ys_te = gs.best_estimator_.predict_proba(X[te])[:, 1]
        yt_te = y[te]
        auc = roc_auc_score(yt_te, ys_te) if len(set(yt_te)) == 2 else float("nan")
        folds.append({"fold": fi, "auc": auc, "best_k": gs.best_params_["select__k"]})
        yt_all.append(yt_te); ys_all.append(ys_te)
        if verbose:
            print(f"  Fold {fi}: AUC={auc:.3f}, best_k={gs.best_params_['select__k']}")
    yt_all = np.concatenate(yt_all); ys_all = np.concatenate(ys_all)
    return {"overall_auc": roc_auc_score(yt_all, ys_all), "fold_results": folds,
            "y_true_all": yt_all, "y_score_all": ys_all}


def run_ensemble_permutation(X, y, groups, n_permutations=50,
                             n_outer_splits=5, n_inner_splits=3, random_state=None,
                             verbose_every=5):
    if random_state is None:
        random_state = config.RANDOM_STATE
    unique_groups = np.unique(groups)
    g2l = {}
    for g in unique_groups:
        labs = y[groups == g]
        assert len(set(labs)) == 1
        g2l[g] = labs[0]

    print("Running true (unshuffled) ensemble nested CV...")
    true = ensemble_nested_cv_auc(X, y, groups, n_outer_splits, n_inner_splits,
                                  random_state, verbose=True)
    true_auc = true["overall_auc"]
    print(f"True overall AUC: {true_auc:.3f}\n")

    permuted = np.zeros(n_permutations)
    print(f"Running {n_permutations} permutations...")
    for i in range(n_permutations):
        rng = np.random.RandomState(random_state + i)
        shuffled = rng.permutation(unique_groups)
        smap = dict(zip(unique_groups, [g2l[g] for g in shuffled]))
        yp = np.array([smap[g] for g in groups])
        res = ensemble_nested_cv_auc(X, yp, groups, n_outer_splits, n_inner_splits,
                                     random_state, verbose=False)
        permuted[i] = res["overall_auc"]
        if (i + 1) % verbose_every == 0:
            print(f"  {i+1}/{n_permutations} | running p={np.mean(permuted[:i+1]>=true_auc):.3f}")

    p = float(np.mean(permuted >= true_auc))
    print(f"\nEnsemble: AUC={true_auc:.3f}, p={p:.4f}")
    return {"true_auc": true_auc, "permuted_aucs": permuted, "p_value": p,
            "true_fold_results": true["fold_results"],
            "true_y_true": true["y_true_all"], "true_y_score": true["y_score_all"]}