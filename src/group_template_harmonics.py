"""
src/group_template_harmonics.py
===============================
Group-template graph harmonics with leakage-free cross-validation.

Motivation
----------
Subject-specific harmonics are not directly comparable across individuals: mode
k in one subject is a different spatial pattern from mode k in another, because
each basis is derived from that subject's own connectivity. A shared basis makes
mode indices comparable.

Leakage control
---------------
The shared basis must be derived from TRAINING subjects only. If the template is
built from all subjects, held-out subjects have contributed to defining the
feature space, which inflates performance estimates. This module therefore
rebuilds the template inside every cross-validation fold.

Cost: per-subject connectivity is computed once and cached; only the averaging,
one eigendecomposition (104x104), and the projections are repeated per fold.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score


# ── Step 1: per-subject connectivity (computed once, reused across folds) ──────

def compute_subject_connectivity(epochs_dict, label_map, fmin, fmax,
                                 compute_wpli_fn):
    """
    Compute one connectivity matrix per subject, plus keep the epoch data needed
    for later projection.

    Returns
    -------
    conn : dict {sid: (n_ch, n_ch) ndarray}
    data : dict {sid: (n_epochs, n_ch, n_times) ndarray}
    chans : list[str]   channel names (assumed consistent across subjects)
    labels : dict {sid: 0/1}
    """
    conn, data, labels = {}, {}, {}
    chans = None
    for sid, ep in epochs_dict.items():
        lab = label_map.get(sid)
        if lab not in ("HC", "MDD") or len(ep) < 2:
            continue
        w, ch = compute_wpli_fn(ep, fmin=fmin, fmax=fmax)
        if chans is None:
            chans = ch
        conn[sid] = w
        data[sid] = ep.copy().pick(ch).get_data()
        labels[sid] = 1 if lab == "MDD" else 0
    return conn, data, chans, labels


# ── Step 2: build a harmonic basis from a set of connectivity matrices ─────────

def build_template_basis(conn_matrices, n_modes=20):
    """
    Average the supplied connectivity matrices, form the normalised Laplacian,
    and return the n_modes lowest-order eigenvectors.

    conn_matrices : list of (n_ch, n_ch) arrays  -- TRAINING SUBJECTS ONLY
    """
    W = np.mean(conn_matrices, axis=0)
    np.fill_diagonal(W, 0.0)
    deg = W.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    L = np.eye(W.shape[0]) - d_inv_sqrt @ W @ d_inv_sqrt
    eigvals, eigvecs = np.linalg.eigh(L)          # ascending by construction
    return eigvals[:n_modes], eigvecs[:, :n_modes]


# ── Step 3: project a subject's data onto a given basis ───────────────────────

def project_to_basis(subject_data, eigvecs):
    """
    subject_data : (n_epochs, n_ch, n_times)
    eigvecs      : (n_ch, n_modes)

    Returns per-epoch power: (n_epochs, n_modes)
    """
    # mode time-courses: (n_epochs, n_modes, n_times)
    tc = np.einsum("mc,ect->emt", eigvecs.T, subject_data)
    return (tc ** 2).mean(axis=2)                  # power = mean squared amplitude


# ── Step 4: leakage-free cross-validation ─────────────────────────────────────

def evaluate_group_template(conn, data, labels, n_modes=20, n_splits=5,
                            aggregate="subject", random_state=42, verbose=True):
    """
    Cross-validate with the harmonic template rebuilt inside each fold.

    aggregate : "subject" -> one feature vector per subject (mean over epochs)
                "epoch"   -> per-epoch features, subject-grouped CV

    Returns dict with pooled y_true, y_score and per-fold AUCs.
    """
    sids = np.array(sorted(conn.keys()))
    y_subj = np.array([labels[s] for s in sids])

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                              random_state=random_state)

    y_true_all, y_score_all, fold_aucs = [], [], []

    for fold, (tr_idx, te_idx) in enumerate(cv.split(sids, y_subj, groups=sids)):
        tr_sids, te_sids = sids[tr_idx], sids[te_idx]

        # --- template from TRAINING subjects only (the leakage-critical step) ---
        _, eigvecs = build_template_basis([conn[s] for s in tr_sids],
                                          n_modes=n_modes)

        # --- project every subject onto this fold's basis ---
        def features_for(sid_list):
            X, y, g = [], [], []
            for s in sid_list:
                p = project_to_basis(data[s], eigvecs)      # (n_epochs, n_modes)
                if aggregate == "subject":
                    X.append(p.mean(axis=0)); y.append(labels[s]); g.append(s)
                else:
                    X.append(p)
                    y.append(np.full(p.shape[0], labels[s]))
                    g.append(np.full(p.shape[0], s, dtype=object))
            if aggregate == "subject":
                return np.array(X), np.array(y), np.array(g)
            return np.vstack(X), np.concatenate(y), np.concatenate(g)

        X_tr, y_tr, g_tr = features_for(tr_sids)
        X_te, y_te, g_te = features_for(te_sids)

        # --- inner-loop tuning on training data only ---
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True,
                                     random_state=random_state)
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(penalty="elasticnet", solver="saga",
                                       max_iter=20000, tol=1e-3)),
        ])
        grid = {"clf__C": [0.01, 0.1, 1.0], "clf__l1_ratio": [0.1, 0.9]}
        gs = GridSearchCV(pipe, grid, scoring="roc_auc",
                          cv=list(inner.split(X_tr, y_tr, groups=g_tr)),
                          n_jobs=1, error_score=0.5)
        gs.fit(X_tr, y_tr)

        scores = gs.best_estimator_.predict_proba(X_te)[:, 1]
        y_true_all.extend(y_te); y_score_all.extend(scores)

        if len(np.unique(y_te)) > 1:
            fa = roc_auc_score(y_te, scores)
            fold_aucs.append(fa)
            if verbose:
                print(f"  Fold {fold}: n_test={len(te_sids)} subjects, "
                      f"AUC={fa:.3f}, params={gs.best_params_}")

    y_true_all = np.array(y_true_all); y_score_all = np.array(y_score_all)
    return {
        "y_true": y_true_all,
        "y_score": y_score_all,
        "overall_auc": roc_auc_score(y_true_all, y_score_all),
        "fold_aucs": fold_aucs,
    }


# ── Optional: template stability check (descriptive, run outside CV) ───────────

def template_stability(conn, n_modes=20, n_splits=5, random_state=42):
    """
    How similar are the templates built from different subject subsets?
    Returns mean absolute spatial correlation of matched modes across folds.

    A high value indicates the template is stable and the fold-wise rebuild
    changes little; a low value indicates the basis itself is sample-dependent,
    which is worth reporting.
    """
    sids = np.array(sorted(conn.keys()))
    rng = np.random.default_rng(random_state)
    bases = []
    for _ in range(n_splits):
        subset = rng.choice(sids, size=int(len(sids) * 0.8), replace=False)
        _, ev = build_template_basis([conn[s] for s in subset], n_modes=n_modes)
        bases.append(ev)

    sims = []
    for i in range(len(bases)):
        for j in range(i + 1, len(bases)):
            # |correlation| per mode, to absorb arbitrary eigenvector sign
            per_mode = [abs(np.corrcoef(bases[i][:, m], bases[j][:, m])[0, 1])
                        for m in range(n_modes)]
            sims.append(np.mean(per_mode))
    return float(np.mean(sims)), float(np.std(sims))