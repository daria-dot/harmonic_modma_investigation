"""
src/subspace_features.py
========================
Rotation-invariant harmonic features, motivated by the spectral diagnostics.

Background
----------
The eigenvalue spectrum of the group Laplacian shows a small number of
well-separated low-order modes followed by a large near-degenerate plateau.
Where eigenvalues are degenerate, individual eigenvectors are not determined:
they rotate freely within the degenerate block under small perturbations of
the data. Per-mode power is therefore an unstable parameterisation of the
signal in that region -- confirmed empirically, with matched-mode similarity
of 0.55 against subspace similarity of 0.85 across subject subsamples.

Two parameterisations follow from this:

  1. BLOCK ENERGY -- total projected power within each near-degenerate block.
     Invariant to rotation inside the block, therefore stable wherever the
     subspace is stable.

  2. SEPARATED MODES -- individual per-mode power, but restricted to the
     leading modes whose eigenvalue gaps are large enough that the modes are
     individually determined.

Both are computed with the harmonic basis rebuilt inside each cross-validation
fold from training subjects only.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# ── Basis construction ────────────────────────────────────────────────────────

def build_basis(conn_matrices, n_modes=None):
    W = np.mean(conn_matrices, axis=0)
    np.fill_diagonal(W, 0.0)
    deg = W.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    L = np.eye(W.shape[0]) - d_inv_sqrt @ W @ d_inv_sqrt
    eigvals, eigvecs = np.linalg.eigh(L)
    if n_modes is not None:
        eigvals, eigvecs = eigvals[:n_modes], eigvecs[:, :n_modes]
    return eigvals, eigvecs


def blocks_from_spectrum(eigvals, rel_gap_threshold=3.0, max_mode=None):
    """
    Partition mode indices into blocks separated by relative eigenvalue gaps
    exceeding the threshold. Modes within a block are treated as degenerate.
    """
    if max_mode is not None:
        eigvals = eigvals[:max_mode]
    gaps = np.diff(eigvals)
    pos = gaps[gaps > 0]
    scale = np.median(pos) if len(pos) else 1.0
    rel = gaps / scale

    blocks, start = [], 0
    for i, r in enumerate(rel):
        if r > rel_gap_threshold:
            blocks.append((start, i))
            start = i + 1
    blocks.append((start, len(eigvals) - 1))
    return blocks


def n_separated_modes(eigvals, rel_gap_threshold=10.0, max_check=30):
    """
    How many leading modes are individually well determined?

    Counts consecutive modes from the start whose following gap exceeds the
    threshold. Stops at the first mode that merges into the plateau.
    """
    gaps = np.diff(eigvals[:max_check + 1])
    pos = gaps[gaps > 0]
    scale = np.median(pos) if len(pos) else 1.0
    rel = gaps / scale
    n = 0
    for r in rel:
        if r > rel_gap_threshold:
            n += 1
        else:
            break
    return max(n, 1)


# ── Feature extraction ────────────────────────────────────────────────────────

def project_power(subject_data, eigvecs):
    """Per-epoch power on each mode. Returns (n_epochs, n_modes)."""
    tc = np.einsum("mc,ect->emt", eigvecs.T, subject_data)
    return (tc ** 2).mean(axis=2)


def block_energy_features(per_mode_power, blocks, normalise=True):
    """
    Sum per-mode power within each block.

    normalise: express each block's energy as a fraction of total harmonic
    power, which removes overall amplitude differences between subjects and
    isolates how energy is *distributed* across spatial scales.
    """
    feats = np.stack([per_mode_power[:, lo:hi + 1].sum(axis=1)
                      for lo, hi in blocks], axis=1)
    if normalise:
        total = feats.sum(axis=1, keepdims=True)
        feats = feats / np.maximum(total, 1e-30)
    return feats


# ── Leakage-free evaluation ───────────────────────────────────────────────────

def evaluate(conn, data, labels, mode="block_energy", n_modes=40,
             rel_gap_threshold=3.0, sep_gap_threshold=10.0,
             normalise=True, n_splits=5, random_state=42, verbose=True):
    """
    Cross-validate with the basis, blocks, and features all derived inside
    each fold from training subjects only.

    mode : "block_energy"    -- rotation-invariant block energies
           "separated_modes" -- per-mode power, leading well-separated modes only
    """
    sids = np.array(sorted(conn.keys()))
    y_subj = np.array([labels[s] for s in sids])
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                              random_state=random_state)

    y_true_all, y_score_all, fold_aucs, n_feats = [], [], [], []

    for fold, (tr_idx, te_idx) in enumerate(cv.split(sids, y_subj, groups=sids)):
        tr_sids, te_sids = sids[tr_idx], sids[te_idx]

        # basis and block structure from TRAINING subjects only
        eigvals, eigvecs = build_basis([conn[s] for s in tr_sids], n_modes=n_modes)

        if mode == "block_energy":
            blocks = blocks_from_spectrum(eigvals, rel_gap_threshold)
            def featurise(p):
                return block_energy_features(p, blocks, normalise=normalise)
        elif mode == "separated_modes":
            k = n_separated_modes(eigvals, sep_gap_threshold)
            def featurise(p):
                f = p[:, :k]
                if normalise:
                    f = f / np.maximum(p.sum(axis=1, keepdims=True), 1e-30)
                return f
        else:
            raise ValueError(f"unknown mode: {mode}")

        def build(sid_list):
            X, y = [], []
            for s in sid_list:
                p = project_power(data[s], eigvecs)
                X.append(featurise(p).mean(axis=0))
                y.append(labels[s])
            return np.array(X), np.array(y)

        X_tr, y_tr = build(tr_sids)
        X_te, y_te = build(te_sids)
        n_feats.append(X_tr.shape[1])

        inner = StratifiedGroupKFold(n_splits=3, shuffle=True,
                                     random_state=random_state)
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(penalty="elasticnet", solver="saga",
                                       max_iter=20000, tol=1e-3)),
        ])
        gs = GridSearchCV(pipe, {"clf__C": [0.01, 0.1, 1.0],
                                 "clf__l1_ratio": [0.1, 0.9]},
                          scoring="roc_auc",
                          cv=list(inner.split(X_tr, y_tr, groups=tr_sids)),
                          n_jobs=1, error_score=0.5)
        gs.fit(X_tr, y_tr)

        scores = gs.best_estimator_.predict_proba(X_te)[:, 1]
        y_true_all.extend(y_te); y_score_all.extend(scores)
        if len(np.unique(y_te)) > 1:
            fa = roc_auc_score(y_te, scores)
            fold_aucs.append(fa)
            if verbose:
                print(f"  Fold {fold}: {X_tr.shape[1]} features, AUC={fa:.3f}")

    y_true_all = np.array(y_true_all); y_score_all = np.array(y_score_all)
    return {"y_true": y_true_all, "y_score": y_score_all,
            "overall_auc": roc_auc_score(y_true_all, y_score_all),
            "fold_aucs": fold_aucs,
            "n_features": (float(np.mean(n_feats)), n_feats)}


# ── Descriptive: energy distribution across spatial scales ────────────────────

def energy_distribution_by_group(conn, data, labels, n_modes=40,
                                 rel_gap_threshold=3.0):
    """
    Group comparison of relative energy per block, using a basis built from
    ALL subjects. Descriptive only -- no generalisation claim, so the
    leakage concern does not apply.
    """
    from scipy import stats

    eigvals, eigvecs = build_basis([conn[s] for s in conn], n_modes=n_modes)
    blocks = blocks_from_spectrum(eigvals, rel_gap_threshold)

    rows = {0: [], 1: []}
    for sid in conn:
        p = project_power(data[sid], eigvecs)
        rows[labels[sid]].append(block_energy_features(p, blocks).mean(axis=0))

    hc = np.array(rows[0]); mdd = np.array(rows[1])
    out = []
    for i, (lo, hi) in enumerate(blocks):
        t, pv = stats.ttest_ind(mdd[:, i], hc[:, i])
        out.append({"block": (lo, hi), "size": hi - lo + 1,
                    "lambda_range": (float(eigvals[lo]), float(eigvals[hi])),
                    "mdd_mean": float(mdd[:, i].mean()),
                    "hc_mean": float(hc[:, i].mean()),
                    "t": float(t), "p": float(pv)})
    return out, blocks, eigvals