"""
src/geometric_harmonics.py
==========================
A subject-independent harmonic basis derived from electrode geometry.

Motivation
----------
Functional harmonics are estimated from each subject's (or each training
fold's) connectivity, and are therefore sample-dependent. Diagnostics on this
dataset showed matched-mode similarity of only 0.55 across subject subsamples,
with rotation-invariant subspace similarity of 0.85 -- indicating that
individual mode coordinates are poorly determined beyond the leading few.

This module removes that dependence entirely by building the graph from the
electrode montage rather than from data. The adjacency is a Gaussian kernel on
inter-electrode distance, so the Laplacian, its eigenvectors, and hence the
harmonic basis are identical for every subject and every fold. There is no
estimation variance and no leakage concern, because no participant data enters
the basis construction.

Limitation
----------
The resulting modes describe the spatial geometry of the sensor array, not
cortical anatomy. They are not connectome harmonics in the sense of Atasoy et
al., where the graph encodes white-matter connectivity. A properly anatomical
alternative is to forward-project cortical Laplace-Beltrami eigenmodes through
a head model, which retains subject-independence while respecting cortical
geometry and volume conduction; that is noted as future work.
"""

from __future__ import annotations

import numpy as np


# ── Graph construction from electrode geometry ────────────────────────────────

def positions_from_info(info, ch_names=None):
    """
    Extract 3D electrode positions (metres) from an MNE Info object.

    Returns (n_ch, 3) array in the order of ch_names, or of info's EEG picks
    if ch_names is None.
    """
    import mne
    if ch_names is None:
        picks = mne.pick_types(info, eeg=True)
        ch_names = [info["ch_names"][i] for i in picks]
    montage_pos = {ch["ch_name"]: ch["loc"][:3] for ch in info["chs"]}
    pos = np.array([montage_pos[c] for c in ch_names], dtype=float)
    if not np.isfinite(pos).all() or np.allclose(pos, 0):
        raise ValueError("Electrode positions missing or degenerate -- "
                         "check that a montage has been set on the data.")
    return pos, ch_names


def geometric_adjacency(pos, sigma=None, knn=None, self_loops=False):
    """
    Gaussian-kernel adjacency on inter-electrode distance:
        W_ij = exp(-d_ij^2 / (2 sigma^2))

    sigma : kernel width. If None, set to the median nearest-neighbour
            distance, which adapts the kernel to the montage density.
    knn   : if given, retain only each node's k nearest neighbours
            (symmetrised), producing a sparser, more local graph.
    """
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)

    if sigma is None:
        # nearest-neighbour distance per node, excluding self
        d_off = d.copy()
        np.fill_diagonal(d_off, np.inf)
        nn = d_off.min(axis=1)
        nn = nn[np.isfinite(nn)]
        if len(nn) == 0 or not np.all(nn > 0):
            raise ValueError("Degenerate electrode positions: cannot estimate sigma.")
        sigma = float(np.median(nn))

    W = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
    if not self_loops:
        np.fill_diagonal(W, 0.0)

    if knn is not None:
        keep = np.zeros_like(W, dtype=bool)
        for i in range(len(W)):
            idx = np.argsort(W[i])[::-1][:knn]
            keep[i, idx] = True
        keep = keep | keep.T          # symmetrise
        W = W * keep

    return W, sigma


def geometric_basis(pos, n_modes=None, sigma=None, knn=None):
    """
    Normalised graph Laplacian of the geometric adjacency, and its
    eigendecomposition. Identical for every subject.
    """
    W, sigma_used = geometric_adjacency(pos, sigma=sigma, knn=knn)
    deg = W.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    L = np.eye(W.shape[0]) - d_inv_sqrt @ W @ d_inv_sqrt
    eigvals, eigvecs = np.linalg.eigh(L)
    if n_modes is not None:
        eigvals, eigvecs = eigvals[:n_modes], eigvecs[:, :n_modes]
    return eigvals, eigvecs, W, sigma_used


# ── Feature extraction (basis is fixed, so no fold-wise rebuild needed) ───────

def project_power(subject_data, eigvecs):
    """Per-epoch power on each mode. subject_data: (n_epochs, n_ch, n_times)."""
    tc = np.einsum("mc,ect->emt", eigvecs.T, subject_data)
    return (tc ** 2).mean(axis=2)


def build_feature_matrix(data, labels, eigvecs, normalise=True,
                         aggregate="subject"):
    """
    Features for every subject on the fixed basis.

    normalise : express each mode's power as a fraction of total, isolating
                the distribution of energy across spatial scales rather than
                overall amplitude.
    """
    X, y, g = [], [], []
    for sid in sorted(data.keys()):
        p = project_power(data[sid], eigvecs)
        if normalise:
            p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-30)
        if aggregate == "subject":
            X.append(p.mean(axis=0)); y.append(labels[sid]); g.append(sid)
        else:
            X.append(p)
            y.append(np.full(p.shape[0], labels[sid]))
            g.append(np.full(p.shape[0], sid, dtype=object))
    if aggregate == "subject":
        return np.array(X), np.array(y), np.array(g)
    return np.vstack(X), np.concatenate(y), np.concatenate(g)


# ── Descriptive group comparison ──────────────────────────────────────────────

def compare_groups(X, y, eigvals=None):
    """
    Per-mode group comparison with Benjamini-Hochberg correction.
    Returns a list of dicts, one per mode.
    """
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    rows, pvals = [], []
    for k in range(X.shape[1]):
        a, b = X[y == 1, k], X[y == 0, k]
        t, p = stats.ttest_ind(a, b)
        rows.append({"mode": k,
                     "lambda": float(eigvals[k]) if eigvals is not None else None,
                     "mdd_mean": float(a.mean()), "hc_mean": float(b.mean()),
                     "t": float(t), "p": float(p)})
        pvals.append(p)

    if pvals:
        _, p_fdr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
        for r, pf in zip(rows, p_fdr):
            r["p_fdr"] = float(pf)
            r["sig_fdr"] = bool(pf < 0.05)
    return rows