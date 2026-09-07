"""
src/spectral_diagnostics.py
===========================
Diagnostics for the harmonic eigenspace, addressing a specific concern:

matched-mode correlation understates stability when eigenvalues are
near-degenerate. Where several eigenvalues cluster, the individual
eigenvectors are not well determined -- small perturbations rotate them
freely within the degenerate block -- but the SUBSPACE they jointly span
may be perfectly stable.

This module therefore measures:
  1. the eigenvalue spectrum and its gap structure, to locate degenerate blocks
  2. subspace stability via principal angles, which is invariant to rotation
     within a subspace and is the correct comparison when modes are degenerate
  3. matched-mode correlation, for direct contrast with (2)
"""

from __future__ import annotations

import numpy as np


# ── Basis construction (shared with the group-template pipeline) ──────────────

def build_basis(conn_matrices, n_modes=None):
    """Average connectivity, normalised Laplacian, eigendecomposition."""
    W = np.mean(conn_matrices, axis=0)
    np.fill_diagonal(W, 0.0)
    deg = W.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    L = np.eye(W.shape[0]) - d_inv_sqrt @ W @ d_inv_sqrt
    eigvals, eigvecs = np.linalg.eigh(L)
    if n_modes is not None:
        eigvals, eigvecs = eigvals[:n_modes], eigvecs[:, :n_modes]
    return eigvals, eigvecs


# ── Step 1: spectrum and gap structure ────────────────────────────────────────

def spectrum_gaps(eigvals, n_report=25):
    """
    Consecutive eigenvalue gaps, and a relative gap that accounts for the
    overall scale of the spectrum.

    Large relative gaps mark boundaries between well-separated blocks;
    runs of small gaps mark near-degenerate blocks whose individual
    eigenvectors are not individually meaningful.
    """
    gaps = np.diff(eigvals)
    scale = np.median(gaps[gaps > 0]) if np.any(gaps > 0) else 1.0
    rel_gaps = gaps / scale

    order = np.argsort(rel_gaps)[::-1][:n_report]
    top = [(int(i), float(eigvals[i]), float(eigvals[i + 1]), float(rel_gaps[i]))
           for i in sorted(order)]
    return {"gaps": gaps, "rel_gaps": rel_gaps, "median_gap": float(scale),
            "largest_gaps": top}


def identify_blocks(eigvals, rel_gap_threshold=3.0):
    """
    Partition the mode indices into blocks separated by relative gaps above
    threshold. Modes within a block are treated as near-degenerate.

    Returns list of (start, end_inclusive) index pairs.
    """
    gaps = np.diff(eigvals)
    pos = gaps[gaps > 0]
    scale = np.median(pos) if len(pos) else 1.0
    rel = gaps / scale

    boundaries = [i for i, r in enumerate(rel) if r > rel_gap_threshold]
    blocks, start = [], 0
    for b in boundaries:
        blocks.append((start, b))
        start = b + 1
    blocks.append((start, len(eigvals) - 1))
    return blocks


# ── Step 2: subspace stability via principal angles ───────────────────────────

def principal_angles(A, B):
    """
    Principal angles between the column spaces of A and B.

    Both are orthonormalised first; the singular values of Q_A^T Q_B are the
    cosines of the principal angles. Returns angles in radians, ascending.
    """
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(s)


def subspace_similarity(A, B):
    """
    Mean cos^2 of the principal angles -- the standard 'projection' measure of
    subspace overlap. 1.0 = identical subspaces, 0.0 = orthogonal.

    Unlike matched-mode correlation this is invariant to rotation within
    either subspace, so it is unaffected by eigenvalue degeneracy.
    """
    theta = principal_angles(A, B)
    return float(np.mean(np.cos(theta) ** 2))


def matched_mode_similarity(A, B):
    """
    Mean |correlation| between mode k of A and mode k of B, for contrast.
    Sensitive to rotation within degenerate blocks, hence pessimistic.
    """
    sims = [abs(np.corrcoef(A[:, k], B[:, k])[0, 1]) for k in range(A.shape[1])]
    return float(np.mean(sims))


def stability_analysis(conn, n_modes=20, n_repeats=10, subset_frac=0.8,
                       blocks=None, random_state=42):
    """
    Compare bases built from independent random subsets of subjects.

    Reports, across repeat pairs:
      subspace_sim  -- rotation-invariant subspace overlap of the first n_modes
      matched_sim   -- matched-mode correlation (for contrast)
      block_sims    -- subspace overlap computed separately per block, if given

    A large gap between subspace_sim and matched_sim is direct evidence that
    apparent instability reflects degeneracy rather than genuine variability.
    """
    sids = np.array(sorted(conn.keys()))
    rng = np.random.default_rng(random_state)

    bases = []
    for _ in range(n_repeats):
        subset = rng.choice(sids, size=int(len(sids) * subset_frac), replace=False)
        _, ev = build_basis([conn[s] for s in subset], n_modes=n_modes)
        bases.append(ev)

    sub_sims, mat_sims = [], []
    block_sims = {b: [] for b in (blocks or [])}

    for i in range(len(bases)):
        for j in range(i + 1, len(bases)):
            sub_sims.append(subspace_similarity(bases[i], bases[j]))
            mat_sims.append(matched_mode_similarity(bases[i], bases[j]))
            for b in (blocks or []):
                lo, hi = b
                hi = min(hi, n_modes - 1)
                if hi > lo:
                    block_sims[b].append(
                        subspace_similarity(bases[i][:, lo:hi + 1],
                                            bases[j][:, lo:hi + 1]))

    out = {
        "subspace_sim_mean": float(np.mean(sub_sims)),
        "subspace_sim_sd": float(np.std(sub_sims)),
        "matched_sim_mean": float(np.mean(mat_sims)),
        "matched_sim_sd": float(np.std(mat_sims)),
    }
    if blocks:
        out["block_sims"] = {str(b): (float(np.mean(v)), float(np.std(v)))
                             for b, v in block_sims.items() if v}
    return out


def cumulative_subspace_stability(conn, max_modes=30, n_repeats=6,
                                  subset_frac=0.8, random_state=42):
    """
    Subspace similarity as a function of how many leading modes are included.

    If the leading few modes form a stable subspace while later ones do not,
    this curve will start high and decline -- which would justify restricting
    features to the stable portion of the spectrum.
    """
    sids = np.array(sorted(conn.keys()))
    rng = np.random.default_rng(random_state)

    bases = []
    for _ in range(n_repeats):
        subset = rng.choice(sids, size=int(len(sids) * subset_frac), replace=False)
        _, ev = build_basis([conn[s] for s in subset], n_modes=max_modes)
        bases.append(ev)

    curve = []
    for k in range(1, max_modes + 1):
        sims = [subspace_similarity(bases[i][:, :k], bases[j][:, :k])
                for i in range(len(bases)) for j in range(i + 1, len(bases))]
        curve.append((k, float(np.mean(sims)), float(np.std(sims))))
    return curve