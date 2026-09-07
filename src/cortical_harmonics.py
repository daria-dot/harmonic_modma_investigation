"""
src/cortical_harmonics.py
=========================
A subject-independent, anatomically-grounded harmonic basis for sensor-space
EEG, following the forward-projection approach.

Rationale
---------
Harmonics estimated from functional connectivity are sample-dependent: on this
dataset, matched-mode similarity across subject subsamples was only 0.55,
with the eigenvalue spectrum collapsing onto a near-degenerate plateau beyond
the leading few modes. A basis derived from cortical anatomy instead is fixed,
identical for every subject, and free of estimation variance.

Construction
------------
  1. Laplace-Beltrami (LB) eigenmodes on the fsaverage cortical surface,
     via cotangent finite-element discretisation with a lumped mass matrix.
  2. Forward solution (lead field L) for the electrode montage, using a
     three-layer BEM on the same template head.
  3. Sensor dictionary D = L @ Phi -- each column is the scalp topography
     produced by one cortical harmonic.
  4. Projection of scalp data onto D by ordinary least squares.

The resulting modes are ordered by cortical spatial frequency and are
comparable across subjects and studies, since the basis derives from template
anatomy rather than from the data being analysed.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


# ── Laplace-Beltrami operator on a triangulated surface ───────────────────────

def cotangent_laplacian(verts, tris):
    """
    Cotangent-weighted stiffness matrix C and lumped mass matrix M for a
    triangle mesh.

    For each triangle, the weight contributed to edge (i, j) is
    cot(angle opposite that edge) / 2. The mass matrix is the barycentric
    (lumped) area, one third of the incident triangle areas per vertex.

    Returns
    -------
    C : (n_vert, n_vert) sparse   stiffness (positive semi-definite)
    M : (n_vert,) array           lumped mass (diagonal entries)
    """
    verts = np.asarray(verts, dtype=float)
    tris = np.asarray(tris, dtype=int)
    n = len(verts)

    i0, i1, i2 = tris[:, 0], tris[:, 1], tris[:, 2]
    v0, v1, v2 = verts[i0], verts[i1], verts[i2]

    # edge vectors opposite each vertex
    e0 = v2 - v1          # opposite vertex 0
    e1 = v0 - v2          # opposite vertex 1
    e2 = v1 - v0          # opposite vertex 2

    # triangle areas via cross product
    cross = np.cross(e2, -e1)
    area2 = np.linalg.norm(cross, axis=1)          # 2 * area
    area = 0.5 * area2
    area = np.maximum(area, 1e-16)

    # cot(angle at vertex k) = dot(adjacent edges) / (2 * area)
    cot0 = np.einsum("ij,ij->i", -e1, e2) / (2.0 * area)
    cot1 = np.einsum("ij,ij->i", -e2, e0) / (2.0 * area)
    cot2 = np.einsum("ij,ij->i", -e0, e1) / (2.0 * area)

    # off-diagonal stiffness entries: w_ij = cot(opposite angle) / 2
    I = np.concatenate([i1, i2, i2, i0, i0, i1])
    J = np.concatenate([i2, i1, i0, i2, i1, i0])
    W = np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2]) * 0.5

    C = sparse.coo_matrix((-W, (I, J)), shape=(n, n)).tocsr()
    # diagonal so that rows sum to zero
    C = C - sparse.diags(np.asarray(C.sum(axis=1)).ravel())

    # lumped mass: one third of incident triangle area
    M = np.zeros(n)
    np.add.at(M, i0, area / 3.0)
    np.add.at(M, i1, area / 3.0)
    np.add.at(M, i2, area / 3.0)
    M = np.maximum(M, 1e-16)

    return C.tocsc(), M


def lb_eigenmodes(verts, tris, n_modes=60):
    """
    Solve the generalised eigenproblem C phi = lambda M phi.

    Returns eigenvalues (ascending) and mass-orthonormal eigenmodes, i.e.
    phi_i^T M phi_j = delta_ij. The first mode is constant with eigenvalue ~0.
    """
    C, M = cotangent_laplacian(verts, tris)
    Msp = sparse.diags(M)

    # sigma-shift solves for the smallest eigenvalues robustly
    vals, vecs = eigsh(C, k=n_modes, M=Msp, sigma=-1e-8, which="LM")

    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]

    # mass-normalise
    for k in range(vecs.shape[1]):
        nrm = np.sqrt(vecs[:, k] @ (M * vecs[:, k]))
        if nrm > 0:
            vecs[:, k] /= nrm
    return vals, vecs


# ── Bilateral alignment of hemispheric modes ──────────────────────────────────

def align_hemispheres(lh_verts, lh_modes, rh_verts, rh_modes):
    """
    Match right-hemisphere modes to left-hemisphere modes by mirroring the
    right hemisphere across the mid-sagittal plane and comparing at nearest
    neighbours. A global sign is chosen per mode so the hemispheres agree.

    Returns the sign-corrected right-hemisphere modes.
    """
    from scipy.spatial import cKDTree

    rh_mirrored = rh_verts.copy()
    rh_mirrored[:, 0] *= -1.0                      # mirror x (left-right)

    tree = cKDTree(lh_verts)
    _, nn = tree.query(rh_mirrored, k=1)

    rh_aligned = rh_modes.copy()
    for k in range(rh_modes.shape[1]):
        corr = np.corrcoef(rh_modes[:, k], lh_modes[nn, k])[0, 1]
        if np.isfinite(corr) and corr < 0:
            rh_aligned[:, k] *= -1.0
    return rh_aligned


# ── Sensor dictionary and projection ──────────────────────────────────────────

def build_sensor_dictionary(lead_field, cortical_modes):
    """
    D = L @ Phi. Columns are the scalp topographies of individual cortical
    harmonics. Each column is normalised to unit norm so that projection
    coefficients are comparable across modes.

    lead_field     : (n_channels, n_sources)
    cortical_modes : (n_sources, n_modes)
    """
    D = lead_field @ cortical_modes
    norms = np.linalg.norm(D, axis=0, keepdims=True)
    return D / np.maximum(norms, 1e-30)


def project_ols(data, D):
    """
    Least-squares projection of scalp data onto the dictionary.

    data : (..., n_channels, n_times)
    D    : (n_channels, n_modes)

    Returns coefficients (..., n_modes, n_times).
    """
    pinv = np.linalg.pinv(D)                       # (n_modes, n_channels)
    return np.einsum("mc,...ct->...mt", pinv, data)


def mode_power(data, D, normalise=True):
    """
    Per-epoch power on each dictionary mode.

    data : (n_epochs, n_channels, n_times)
    Returns (n_epochs, n_modes).
    """
    coef = project_ols(data, D)
    p = (coef ** 2).mean(axis=-1)
    if normalise:
        p = p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-30)
    return p