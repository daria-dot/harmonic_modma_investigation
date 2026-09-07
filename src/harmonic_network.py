"""
src/harmonic_network.py
=======================
Harmonic-network features (novel): build a graph where NODES are connectome
harmonic modes and EDGES are the temporal phase coupling (phase-locking value)
between mode activation time-courses, then compute graph-theoretic metrics on
THAT network.

Rationale
---------
The standard analysis uses each harmonic mode's power as a feature. This instead
characterises how the modes *relate to one another* over time -- a "network of
harmonics." Because graph harmonics are spatially orthogonal by construction,
coupling is defined on their (non-orthogonal) TEMPORAL activations, not their
spatial patterns. Graph-theoretic summaries of this harmonic network
(modularity, efficiency, clustering, hubness, connectivity entropy) form a
compact, interpretable, and genuinely novel feature space.

This is a PER-SUBJECT feature (one harmonic network per subject, averaged over
epochs) -> analysed with leave-one-subject-out, like the HRV/reactivity work.

Pre-specified: reported regardless of outcome.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.signal import hilbert

from src import objective2

METRIC_NAMES = ["hn_mean_degree", "hn_mean_strength", "hn_clustering",
                "hn_transitivity", "hn_global_efficiency", "hn_modularity",
                "hn_n_communities", "hn_degree_heterogeneity", "hn_conn_entropy"]


def build_harmonic_network(mode_timecourses):
    """
    mode_timecourses: (n_epochs, n_modes, n_times)
    Returns n_modes x n_modes PLV adjacency, averaged over epochs.
    """
    n_ep, n_md, n_t = mode_timecourses.shape
    adj_sum = np.zeros((n_md, n_md))
    for e in range(n_ep):
        phases = np.angle(hilbert(mode_timecourses[e], axis=1))
        for i in range(n_md):
            for j in range(i + 1, n_md):
                plv = np.abs(np.mean(np.exp(1j * (phases[i] - phases[j]))))
                adj_sum[i, j] += plv
                adj_sum[j, i] += plv
    return adj_sum / max(n_ep, 1)


def harmonic_network_metrics(adj, density=0.3):
    """Graph-theoretic metrics on the harmonic network."""
    adj = adj.copy()
    n = adj.shape[0]
    np.fill_diagonal(adj, 0)
    tri = adj[np.triu_indices(n, 1)]
    thresh = np.percentile(tri, 100 * (1 - density)) if len(tri) else 0
    adj_bin = (adj > thresh).astype(float)

    G = nx.from_numpy_array(adj)
    Gb = nx.from_numpy_array(adj_bin)

    f = {}
    degs = np.array([d for _, d in Gb.degree()], dtype=float)
    f["hn_mean_degree"] = degs.mean()
    f["hn_mean_strength"] = adj.sum(axis=0).mean()
    try: f["hn_clustering"] = nx.average_clustering(G, weight="weight")
    except Exception: f["hn_clustering"] = np.nan
    try: f["hn_transitivity"] = nx.transitivity(Gb)
    except Exception: f["hn_transitivity"] = np.nan
    try: f["hn_global_efficiency"] = nx.global_efficiency(Gb)
    except Exception: f["hn_global_efficiency"] = np.nan
    try:
        comms = nx.community.greedy_modularity_communities(Gb)
        f["hn_modularity"] = nx.community.modularity(Gb, comms)
        f["hn_n_communities"] = float(len(comms))
    except Exception:
        f["hn_modularity"] = np.nan; f["hn_n_communities"] = np.nan
    f["hn_degree_heterogeneity"] = degs.std() / degs.mean() if degs.mean() > 0 else 0.0
    w = adj[np.triu_indices(n, 1)]
    w = w / w.sum() if w.sum() > 0 else w
    f["hn_conn_entropy"] = float(-np.sum(w[w > 0] * np.log(w[w > 0])))
    return f


def _project_timecourse(epochs, eigenvectors, ch_names, n_modes=20):
    eeg = epochs.copy().pick(ch_names)
    data = eeg.get_data()  # (n_epochs, n_ch, n_times)
    signal_modes = np.einsum("mc,ect->emt", eigenvectors.T, data)
    return signal_modes[:, :n_modes, :]


def build_harmonic_network_table(epochs_dict, labels_df, band=(8.0, 13.0), n_modes=20):
    """
    One row per subject: subject, label, + harmonic-network graph metrics.
    Returns a DataFrame (per-subject; analyse with LOSO).
    """
    import pandas as pd
    label_map = dict(zip(labels_df["subject"], labels_df["label"]))
    rows = []
    for sid, ep in epochs_dict.items():
        label = label_map.get(sid)
        if label not in ("HC", "MDD") or len(ep) < 2:
            continue
        wpli, ch = objective2.compute_wpli(ep, fmin=band[0], fmax=band[1])
        _, eigvecs = objective2.compute_graph_harmonics(wpli)
        tc = _project_timecourse(ep, eigvecs, ch, n_modes=n_modes)
        adj = build_harmonic_network(tc)
        feats = harmonic_network_metrics(adj)
        rows.append({"subject": sid, "label": label, **feats})
        print(f"  {sid} ({label}): modularity={feats['hn_modularity']:.3f}, "
              f"efficiency={feats['hn_global_efficiency']:.3f}")
    return pd.DataFrame(rows)


# ── Phase-synchrony summaries (final harmonic feature) ─────────────────────────

PHASE_SYNC_NAMES = ["ps_global", "ps_within_low", "ps_within_high", "ps_cross_scale"]


def phase_synchrony_features(adj, n_low=5, n_high=5):
    """
    Direct summaries of the harmonic phase-coupling matrix (not graph metrics):
      ps_global      : mean PLV across all mode pairs (overall coordination)
      ps_within_low  : mean PLV among low-order (global) modes
      ps_within_high : mean PLV among high-order (local) modes
      ps_cross_scale : mean PLV between low- and high-order modes
                       (integration between global and local scales)
    """
    n = adj.shape[0]
    def triu(M):
        return M[np.triu_indices_from(M, 1)]
    low_idx = list(range(min(n_low, n)))
    high_idx = list(range(max(0, n - n_high), n))

    glob = triu(adj).mean()
    within_low = triu(adj[np.ix_(low_idx, low_idx)]).mean()
    within_high = triu(adj[np.ix_(high_idx, high_idx)]).mean()
    cross = adj[np.ix_(low_idx, high_idx)].mean()
    return {"ps_global": float(glob), "ps_within_low": float(within_low),
            "ps_within_high": float(within_high), "ps_cross_scale": float(cross)}


def build_phase_synchrony_table(epochs_dict, labels_df, band=(8.0, 13.0),
                                n_modes=20, n_low=5, n_high=5):
    """One row per subject: subject, label, + phase-synchrony summaries."""
    import pandas as pd
    label_map = dict(zip(labels_df["subject"], labels_df["label"]))
    rows = []
    for sid, ep in epochs_dict.items():
        label = label_map.get(sid)
        if label not in ("HC", "MDD") or len(ep) < 2:
            continue
        wpli, ch = objective2.compute_wpli(ep, fmin=band[0], fmax=band[1])
        _, eigvecs = objective2.compute_graph_harmonics(wpli)
        tc = _project_timecourse(ep, eigvecs, ch, n_modes=n_modes)
        adj = build_harmonic_network(tc)
        feats = phase_synchrony_features(adj, n_low=n_low, n_high=n_high)
        rows.append({"subject": sid, "label": label, **feats})
        print(f"  {sid} ({label}): global={feats['ps_global']:.3f}, "
              f"cross-scale={feats['ps_cross_scale']:.3f}")
    return pd.DataFrame(rows)