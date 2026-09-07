"""
src/unsupervised_analysis.py
============================
Unsupervised clustering of subjects from their EEG features, then testing
whether the data-driven clusters correspond to diagnosis.

Asks a different question from all the supervised analyses: does the data
organise into natural groups on its own, and do those groups align with
MDD/HC? Reported regardless of outcome.

Metrics:
  silhouette  : cluster quality/separation (>0.5 strong, <0.25 weak/none)
  ARI, AMI    : alignment of clusters with diagnosis (0 = chance)
  chi-square  : whether cluster membership relates to diagnosis
Also sweeps k and checks whether the natural number of clusters is 2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             adjusted_mutual_info_score)
from scipy.stats import chi2_contingency


def aggregate_to_subjects(X, y, groups):
    """Collapse per-epoch features to one mean vector per subject."""
    subs = np.unique(groups)
    Xs, ys = [], []
    for s in subs:
        mask = groups == s
        Xs.append(X[mask].mean(axis=0))
        ys.append(y[mask][0])
    return np.array(Xs), np.array(ys), subs


def run_clustering(X_subj, y_subj, use_pca=True, n_pca=10):
    """
    Cluster subjects and test alignment with diagnosis.
    Returns a results dict.
    """
    Xs = StandardScaler().fit_transform(X_subj)
    if use_pca and Xs.shape[1] > n_pca:
        Xs = PCA(n_components=n_pca, random_state=42).fit_transform(Xs)

    results = {"k_sweep": {}, "k2": {}}

    # 1. Sweep k to find the natural number of clusters (silhouette)
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
        results["k_sweep"][k] = silhouette_score(Xs, km.labels_)

    # 2. Detailed k=2 analysis (does it match diagnosis?)
    for name, algo in [("kmeans", KMeans(n_clusters=2, n_init=10, random_state=42)),
                       ("hierarchical", AgglomerativeClustering(n_clusters=2))]:
        clusters = algo.fit_predict(Xs)
        sil = silhouette_score(Xs, clusters)
        ari = adjusted_rand_score(y_subj, clusters)
        ami = adjusted_mutual_info_score(y_subj, clusters)
        tab = np.zeros((2, 2), dtype=int)
        for c, l in zip(clusters, y_subj):
            tab[c, l] += 1
        try:
            chi2, p, _, _ = chi2_contingency(tab)
        except Exception:
            chi2, p = np.nan, np.nan
        results["k2"][name] = {"silhouette": float(sil), "ARI": float(ari),
                               "AMI": float(ami), "chi2_p": float(p),
                               "contingency": tab.tolist()}
    return results


def print_results(results):
    print("Natural cluster count (silhouette by k):")
    for k, s in results["k_sweep"].items():
        print(f"  k={k}: silhouette={s:.3f}")
    best_k = max(results["k_sweep"], key=results["k_sweep"].get)
    print(f"  -> best k = {best_k} (silhouette {results['k_sweep'][best_k]:.3f})")
    print("\nk=2 alignment with diagnosis:")
    for name, r in results["k2"].items():
        print(f"  {name}: silhouette={r['silhouette']:.3f}, ARI={r['ARI']:.3f}, "
              f"AMI={r['AMI']:.3f}, chi2 p={r['chi2_p']:.3f}")
        print(f"    contingency (rows=cluster, cols=HC|MDD): {r['contingency']}")