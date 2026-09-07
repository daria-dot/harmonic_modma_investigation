"""
Subgroup analysis runner: LOSO classification within demographic subgroups,
for two feature sets (new-features-only, and all-harmonics-combined).
Reuses tested modules. Subject-grouped throughout (no leakage).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def subgroup_loso(X, y, groups, keep_subjects, label, metrics_mod):
    """LOSO within one subgroup. Returns dict of results or None if too small."""
    mask = np.array([g in keep_subjects for g in groups])
    Xs, ys, gs = X[mask], y[mask], groups[mask]
    uniq = np.unique(gs)
    n_mdd = sum(1 for g in uniq if ys[gs == g][0] == 1)
    n_hc = len(uniq) - n_mdd
    print(f"\n  {label}: {len(uniq)} subj ({n_mdd} MDD, {n_hc} HC), {Xs.shape[1]} features", end="")
    if n_mdd < 3 or n_hc < 3:
        print("  -> too few per class, skipped")
        return None
    # overfitting warning
    if Xs.shape[1] > len(uniq) / 2:
        print(f"  [!] {Xs.shape[1]} features vs {len(uniq)} subjects: high overfitting risk", end="")

    logo = LeaveOneGroupOut()
    yt_all, ys_all = [], []
    for tr, te in logo.split(Xs, ys, groups=gs):
        ig = gs[tr]
        ns = min(3, len(np.unique(ig)))
        inner = StratifiedGroupKFold(n_splits=ns, shuffle=True, random_state=42)
        isplits = list(inner.split(Xs[tr], ys[tr], groups=ig))
        pipe = Pipeline([("s", StandardScaler()),
                         ("clf", LogisticRegression(penalty="elasticnet", solver="saga",
                                  l1_ratio=0.5, max_iter=5000, tol=1e-3))])
        gsearch = GridSearchCV(pipe, {"clf__C": [0.01, 0.1, 1.0], "clf__l1_ratio": [0.1, 0.9]},
                               scoring="roc_auc", cv=isplits, n_jobs=1, error_score=0.5)
        gsearch.fit(Xs[tr], ys[tr])
        ys_all.append(gsearch.best_estimator_.predict_proba(Xs[te])[:, 1])
        yt_all.append(ys[te])
    yt = np.concatenate(yt_all); ysc = np.concatenate(ys_all)
    auc = roc_auc_score(yt, ysc) if len(set(yt)) == 2 else float("nan")
    m = metrics_mod.compute_metrics(yt, ysc)
    print(f"\n     -> AUC={auc:.3f}, bal_acc={m['balanced_accuracy']:.3f}, "
          f"MDD={m['sensitivity_MDD']:.3f}, HC={m['specificity_HC']:.3f}")
    return {"label": label, "n_subj": len(uniq), "auc": auc,
            "balanced_accuracy": m["balanced_accuracy"],
            "sensitivity_MDD": m["sensitivity_MDD"], "specificity_HC": m["specificity_HC"]}