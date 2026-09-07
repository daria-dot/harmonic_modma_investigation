"""
model_comparison_final.py
=========================
Definitive re-run of the subject-level model comparison.

Adds to the previous version:
  * bootstrap 95% CIs on AUC, resampled over SUBJECTS (n = 53), which is the
    unit of independence -- not over predictions
  * subject-level permutation tests with the (1 + k) / (1 + n) estimator, so a
    p-value can never come out as exactly zero
  * checkpointing after every permutation, so a Colab disconnect costs one
    permutation rather than the whole run
  * a runtime estimate printed before the permutations start
  * LaTeX table emitted at the end

Run the CONFIG cell, then this. Tree models are slow to permute; set their
N_PERM low or to 0 and say so in the write-up.
"""

import json, pickle, sys, time, datetime
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier,
                              HistGradientBoostingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

# ─────────────────────────────── CONFIG ──────────────────────────────────
ROOT      = Path("/content/drive/MyDrive/mdd_eeg/mdd_eeg_pipeline 2")
SEED      = 42
N_BOOT    = 2000
N_JOBS    = -1               # use all cores; does not change results
N_MODES   = 20
BAND      = (8, 13)          # alpha
OUT_JSON  = "model_comparison_final.json"

# Permutations per model. Tree models are ~50x slower; 0 disables.
N_PERM = {
    "LDA":                 200,
    "Linear SVM":          200,
    "ElasticNet":          200,
    "RBF-SVM":             200,
    # Below-chance models: a permutation test asks whether the observed AUC is
    # unusually HIGH under the null, so it is uninformative when the observed
    # value is already under 0.5. Set these to 25 if you want the numbers
    # anyway (~30 min each with N_JOBS=-1).
    "Random forest":         0,
    "Gradient boosting":     0,
    "Stacking ensemble":     0,
}

sys.path.insert(0, str(ROOT))            # config.py lives here
sys.path.insert(0, str(ROOT / "src"))    # modma_load.py lives here
import config                                              # noqa: E402
from src import objective2, metrics                        # noqa: E402
import modma_load as ml                                    # noqa: E402

CKPT = Path(config.CACHE_DIR) / "model_comparison_final_ckpt.json"

# ──────────────────────────── FEATURES ───────────────────────────────────
def build_features():
    with open(Path(config.CACHE_DIR) / "modma_epochs_ica45.pkl", "rb") as f:
        epochs_ica = pickle.load(f)
    modma_dir = ROOT / "modma" / "EEG_128channels_resting_lanzhou_2015"
    labels = ml.load_modma_labels(
        modma_dir / "subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx")
    label_map = dict(zip(labels["subject"], labels["label"]))

    X, y, g = [], [], []
    for sid, ep in epochs_ica.items():
        lab = label_map.get(sid)
        if lab not in ("HC", "MDD") or len(ep) < 2:
            continue
        w, ch_ = objective2.compute_wpli(ep, fmin=BAND[0], fmax=BAND[1])
        _, ev = objective2.compute_graph_harmonics(w)
        p = objective2.project_and_extract_features(
            ep, ev, ch_)["per_epoch_power"][:, :N_MODES]
        X.append(p.mean(axis=0))
        y.append(1 if lab == "MDD" else 0)
        g.append(sid)
    return np.array(X), np.array(y), np.array(g)


# ──────────────────────── EVALUATION HELPERS ─────────────────────────────
def nested_cv(estimator, grid, X, y, g, seed=SEED):
    """One nested-CV pass. Returns pooled out-of-fold (y_true, y_score)."""
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    yt, ys = [], []
    for tr, te in outer.split(X, y, groups=g):
        pipe = Pipeline([("s", StandardScaler()), ("clf", estimator)])
        if grid:
            inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
            model = GridSearchCV(
                pipe, grid, scoring="roc_auc",
                cv=list(inner.split(X[tr], y[tr], groups=g[tr])),
                n_jobs=N_JOBS, error_score=np.nan)   # nan, not 0.5: see note
        else:
            model = pipe
        model.fit(X[tr], y[tr])
        est = model.best_estimator_ if grid else model
        ys.extend(est.predict_proba(X[te])[:, 1])
        yt.extend(y[te])
    return np.asarray(yt), np.asarray(ys)


def bootstrap_auc_ci(y_true, y_score, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    """Percentile CI for AUC, resampling subjects with replacement."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:      # degenerate resample
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    if len(aucs) < 100:
        return (float("nan"), float("nan"), 0)
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), len(aucs)


def permutation_p(estimator_fn, grid, X, y, g, n_perm, observed_auc, seed=SEED,
                  label="", ckpt_key=None):
    """
    Subject-level permutation test.

    p = (1 + #{perm >= observed}) / (1 + n_perm), the Phipson & Smyth (2010)
    estimator. The naive mean(perm >= obs) can return exactly 0, which is not
    a valid p-value.
    """
    if n_perm <= 0:
        return None, []

    done = load_ckpt(ckpt_key) if ckpt_key else []
    perms = list(done)
    start = len(perms)
    if start:
        print(f"    resuming from {start}/{n_perm}")

    t0 = time.time()
    for i in range(start, n_perm):
        rng = np.random.default_rng(seed + 1000 + i)
        yp = rng.permutation(y)                  # one label per subject already
        _, ys = nested_cv(estimator_fn(), grid, X, yp, g, seed=seed)
        perms.append(float(roc_auc_score(yp, ys)))
        if ckpt_key:
            save_ckpt(ckpt_key, perms)
        if i == start:
            per = time.time() - t0
            eta = per * (n_perm - start)
            print(f"    ~{per:.1f}s per permutation, ETA "
                  f"{datetime.timedelta(seconds=int(eta))}")
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{n_perm}")

    arr = np.asarray(perms)
    p = (1 + int(np.sum(arr >= observed_auc))) / (1 + len(arr))
    return float(p), perms


def load_ckpt(key):
    if CKPT.exists():
        try:
            return json.loads(CKPT.read_text()).get(key, [])
        except json.JSONDecodeError:
            return []
    return []


def save_ckpt(key, perms):
    d = {}
    if CKPT.exists():
        try:
            d = json.loads(CKPT.read_text())
        except json.JSONDecodeError:
            pass
    d[key] = perms
    CKPT.write_text(json.dumps(d))


# ───────────────────────────── MODELS ────────────────────────────────────
def model_specs():
    """Callables, so each permutation gets a fresh unfitted estimator."""
    return {
        "LDA": (lambda: LinearDiscriminantAnalysis(), None),
        "Linear SVM": (
            lambda: SVC(kernel="linear", probability=True, random_state=SEED), None),
        "ElasticNet": (
            lambda: LogisticRegression(penalty="elasticnet", solver="saga",
                                       max_iter=20000, tol=1e-3),
            {"clf__C": [0.01, 0.1, 1.0], "clf__l1_ratio": [0.1, 0.9]}),
        "RBF-SVM": (
            lambda: SVC(kernel="rbf", probability=True, class_weight="balanced",
                        random_state=SEED),
            {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.1]}),
        "Random forest": (
            lambda: RandomForestClassifier(n_estimators=300,
                                           class_weight="balanced",
                                           random_state=SEED, n_jobs=N_JOBS),
            {"clf__max_depth": [3, 5, None], "clf__min_samples_leaf": [1, 5, 20]}),
        "Gradient boosting": (
            lambda: HistGradientBoostingClassifier(early_stopping=True,
                                                   validation_fraction=0.15,
                                                   random_state=SEED),
            {"clf__max_depth": [2, 3], "clf__learning_rate": [0.03, 0.1],
             "clf__min_samples_leaf": [20, 50]}),
        "Stacking ensemble": (
            lambda: StackingClassifier(
                estimators=[
                    ("enet", LogisticRegression(penalty="elasticnet", solver="saga",
                                                C=0.1, l1_ratio=0.5,
                                                max_iter=20000, tol=1e-3)),
                    ("rf", RandomForestClassifier(n_estimators=300, max_depth=3,
                                                  class_weight="balanced",
                                                  random_state=SEED,
                                                  n_jobs=N_JOBS)),
                    ("gb", HistGradientBoostingClassifier(max_depth=2,
                                                          early_stopping=True,
                                                          random_state=SEED))],
                final_estimator=LogisticRegression(max_iter=5000), cv=3), None),
    }


# ─────────────────────────────── LATEX ───────────────────────────────────
DISPLAY = {
    "LDA": "Linear discriminant analysis",
    "ElasticNet": "Elastic-net logistic regression",
    "Linear SVM": "Support vector machine (linear)",
    "RBF-SVM": "Support vector machine (RBF)",
    "Random forest": "Random forest",
    "Gradient boosting": "Gradient boosting",
    "Stacking ensemble": "Stacking ensemble",
}
ORDER = ["LDA", "ElasticNet", "Linear SVM", "RBF-SVM",
         "Random forest", "Gradient boosting", "Stacking ensemble"]


def latex_table(res):
    lines = [
        r"\begin{table}[htbp]", r"  \centering",
        r"  \caption{Subject-level classification performance across model",
        r"  architectures (alpha band, 20 harmonic modes, 53 participants).",
        r"  Confidence intervals are percentile bootstrap over participants;",
        r"  $p$-values are from subject-level label permutation; models scoring",
        r"  below chance were not permuted, the test being uninformative there.}",
        r"  \label{tab:models}", r"  \small",
        r"  \begin{tabular}{@{}l c c c c c@{}}", r"    \toprule",
        r"    Classifier & AUC & 95\% CI & Bal.\ acc. & Sens. & $p$ \\",
        r"    \midrule",
    ]
    for k in ORDER:
        r = res.get(k)
        if not r:
            continue
        ci = (f"[{r['auc_ci'][0]:.3f}, {r['auc_ci'][1]:.3f}]"
              if r["auc_ci"][0] == r["auc_ci"][0] else "---")
        p = f"{r['p_value']:.3f}" if r.get("p_value") is not None else "---"
        star = r"$^{*}$" if r["sens"] == 0.0 else ""
        lines.append(f"    {DISPLAY[k]} & {r['auc']:.3f} & {ci} & "
                     f"{r['bal_acc']:.3f}{star} & {r['sens']:.3f} & {p} \\\\")
    lines += [
        r"    \midrule",
        r"    Chance & 0.500 & --- & 0.500 & --- & --- \\",
        r"    \bottomrule", r"  \end{tabular}",
        r"  \vspace{0.4em}",
        r"  \begin{minipage}{0.92\textwidth}\footnotesize",
        r"  $^{*}$ Predicted the majority class for every participant;",
        r"  balanced accuracy is 0.500 by construction.",
        r"  \end{minipage}", r"\end{table}",
    ]
    return "\n".join(lines)


# ──────────────────────────────── MAIN ───────────────────────────────────
def main():
    X, y, g = build_features()
    print(f"{X.shape[0]} subjects, {X.shape[1]} features "
          f"({int(y.sum())} MDD, {int((1-y).sum())} HC)\n")

    res = {}
    for name, (fn, grid) in model_specs().items():
        print(f"{name}")
        yt, ys = nested_cv(fn(), grid, X, y, g)
        m = metrics.compute_metrics(yt, ys)
        auc = float(roc_auc_score(yt, ys))
        lo, hi, nb = bootstrap_auc_ci(yt, ys)
        print(f"  AUC={auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
              f"bal={m['balanced_accuracy']:.3f}  "
              f"sens={m['sensitivity_MDD']:.3f}  spec={m['specificity_HC']:.3f}")

        p, perms = permutation_p(fn, grid, X, y, g, N_PERM.get(name, 0), auc,
                                 label=name, ckpt_key=name)
        if p is not None:
            print(f"  permutation p = {p:.4f}  ({len(perms)} shuffles)")

        res[name] = {"auc": auc, "auc_ci": [lo, hi], "n_boot_valid": nb,
                     "bal_acc": float(m["balanced_accuracy"]),
                     "sens": float(m["sensitivity_MDD"]),
                     "spec": float(m["specificity_HC"]),
                     "p_value": p, "n_perm": len(perms),
                     "perm_aucs": perms}
        print()

    out = Path(config.CACHE_DIR) / OUT_JSON
    out.write_text(json.dumps(res, indent=2))
    print(f"Saved {out}\n")
    print(latex_table(res))


if __name__ == "__main__":
    main()