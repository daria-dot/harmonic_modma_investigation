"""
make_results_figures.py
=======================
Builds three figures from model_comparison_final.json using stock matplotlib
defaults: default font, default colour cycle, default spines and ticks.

  fig_model_forest       AUC with 95% bootstrap CI per model, chance line
  fig_permutation_nulls  null distributions with the observed AUC marked
  fig_sens_spec          sensitivity vs specificity, showing majority-class bias

Colab:
    !python make_results_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CACHE   = Path("/content/drive/MyDrive/mdd_eeg/cache")
IN_JSON = CACHE / "model_comparison_final.json"
OUTDIR  = Path("/content/drive/MyDrive/mdd_eeg/figures")

DISPLAY = {
    "LDA": "Linear discriminant analysis",
    "ElasticNet": "Elastic-net logistic",
    "Linear SVM": "SVM (linear)",
    "RBF-SVM": "SVM (RBF)",
    "Random forest": "Random forest",
    "Gradient boosting": "Gradient boosting",
    "Stacking ensemble": "Stacking ensemble",
}
ORDER = ["LDA", "RBF-SVM", "Linear SVM", "ElasticNet",
         "Gradient boosting", "Stacking ensemble", "Random forest"]


def save(fig, name):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
        fig.savefig(OUTDIR / f"{name}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print("wrote", name)


def present(res):
    return [k for k in ORDER if k in res] + [k for k in res if k not in ORDER]


def forest(res):
    keys = present(res)
    n = len(keys)
    aucs = np.array([res[k]["auc"] for k in keys])
    lo   = np.array([res[k]["auc_ci"][0] for k in keys])
    hi   = np.array([res[k]["auc_ci"][1] for k in keys])
    ypos = np.arange(n)[::-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(aucs, ypos, xerr=[aucs - lo, hi - aucs], fmt="o",
                capsize=3, color="C0")
    ax.axvline(0.5, color="k", linestyle="--", label="chance")

    for k, yv in zip(keys, ypos):
        p = res[k].get("p_value")
        ax.text(0.80, yv, f"p = {p:.3f}" if p is not None else "not permuted",
                va="center", fontsize=9)

    ax.set_yticks(ypos)
    ax.set_yticklabels([DISPLAY.get(k, k) for k in keys])
    ax.set_xlabel("Area under the ROC curve")
    ax.set_xlim(0.2, 0.95)
    ax.legend()
    save(fig, "fig_model_forest")


def nulls(res):
    keys = [k for k in present(res) if res[k].get("perm_aucs")]
    if not keys:
        print("  (no permutation distributions saved -- skipping)")
        return
    ncol = min(2, len(keys))
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                             squeeze=False)

    for ax, k in zip(axes.ravel(), keys):
        r = res[k]
        perms = np.asarray(r["perm_aucs"], float)
        ax.hist(perms, bins=20, color="C0", edgecolor="k", linewidth=0.5,
                label="permuted labels")
        ax.axvline(r["auc"], color="C3", label=f"observed ({r['auc']:.3f})")
        ax.axvline(0.5, color="k", linestyle="--", label="chance")
        ax.set_title(f"{DISPLAY.get(k, k)}  (p = {r['p_value']:.3f}, "
                     f"n = {len(perms)})")
        ax.set_xlabel("AUC")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.tight_layout()
    save(fig, "fig_permutation_nulls")


def sens_spec(res):
    keys = present(res)
    x = np.arange(len(keys))
    sens = [res[k]["sens"] for k in keys]
    spec = [res[k]["spec"] for k in keys]
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w/2, sens, w, label="Sensitivity (MDD)")
    ax.bar(x + w/2, spec, w, label="Specificity (HC)")
    ax.axhline(0.5, color="k", linestyle="--", label="chance")

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY.get(k, k) for k in keys], rotation=30,
                       ha="right")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.1)
    ax.legend()
    fig.tight_layout()
    save(fig, "fig_sens_spec")


def main():
    res = json.loads(Path(IN_JSON).read_text())
    print(f"loaded {len(res)} models from {IN_JSON}\n")
    forest(res)
    nulls(res)
    sens_spec(res)
    print(f"\nfigures in {OUTDIR}")


if __name__ == "__main__":
    main()