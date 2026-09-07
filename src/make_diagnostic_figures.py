"""
make_diagnostic_figures.py
==========================
Diagnostics for Section 4.x (majority-class prediction), built from the pooled
held-out predictions saved by band_component_sweep.py.

  fig_prob_distributions  predicted probability by true class, pooled over folds
  fig_confusion           pooled confusion matrix, counts and row-normalised
  extra metrics printed:  MCC and average precision (PR-AUC)

Stock matplotlib defaults throughout.

Colab:
    !python make_diagnostic_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (confusion_matrix, matthews_corrcoef,
                             average_precision_score, roc_auc_score)

CACHE   = Path("/content/drive/MyDrive/mdd_eeg/cache")
IN_JSON = CACHE / "band_component_results.json"
OUTDIR  = Path("/content/drive/MyDrive/mdd_eeg/figures")

KEY = "Alpha"          # which analysis to diagnose
THRESH = 0.5


def save(fig, name):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
        fig.savefig(OUTDIR / f"{name}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print("wrote", name)


def prob_distributions(yt, ys, title):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(ys[yt == 0], bins=bins, alpha=0.6, label="Controls (n = %d)" % (yt == 0).sum())
    ax.hist(ys[yt == 1], bins=bins, alpha=0.6, label="MDD (n = %d)" % (yt == 1).sum())
    ax.axvline(THRESH, color="k", linestyle="--", label="decision threshold")
    ax.set_xlabel("Predicted probability of MDD")
    ax.set_ylabel("Number of participants")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    save(fig, "fig_prob_distributions")


def confusion(yt, ys, title):
    pred = (ys >= THRESH).astype(int)
    cm = confusion_matrix(yt, pred, labels=[0, 1])
    cmn = cm / cm.sum(axis=1, keepdims=True)
    labels = ["Control", "MDD"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for k, (ax, mat, fmt, sub) in enumerate(((axes[0], cm, "d", "counts"),
                                             (axes[1], cmn, ".2f",
                                              "row-normalised"))):
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                        color="white" if cmn[i, j] > 0.6 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(labels if k == 0 else ["", ""])
        ax.set_xlabel("Predicted")
        if k == 0:
            ax.set_ylabel("True")
        ax.set_title(sub)
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.04,
                 label="proportion of true class")
    fig.suptitle(title, y=1.02)
    save(fig, "fig_confusion")
    return cm


def main():
    res = json.loads(Path(IN_JSON).read_text())
    if KEY not in res or "y_true" not in res[KEY]:
        raise SystemExit(f"'{KEY}' has no saved predictions in {IN_JSON}. "
                         "Re-run band_component_sweep.py first.")
    r = res[KEY]
    yt = np.asarray(r["y_true"]); ys = np.asarray(r["y_score"])
    title = f"{KEY} harmonics, pooled held-out predictions"

    prob_distributions(yt, ys, title)
    cm = confusion(yt, ys, title)

    pred = (ys >= THRESH).astype(int)
    print(f"\n{KEY}: n = {len(yt)}  ({(yt==1).sum()} MDD, {(yt==0).sum()} HC)")
    print(f"  AUC                 {roc_auc_score(yt, ys):.3f}")
    print(f"  Average precision   {average_precision_score(yt, ys):.3f}"
          f"   (baseline {(yt==1).mean():.3f})")
    print(f"  MCC                 {matthews_corrcoef(yt, pred):.3f}")
    print(f"  predicted MDD       {pred.sum()} of {len(pred)} participants")
    print(f"  probability range   {ys.min():.3f} to {ys.max():.3f}")
    print(f"  confusion matrix    {cm.tolist()}  (rows true, cols predicted)")


if __name__ == "__main__":
    main()