"""
src/resumable_permutation.py
=============================
A checkpointing wrapper around the nested-CV permutation test.

Motivation: on larger datasets (e.g. MODMA, ~6,800 epochs across 53 subjects)
a full 200-permutation run exceeds the lifetime of a free Colab session. The
original permutation_test() only writes results at the very end, so a
disconnect loses everything.

This version writes a checkpoint every `save_every` permutations. If the
session dies, simply re-run the same call: it loads the checkpoint and
continues from where it stopped rather than starting over.

Usage
-----
    from src import resumable_permutation as rp

    result = rp.run_resumable_permutation_test(
        X, y, groups,
        checkpoint_path=config.CACHE_DIR / "modma_perm_checkpoint.npz",
        n_permutations=200,
        n_outer_splits=5,
        n_inner_splits=3,
        save_every=10,
    )

Re-running the identical call after a disconnect resumes automatically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import config
from src import objective3


def _load_checkpoint(path: Path):
    """Return (permuted_aucs, n_done, true_auc) from a checkpoint, or None."""
    if not Path(path).exists():
        return None
    data = np.load(path, allow_pickle=True)
    return (
        data["permuted_aucs"],
        int(data["n_done"]),
        float(data["true_auc"]),
    )


def _save_checkpoint(path: Path, permuted_aucs, n_done, true_auc):
    """Write progress atomically-ish (temp file then replace)."""
    path = Path(path)
    # np.savez appends '.npz' if the name lacks it, so build the temp name
    # with the suffix already present to keep the actual written path known.
    tmp = path.with_name(path.stem + "_tmp.npz")
    np.savez(
        tmp,
        permuted_aucs=permuted_aucs,
        n_done=n_done,
        true_auc=true_auc,
    )
    tmp.replace(path)


def run_resumable_permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    checkpoint_path: Path,
    n_permutations: int = 200,
    n_outer_splits: int = 5,
    n_inner_splits: int = 3,
    random_state: int | None = None,
    save_every: int = 10,
):
    """
    Permutation test with checkpointing.

    Labels are shuffled at the SUBJECT level (preserving the grouping
    structure), matching objective3.permutation_test.

    Parameters
    ----------
    checkpoint_path : Path
        Where progress is written. Delete this file to force a fresh start.
    save_every : int
        Write a checkpoint every this many permutations.

    Returns
    -------
    dict with keys: true_auc, permuted_aucs, p_value, n_permutations
    """
    if random_state is None:
        random_state = config.RANDOM_STATE

    checkpoint_path = Path(checkpoint_path)

    unique_groups = np.unique(groups)
    group_to_label = {}
    for g in unique_groups:
        labels_for_group = y[groups == g]
        assert len(set(labels_for_group)) == 1, f"Subject {g} has inconsistent labels"
        group_to_label[g] = labels_for_group[0]

    # ── Resume or start fresh ──────────────────────────────────────────────
    ckpt = _load_checkpoint(checkpoint_path)

    if ckpt is not None:
        permuted_aucs, n_done, true_auc = ckpt
        if len(permuted_aucs) != n_permutations:
            # Grow/shrink the array if n_permutations changed between runs
            grown = np.full(n_permutations, np.nan)
            keep = min(len(permuted_aucs), n_permutations)
            grown[:keep] = permuted_aucs[:keep]
            permuted_aucs = grown
            n_done = min(n_done, n_permutations)
        print(f"Resuming from checkpoint: {n_done}/{n_permutations} permutations done")
        print(f"  (true AUC from checkpoint: {true_auc:.3f})")
    else:
        print("No checkpoint found — running true (unshuffled) nested CV first...")
        true_result = objective3.nested_cv_auc(
            X, y, groups,
            n_outer_splits=n_outer_splits,
            n_inner_splits=n_inner_splits,
            random_state=random_state,
        )
        true_auc = true_result["overall_auc"]
        print(f"True overall AUC: {true_auc:.3f}\n")

        permuted_aucs = np.full(n_permutations, np.nan)
        n_done = 0
        _save_checkpoint(checkpoint_path, permuted_aucs, n_done, true_auc)

    # ── Run remaining permutations ─────────────────────────────────────────
    # Seed per-permutation so resumed runs reproduce the same shuffles
    print(f"Running permutations {n_done + 1}..{n_permutations}")

    for i in range(n_done, n_permutations):
        rng = np.random.RandomState(random_state + i)
        shuffled = rng.permutation(unique_groups)
        shuffled_map = dict(zip(unique_groups, [group_to_label[g] for g in shuffled]))
        y_perm = np.array([shuffled_map[g] for g in groups])

        res = objective3.nested_cv_auc(
            X, y_perm, groups,
            n_outer_splits=n_outer_splits,
            n_inner_splits=n_inner_splits,
            random_state=random_state,
        )
        permuted_aucs[i] = res["overall_auc"]

        if (i + 1) % save_every == 0 or (i + 1) == n_permutations:
            _save_checkpoint(checkpoint_path, permuted_aucs, i + 1, true_auc)
            done = i + 1
            valid = permuted_aucs[:done]
            running_p = float(np.mean(valid >= true_auc))
            print(f"  {done}/{n_permutations} done  |  running p = {running_p:.4f}  [saved]")

    valid = permuted_aucs[~np.isnan(permuted_aucs)]
    p_value = float(np.mean(valid >= true_auc))

    print(f"\nComplete. True AUC = {true_auc:.3f}, p = {p_value:.4f} "
          f"({len(valid)} permutations)")

    return {
        "true_auc": true_auc,
        "permuted_aucs": valid,
        "p_value": p_value,
        "n_permutations": len(valid),
    }