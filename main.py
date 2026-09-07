"""
main.py
=======
Runs the full pipeline end to end:
  Objective 1: load -> preprocess -> epoch
  Objective 2: graph harmonic feature extraction -> interpretive figures

Usage:
    python main.py
"""


from __future__ import annotations
import pickle

import config
from src import epochs as epochs_mod

from src import figures, labels, load, objective2, objective3, preprocess

def run_objective1():
    """Load, preprocess, and epoch all subjects. Returns everything needed downstream."""
    loaded = load.load_all_subjects()
    load.inspect(loaded)

    raws = {sid: raw for sid, (raw, _df) in loaded.items()}
    raw_dfs = {sid: df for sid, (_raw, df) in loaded.items()}

    preprocessed, bad_channel_report = preprocess.preprocess_all(loaded)
    preprocess.print_bad_channel_report(bad_channel_report)

    print("\nExtracting eyes-closed epochs...")
    epochs_dict = epochs_mod.extract_all_epochs(preprocessed, raw_dfs)

    return preprocessed, epochs_dict, bad_channel_report


def save_cache(preprocessed, epochs_dict, bad_channel_report):
    """Cache Objective 1 outputs so Objective 2/3 can be re-run without reprocessing."""
    cache_path = config.CACHE_DIR / "objective1_outputs.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "epochs_dict": epochs_dict,
                "bad_channel_report": bad_channel_report,
            },
            f,
        )
    print(f"\nCached Objective 1 outputs to {cache_path}")


def load_cache():
    """Load previously cached Objective 1 outputs."""
    cache_path = config.CACHE_DIR / "objective1_outputs.pkl"
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def run_objective2(epochs_dict):
    """Run graph harmonic feature extraction and generate interpretive figures."""
    labels_df = labels.get_labels()
    print("\nLabels:")
    print(labels_df.to_string(index=False))

    results, feature_table = objective2.run_objective2(epochs_dict, labels_df)

    print("\n\n══ FEATURE TABLE ══════════════════════════════════════")
    print(feature_table.to_string(index=False))

    print("\nGenerating interpretive figures...")
    figures.make_interpretive_figures(results, subject_ids=list(results.keys())[:1])

    return results, feature_table

def run_objective3(results, labels_df):
    """Run the ElasticNet classifier with nested grouped CV and permutation test."""
    result = objective3.run_objective3(
        results,
        labels_df,
        n_outer_splits=config.N_OUTER_SPLITS,
        n_inner_splits=config.N_INNER_SPLITS,
        n_permutations=config.N_PERMUTATIONS,
    )
    return result


if __name__ == "__main__":
    preprocessed, epochs_dict, bad_channel_report = run_objective1()
    save_cache(preprocessed, epochs_dict, bad_channel_report)

    results, feature_table = run_objective2(epochs_dict)

    feature_table.to_csv(config.CACHE_DIR / "feature_table.csv", index=False)
    print(f"\nFeature table saved to {config.CACHE_DIR / 'feature_table.csv'}")

    labels_df = labels.get_labels()
    objective3_result = run_objective3(results, labels_df)