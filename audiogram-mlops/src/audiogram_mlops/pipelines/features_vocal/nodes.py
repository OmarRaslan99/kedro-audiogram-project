"""Nodes for the vocal features pipeline."""

import pandas as pd
from sklearn.model_selection import train_test_split

# Targets to predict
TARGET_COLS = ["true_srt50", "true_gain_db"]


def make_vocal_train_test(df: pd.DataFrame):
    """Pivot the long-format vocal data into one row per (patient, is_aided),
    then split into train / test sets.

    Features (X):
        - is_aided
        - type_surdite  (one-hot encoded)
        - score_0dB, score_5dB, …, score_100dB  (recognition_score pivoted)

    Targets (y):
        - true_srt50
        - true_gain_db
    """
    # ── Pivot recognition scores to wide format ───────────────────────
    # Each (patient_id, is_aided) becomes one row with score columns per intensity
    scores_wide = df.pivot_table(
        index=["patient_id", "is_aided"],
        columns="intensity_db",
        values="recognition_score",
        aggfunc="first",
    )
    scores_wide.columns = [f"score_{int(c)}dB" for c in scores_wide.columns]
    scores_wide = scores_wide.reset_index()

    # ── Retrieve per-row metadata (constant within a patient+is_aided group) ─
    meta = (
        df.groupby(["patient_id", "is_aided"])
        .agg(type_surdite=("type_surdite", "first"),
             true_srt50=("true_srt50", "first"),
             true_gain_db=("true_gain_db", "first"))
        .reset_index()
    )

    wide = scores_wide.merge(meta, on=["patient_id", "is_aided"])

    # ── One-hot encode type_surdite ──────────────────────────────────
    wide = pd.get_dummies(wide, columns=["type_surdite"], dtype=int)

    # ── Separate X and y ─────────────────────────────────────────────
    feature_cols = [c for c in wide.columns
                    if c not in ["patient_id"] + TARGET_COLS]
    X = wide[feature_cols].copy()
    y = wide[TARGET_COLS].copy()

    return train_test_split(X, y, test_size=0.2, random_state=42)
