"""Nodes for the vocal ingestion pipeline."""

import pandas as pd


def clean_vocal_exams(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning of the synthetic vocal dataset.

    The data was generated programmatically so it is already well-formed.
    We still apply minimal safety checks (drop NAs, enforce types,
    range guard on scores) to mirror the tonal ingestion logic.
    """
    df = df.copy()

    # Enforce numeric types on numeric columns
    numeric_cols = ["patient_id", "is_aided", "intensity_db",
                    "recognition_score", "true_srt50", "true_gain_db"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing values
    df = df.dropna()

    # Basic range guards
    df = df[df["intensity_db"].between(0, 120)]
    df = df[df["recognition_score"].between(0, 100)]

    df = df.reset_index(drop=True)
    return df
