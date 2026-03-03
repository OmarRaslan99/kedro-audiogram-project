"""Nodes for the vocal evaluation pipeline."""

import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from audiogram_mlops.utils.mlflow_setup import setup_mlflow


def evaluate_vocal_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    parameters: dict,
):
    """Evaluate the vocal multi-output model and log results to MLflow.

    Metrics computed:
        - MAE and R² per target (srt50, gain_db)
        - Clinical accuracy per target and combined (within tolerances)
    """
    logger = logging.getLogger(__name__)
    setup_mlflow("audiogram-vocal-silence")

    predictions = model.predict(X_test)

    # Ensure array shape even if y_test is a DataFrame
    y_true = y_test.values if hasattr(y_test, "values") else np.array(y_test)
    y_pred = np.array(predictions)

    # ── Per-target metrics ────────────────────────────────────────────
    mae_srt50 = float(mean_absolute_error(y_true[:, 0], y_pred[:, 0]))
    mae_gain = float(mean_absolute_error(y_true[:, 1], y_pred[:, 1]))
    r2_srt50 = float(r2_score(y_true[:, 0], y_pred[:, 0]))
    r2_gain = float(r2_score(y_true[:, 1], y_pred[:, 1]))

    # ── Clinical accuracy ─────────────────────────────────────────────
    tol_srt50 = parameters.get("tol_srt50", 5.0)
    tol_gain = parameters.get("tol_gain_db", 5.0)

    correct_srt50 = np.abs(y_true[:, 0] - y_pred[:, 0]) <= tol_srt50
    correct_gain = np.abs(y_true[:, 1] - y_pred[:, 1]) <= tol_gain
    correct_both = correct_srt50 & correct_gain

    accuracy_srt50 = float(correct_srt50.mean())
    accuracy_gain = float(correct_gain.mean())
    accuracy_both = float(correct_both.mean())

    # ── MLflow logging ────────────────────────────────────────────────
    with mlflow.start_run(run_name="evaluation_vocal") as run:
        logger.info("[MLflow] evaluation_vocal run_id = %s", run.info.run_id)

        mlflow.log_metric("mae_srt50", mae_srt50)
        mlflow.log_metric("mae_gain_db", mae_gain)
        mlflow.log_metric("r2_srt50", r2_srt50)
        mlflow.log_metric("r2_gain_db", r2_gain)
        mlflow.log_metric("accuracy_srt50", accuracy_srt50)
        mlflow.log_metric("accuracy_gain_db", accuracy_gain)
        mlflow.log_metric("accuracy_both", accuracy_both)

        mlflow.log_param("tol_srt50", tol_srt50)
        mlflow.log_param("tol_gain_db", tol_gain)

        # ── Evaluation report artefact ────────────────────────────────
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        report = {
            "mae_srt50": mae_srt50,
            "mae_gain_db": mae_gain,
            "r2_srt50": r2_srt50,
            "r2_gain_db": r2_gain,
            "accuracy_srt50": accuracy_srt50,
            "accuracy_gain_db": accuracy_gain,
            "accuracy_both": accuracy_both,
            "tol_srt50": tol_srt50,
            "tol_gain_db": tol_gain,
        }
        report_path = out_dir / "evaluation_report_vocal.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(report_path))

        logger.info("Vocal evaluation — MAE_SRT50=%.2f  MAE_gain=%.2f", mae_srt50, mae_gain)
        logger.info("Vocal evaluation — R²_SRT50=%.4f  R²_gain=%.4f", r2_srt50, r2_gain)
        logger.info(
            "Vocal evaluation — Acc_SRT50=%.2f%%  Acc_gain=%.2f%%  Acc_both=%.2f%%",
            accuracy_srt50 * 100,
            accuracy_gain * 100,
            accuracy_both * 100,
        )

    return {
        "mae_srt50": {"value": mae_srt50, "step": 1},
        "mae_gain_db": {"value": mae_gain, "step": 1},
        "r2_srt50": {"value": r2_srt50, "step": 1},
        "r2_gain_db": {"value": r2_gain, "step": 1},
        "accuracy_srt50": {"value": accuracy_srt50, "step": 1},
        "accuracy_gain_db": {"value": accuracy_gain, "step": 1},
        "accuracy_both": {"value": accuracy_both, "step": 1},
    }
