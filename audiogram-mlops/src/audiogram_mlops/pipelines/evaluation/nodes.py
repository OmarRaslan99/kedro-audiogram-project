"""
This is a boilerplate pipeline 'evaluation'
generated using Kedro 1.2.0
"""

import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, precision_score, recall_score, f1_score


def evaluate_model(model, test_data: pd.DataFrame, test_labels: pd.DataFrame):
    """
    Évalue la performance du modèle de prédiction de gains,
    et loggue un run MLflow séparé (Run #2 : evaluation).

    - Log manuel des métriques : mae, r2, f1_score (tolérance 5 dB)
    - Log d'un artefact : outputs/evaluation_report.json
    """
    logger = logging.getLogger(__name__)

    # Nom d'expérience (simple et stable pour le TP)
    mlflow.set_experiment("audiogram-mlops")

    # 1) Prédictions
    predictions = model.predict(test_data)

    # 2) Métriques de Régression
    mae = mean_absolute_error(test_labels, predictions)
    r2 = r2_score(test_labels, predictions)

    # 3) Métrique "F1 tolérance 5dB"
    tolerance = 5
    is_correct_pred = np.abs(test_labels.values.flatten() - predictions.flatten()) <= tolerance

    y_true_binary = np.ones(len(is_correct_pred))
    y_pred_binary = is_correct_pred.astype(int)

    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

    # Logs console
    logger.info("--- Rapport d'Évaluation ---")
    logger.info(f"MAE (Erreur moyenne) : {mae:.2f} dB")
    logger.info(f"R2 Score : {r2:.2f}")
    logger.info(f"F1-Score (tolérance {tolerance}dB) : {f1:.2f}")
    logger.info("---------------------------")

    # --- MLflow Run #2 ---
    with mlflow.start_run(run_name="evaluation") as run:
        logger.info(f"[MLflow] evaluation run_id = {run.info.run_id}")

        # Log des métriques (float obligatoire)
        mlflow.log_metric("mae_db", float(mae))
        mlflow.log_metric("r2", float(r2))
        mlflow.log_metric("f1_5db", float(f1))

        # Log de paramètres utiles
        mlflow.log_param("tolerance_db", tolerance)
        mlflow.log_param("precision_5db", float(precision))
        mlflow.log_param("recall_5db", float(recall))

        # Artifact : rapport JSON
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)

        report = {
            "mae_db": float(mae),
            "r2": float(r2),
            "f1_5db": float(f1),
            "tolerance_db": tolerance,
            "precision_5db": float(precision),
            "recall_5db": float(recall),
        }
        report_path = out_dir / "evaluation_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        mlflow.log_artifacts(str(out_dir))

    # On garde ton format de retour initial (compatible Kedro MemoryDataset)
    return {
        "mae": {"value": float(mae), "step": 1},
        "r2": {"value": float(r2), "step": 1},
        "f1_score": {"value": float(f1), "step": 1},
    }