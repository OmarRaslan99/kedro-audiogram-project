"""Nodes for the vocal training pipeline."""

import logging

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from audiogram_mlops.utils.mlflow_setup import setup_mlflow


def train_vocal_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    parameters: dict,
):
    """Train a multi-output RandomForestRegressor for vocal audiometry.

    Targets: true_srt50, true_gain_db.
    """
    logger = logging.getLogger(__name__)
    setup_mlflow("audiogram-vocal-silence")

    n_estimators = parameters.get("n_estimators", 100)
    max_depth = parameters.get("max_depth", 6)
    random_state = parameters.get("random_state", 42)

    mlflow.sklearn.autolog(log_models=True)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )

    with mlflow.start_run(run_name="training_vocal") as run:
        logger.info("[MLflow] training_vocal run_id = %s", run.info.run_id)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        logger.info(
            "Training vocal RandomForest (n=%s, depth=%s)...",
            n_estimators,
            max_depth,
        )
        model.fit(X_train, y_train)
        logger.info("Vocal model trained successfully.")

    return model
