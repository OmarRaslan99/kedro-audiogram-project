import logging
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from audiogram_mlops.utils.mlflow_setup import setup_mlflow

# Ajoute 'parameters' dans les arguments ici
def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: dict):
    logger = logging.getLogger(__name__)

    # 1. Configuration MLflow 
    setup_mlflow("audiogram-mlops")

    # 2. Récupération sécurisée des hyperparamètres
    n_estimators = parameters.get("n_estimators", 100)
    max_depth = parameters.get("max_depth", 6)
    random_state = parameters.get("random_state", 42)

    # 3. Activation de l'autologging 
    mlflow.sklearn.autolog(log_models=True)

    model = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=random_state
    )

    with mlflow.start_run(run_name="training") as run:
        logger.info(f"[MLflow] training run_id = {run.info.run_id}")

        # Logs manuels pour plus de clarté dans le tableau
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        logger.info(f"Entraînement RandomForest (n={n_estimators}, depth={max_depth})...")
        model.fit(X_train, y_train)
        logger.info("Modèle entraîné avec succès.")
        
        return model