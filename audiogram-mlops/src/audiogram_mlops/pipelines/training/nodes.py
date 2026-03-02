import logging
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """
    Entraîne le modèle pour prédire les gains (y) à partir des features (X),
    et loggue un run MLflow séparé (Run #1 : training).

    - Utilise mlflow.sklearn.autolog() pour historiser automatiquement :
      paramètres, métriques, artefacts, modèle.
    - Crée un run MLflow dédié nommé "training".
    """
    logger = logging.getLogger(__name__)

    # Nom d'expérience (simple et stable pour le TP)
    mlflow.set_experiment("audiogram-mlops")

    # Active l'autolog scikit-learn
    mlflow.sklearn.autolog(log_models=True)

    # Hyperparamètres (on les garde explicitement aussi, utile pour lecture)
    n_estimators = 100
    random_state = 42

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    with mlflow.start_run(run_name="training") as run:
        logger.info(f"[MLflow] training run_id = {run.info.run_id}")

        # Log manuel (optionnel mais clair) : params
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)

        logger.info("Entraînement du modèle en cours...")
        model.fit(X_train, y_train)

        logger.info("Modèle entraîné avec succès.")
        return model