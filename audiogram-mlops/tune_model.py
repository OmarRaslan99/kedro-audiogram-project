import optuna
import os
import mlflow
from audiogram_mlops.utils.mlflow_setup import setup_mlflow

def objective(trial):
    # On définit les plages de tests
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 2, 12)
    
    # On lance le pipeline Kedro en écrasant les paramètres
    os.system(f"kedro run --params model_options.n_estimators={n_estimators},model_options.max_depth={max_depth}")
    
    # On récupère le score pour Optuna 
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("audiogram-mlops")
    last_run = client.search_runs(experiment_ids=[experiment.experiment_id], max_results=1)[0]
    
    return last_run.data.metrics["training_mean_absolute_error"]

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20) 