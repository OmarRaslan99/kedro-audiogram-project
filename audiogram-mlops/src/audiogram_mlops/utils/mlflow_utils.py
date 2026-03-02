import mlflow

def setup_mlflow(experiment_name: str) -> None:
    """
    Configure MLflow tracking. For TP, we keep the default local tracking store (./mlruns).
    """
    mlflow.set_experiment(experiment_name)