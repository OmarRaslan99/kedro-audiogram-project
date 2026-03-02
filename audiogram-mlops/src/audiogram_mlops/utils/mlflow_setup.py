from pathlib import Path
import mlflow


def setup_mlflow(experiment_name: str) -> None:
    """
    Configure MLflow pour utiliser un tracking store LOCAL au projet (./mlruns).
    Cela évite tout chemin absolu lié à une machine (ex: /home/raslan).
    """
    tracking_dir = Path.cwd() / "mlruns"
    tracking_dir.mkdir(exist_ok=True)

    # URI file://... absolue mais basée sur le cwd de la machine courante
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment(experiment_name)