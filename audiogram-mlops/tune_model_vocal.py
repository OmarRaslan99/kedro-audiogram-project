import optuna
import os
import mlflow
import logging
from audiogram_mlops.utils.mlflow_setup import setup_mlflow


def objective(trial):
    """Fonction objectif pour Optuna : cherche à minimiser la MAE SRT50 vocale."""

    # 1. Définition de l'espace de recherche
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 2, 15)

    # 2. Exécution des pipelines training_vocal + evaluation_vocal
    params = f"model_options_vocal.n_estimators={n_estimators},model_options_vocal.max_depth={max_depth}"
    os.system(f"kedro run --pipeline training_vocal --params {params}")
    os.system(f"kedro run --pipeline evaluation_vocal")

    # 3. Récupération du score MAE_SRT50 depuis MLflow
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("audiogram-vocal-silence")

    if experiment is None:
        print("Attention : expérience 'audiogram-vocal-silence' introuvable.")
        return float("inf")

    # On récupère le run d'évaluation le plus récent
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'evaluation_vocal'",
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )

    if not runs:
        return float("inf")

    last_run_metrics = runs[0].data.metrics

    # Métrique cible : mae_srt50 (la plus cliniquement pertinente)
    if "mae_srt50" not in last_run_metrics:
        print(f"Attention : mae_srt50 non trouvée dans les métriques : {list(last_run_metrics.keys())}")
        return float("inf")

    return last_run_metrics["mae_srt50"]


if __name__ == "__main__":
    # Initialisation du logger pour voir les étapes
    logging.basicConfig(level=logging.INFO)

    # Configuration du dossier de stockage local
    setup_mlflow("audiogram-vocal-silence")

    # Création de l'étude Optuna
    study = optuna.create_study(direction="minimize")

    # Lancement de l'optimisation sur 20 essais
    study.optimize(objective, n_trials=20)

    best = study.best_params
    print("\n" + "=" * 50)
    print("OPTIMISATION VOCALE TERMINÉE")
    print(f"Meilleure MAE_SRT50 trouvée : {study.best_value:.4f} dB")
    print(f"Paramètres optimaux : {best}")
    print("=" * 50)

    # 🔁 Re-run final avec les meilleurs paramètres pour sauvegarder le modèle optimal
    print("\n🔁 Re-entraînement vocal final avec les meilleurs paramètres...")
    best_params = f"model_options_vocal.n_estimators={best['n_estimators']},model_options_vocal.max_depth={best['max_depth']}"
    os.system(f"kedro run --pipeline training_vocal --params {best_params}")
    os.system(f"kedro run --pipeline evaluation_vocal")
    print("✅ Modèle vocal optimal sauvegardé dans data/06_models/audiogram_vocal_model.pkl")
