import optuna
import os
import mlflow
import logging
from audiogram_mlops.utils.mlflow_setup import setup_mlflow

def objective(trial):
    """Fonction objectif pour Optuna : cherche à minimiser l'erreur (MAE)."""
    
    # 1. Définition de l'espace de recherche
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 2, 15)
    
    # 2. Exécution des pipelines training + evaluation avec les paramètres suggérés
    params = f"model_options.n_estimators={n_estimators},model_options.max_depth={max_depth}"
    os.system(f"kedro run --pipeline training --params {params}")
    os.system(f"kedro run --pipeline evaluation")
    
    # 3. Récupération sécurisée du score MAE depuis MLflow
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("audiogram-mlops")
    
    # On récupère le run le plus récent de cette expérience
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1,
        order_by=["attributes.start_time DESC"]
    )
    
    if not runs:
        return float("inf")
    
    last_run_metrics = runs[0].data.metrics
    
    # On cherche la clé qui correspond à la MAE
    mae_keys = [k for k in last_run_metrics.keys() if "mae" in k.lower() or "mean_absolute_error" in k.lower()]
    
    if not mae_keys:
        print(f"Attention : MAE non trouvée dans les métriques : {list(last_run_metrics.keys())}")
        return float("inf")
        
    return last_run_metrics[mae_keys[0]]

if __name__ == "__main__":
    # Initialisation du logger pour voir les étapes
    logging.basicConfig(level=logging.INFO)
    
    # Configuration du dossier de stockage local
    setup_mlflow("audiogram-mlops")
    
    # Création de l'étude Optuna 
    study = optuna.create_study(direction="minimize")
    
    # Lancement de l'optimisation sur 20 essais
    study.optimize(objective, n_trials=20)
    
    best = study.best_params
    print("\n" + "="*50)
    print("OPTIMISATION TERMINÉE")
    print(f"Meilleure MAE trouvée : {study.best_value:.4f} dB")
    print(f"Paramètres optimaux : {best}")
    print("="*50)

    # 🔁 Re-run final avec les meilleurs paramètres pour sauvegarder le modèle optimal
    print("\n🔁 Re-entraînement final avec les meilleurs paramètres...")
    best_params = f"model_options.n_estimators={best['n_estimators']},model_options.max_depth={best['max_depth']}"
    os.system(f"kedro run --pipeline training --params {best_params}")
    os.system(f"kedro run --pipeline evaluation")
    print("✅ Modèle optimal sauvegardé dans data/06_models/audiogram_model.pkl")