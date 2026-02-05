"""
This is a boilerplate pipeline 'evaluation'
generated using Kedro 1.2.0
"""

import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, precision_score, recall_score, f1_score

def evaluate_model(model, test_data: pd.DataFrame, test_labels: pd.DataFrame):
    """
    Évalue la performance du modèle de prédiction de gains.
   
    """
    logger = logging.getLogger(__name__)
    
    # 1. Prédictions
    predictions = model.predict(test_data)
    
    # 2. Métriques de Régression (Précision brute)
    mae = mean_absolute_error(test_labels, predictions)
    r2 = r2_score(test_labels, predictions)
    
    # 3. Métriques de Classification (pour le F1-score du TP)
    # On définit un seuil de tolérance (ex: +/- 5dB) pour dire si la prédiction est "bonne"
    tolerance = 5
    is_correct_pred = np.abs(test_labels.values.flatten() - predictions.flatten()) <= tolerance
    
    # Pour le calcul, on compare à une réussite théorique (toutes les prédictions devraient être True)
    # Note : C'est une adaptation pour répondre aux exigences de ton support
    y_true_binary = np.ones(len(is_correct_pred)) 
    y_pred_binary = is_correct_pred.astype(int)

    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    
    # Formule du cours : F1 = 2 * (perf * recall) / (precision + recall)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

    # Affichage des résultats
    logger.info(f"--- Rapport d'Évaluation ---")
    logger.info(f"MAE (Erreur moyenne) : {mae:.2f} dB")
    logger.info(f"R2 Score : {r2:.2f}")
    logger.info(f"F1-Score (tolérance {tolerance}dB) : {f1:.2f}")
    logger.info(f"---------------------------")

    return {
        "mae": {"value": mae, "step": 1},
        "r2": {"value": r2, "step": 1},
        "f1_score": {"value": f1, "step": 1}
    }