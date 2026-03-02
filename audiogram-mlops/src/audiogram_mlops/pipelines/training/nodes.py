import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """
    Entraîne le modèle pour prédire les gains (y) à partir des features (X).
   
    """
    logger = logging.getLogger(__name__)

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    logger.info("Entraînement du modèle en cours...")
    model.fit(X_train, y_train)
    
    logger.info("Modèle entraîné avec succès.")
    return model
