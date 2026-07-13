import os
import sys
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_SERVER = os.getenv("MLFLOW_TRACKING_SERVER", "http://mlflow_server:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_SERVER)
mlflow.set_registry_uri(MLFLOW_TRACKING_SERVER)

MODEL_NAME = "RandomForestRegressor"

# AJOUT DU CACHE GLOBAL
_cached_model = None

def load_model_from_mlflow(version="champion", force_reload=False):
    """Charge le modèle depuis MLflow (avec système de cache)"""
    global _cached_model
    
    # Si le modèle est déjà en RAM et qu'on ne force pas le rechargement, on le réutilise direct.
    if _cached_model is not None and not force_reload:
        return _cached_model

    logger.info(f"Téléchargement du modèle '{MODEL_NAME}' (alias: {version}) depuis MLflow...")
    model_uri = f"models:/{MODEL_NAME}@{version}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Modèle '{MODEL_NAME}@{version}' chargé avec succès en mémoire.")
        _cached_model = model  # On le sauvegarde dans le cache
        return model
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle depuis MLflow : {e}")
        return None

def make_prediction(input_data: dict, model_version="champion"):
    """Prédit le montant de l'engagement en utilisant le modèle en cache."""
    # Le modèle répondra instantanément s'il est déjà chargé
    model = load_model_from_mlflow(version=model_version)
    if model is None:
        logger.error("Inférence annulée : Modèle introuvable.")
        raise ValueError("Modèle introuvable ou non disponible.") 

    try:
        df_input = pd.DataFrame([input_data])

        # Normalisation identique à l'entraînement
        df_input.columns = (
            df_input.columns
            .str.lower()
            .str.strip()
            .str.normalize("NFKD")
            .str.encode("ascii", errors="ignore")
            .str.decode("utf-8")
            .str.replace(r"[^\w]+", "_", regex=True)
        )

        y_pred_log = model.predict(df_input)
        y_pred_real = np.expm1(y_pred_log[0])

        return float(y_pred_real)

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        raise e