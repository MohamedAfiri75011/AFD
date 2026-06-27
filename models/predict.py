# -*- coding: utf-8 -*-
import mlflow.sklearn
import pandas as pd
import numpy as np

# Configuration MLflow locale
MLFLOW_TRACKING_URI = "http://mlflow_server:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_URI = "models:/RandomForestRegressor@champion"

def make_prediction(input_data: dict) -> float:
    """ Charge le modèle Champion actuel et réalise l'inférence sur 5 variables. """
    try:
        # 1. Chargement dynamique du modèle depuis le registre MLflow
        model = mlflow.sklearn.load_model(MODEL_URI)
        
        # 2. Transformation du dictionnaire reçu en DataFrame d'une seule ligne
        # On s'assure que les clés soient strictement au bon format (minuscules et sans caractères complexes)
        clean_data = {}
        for key, value in input_data.items():
            clean_key = (
                key.lower()
                   .strip()
                   .replace("é", "e")
                   .replace("è", "e")
                   .replace("à", "a")
                   .replace("/", "_")
                   .replace(".", "_")
                   .replace(" ", "_")
            )
            clean_data[clean_key] = [value]
            
        df_input = pd.DataFrame(clean_data)
        
        # Ordre strict attendu par le ColumnTransformer de ton script d'entraînement
        expected_order = ["agence", "secteur", "bi_multi_1", "type_de_financement", "pays_beneficiaire"]
        df_input = df_input[expected_order]
        
        # 3. Inférence (Le modèle prédisant log_engagements)
        predicted_log = model.predict(df_input)[0]
        
        # 4. Conversion en k€ réelle
        predicted_real_k_eur = np.expm1(predicted_log)
        
        return float(predicted_real_k_eur)
        
    except Exception as e:
        print(f"[ERROR] Échec de l'inférence avec le modèle Champion : {str(e)}")
        return None