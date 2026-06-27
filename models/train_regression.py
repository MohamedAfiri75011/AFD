# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import mlflow
import mlflow.sklearn
import mlflow.data
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.tracking import MlflowClient  # Pour gérer automatiquement les alias
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

# --- GESTION DYNAMIQUE DU PYTHON PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from api.database import engine
    print("[TRAIN] Engine importé avec succès depuis api.database.")
except ImportError:
    # Utilisation de l'URI IPv4 validée de ton pooler Supabase
    SUPABASE_DB_URI = "postgresql://postgres.vsusfuhifwtuxohnbmwi:Uv7K6MelZ4xMVcDS@aws-0-eu-west-1.pooler.supabase.com:5432/postgres?sslmode=require"
    engine = create_engine(SUPABASE_DB_URI, pool_pre_ping=True)
    print("[TRAIN] Bascule sur l'engine local de secours avec configuration IPv4.")
    
# Configuration
TABLE_NAME = "afd"
MODEL_NAME = "RandomForestRegressor"

def get_git_revision_hash() -> str:
    """Récupère le hash Git injecté par Docker ou via la commande locale."""
    # 1. On force la lecture de la variable d'environnement injectée dans le conteneur
    git_env = os.getenv("GIT_COMMIT", "").strip()
    if git_env and git_env != "indisponible":
        return git_env
        
    # 2. Sécurité : Si exécuté en local hors Docker
    try:
        import subprocess
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "indisponible"
    
def train_model(n_estimators=100, random_state=42, demo_mode=True, **kwargs):
    print("Récupération des données depuis Supabase...")
    
    current_code_version = get_git_revision_hash()
    
    # 1. Extraction des données
    try:
        query = f'SELECT * FROM public."{TABLE_NAME}";'
        df = pd.read_sql_query(query, con=engine)
    except Exception as e:
        return {"status": "failed", "error": f"Erreur Supabase : {str(e)}"}

    if df.empty:
        return {"status": "failed", "error": f"La table '{TABLE_NAME}' est vide."}

    # 🎯 OPTIMISATION SPECIAL SOUTENANCE : Mode Démo / Échantillonnage
    total_rows_database = len(df)
    if demo_mode and total_rows_database > 1000:
        print(f"[DEMO MODE] Activation du mode démo. Échantillonnage de {total_rows_database} lignes à 1 000 lignes.")
        df = df.sample(n=1000, random_state=int(random_state)).reset_index(drop=True)

    # 2. Nettoyage standardisé des noms de colonnes
    df.columns = [
        col.lower()
           .strip()
           .replace("é", "e")
           .replace("è", "e")
           .replace("à", "a")
           .replace("ô", "o")
           .replace("'", "_")
           .replace("-", "_")
           .replace(" ", "_")
           .replace("(", "")
           .replace(")", "")
           .replace("/", "_")
           .replace(".", "_")
        for col in df.columns
    ]

    # 3. Vérification et isolement de la cible (y)
    if "log_engagements" in df.columns:
        y_log = df["log_engagements"]
    else:
        return {"status": "failed", "error": "La colonne 'log_engagements' est introuvable."}

    # 4. Isolement strict des 5 caractéristiques (X) demandées
    features_api = [
        "agence", 
        "secteur", 
        "bi_multi_1", 
        "type_de_financement", 
        "pays_beneficiaire"
    ]

    for col in features_api:
        if col not in df.columns:
            return {"status": "failed", "error": f"La colonne requise '{col}' est introuvable après traitement."}

    X = df[features_api]

    X_train, X_val, y_train_log, y_val_log = train_test_split(
        X, y_log, test_size=0.2, random_state=int(random_state)
    )

    # 5. Définition EXPLICITE des types de colonnes
    categorical_cols = [
        "agence", 
        "secteur", 
        "bi_multi_1", 
        "type_de_financement", 
        "pays_beneficiaire"
    ]
    numerical_cols = []

    # 6. Construction du Pipeline de Preprocessing
    numerical_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    # 7. Pipeline Global avec RandomForest OPTIMISÉ
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=int(n_estimators), 
            random_state=int(random_state), 
            max_depth=12,             
            min_samples_leaf=5,        
            max_features="sqrt",       
            n_jobs=-1                  
        ))
    ])

    # 8. Entraînement et Tracking MLflow
    try:
        mlflow.set_experiment("Agence_Developpement_Regression")
        
        with mlflow.start_run() as run:
            print("Entraînement du modèle...")
            mlflow.set_tag("git_commit", current_code_version)
            
            # Versionnement du Dataset
            mlflow_dataset: PandasDataset = mlflow.data.from_pandas(
                df=pd.concat([X_train, y_train_log], axis=1), 
                targets="log_engagements", 
                name="afd_train_dataset"
            )
            mlflow.log_input(mlflow_dataset, context="training")
            
            # Ajustement du modèle
            model_pipeline.fit(X_train, y_train_log)
            
            # Prédictions
            y_train_pred_log = model_pipeline.predict(X_train)
            y_val_pred_log = model_pipeline.predict(X_val)
            
            # Calcul du R2
            train_r2 = model_pipeline.score(X_train, y_train_log)
            val_r2 = model_pipeline.score(X_val, y_val_log)
            
            # Conversion k€
            y_train_real = np.expm1(y_train_log)
            y_train_pred_real = np.expm1(y_train_pred_log)
            y_val_real = np.expm1(y_val_log)
            y_val_pred_real = np.expm1(y_val_pred_log)
            
            # Erreurs réelles
            train_rmse_k_eur = root_mean_squared_error(y_train_real, y_train_pred_real)
            train_mae_k_eur = mean_absolute_error(y_train_real, y_train_pred_real)
            val_rmse_k_eur = root_mean_squared_error(y_val_real, y_val_pred_real)
            val_mae_k_eur = mean_absolute_error(y_val_real, y_val_pred_real)
            
            # Logging MLflow
            mlflow.log_param("code_version", current_code_version)
            mlflow.log_param("model_type", MODEL_NAME)
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("demo_mode", demo_mode)
            mlflow.log_param("rows_trained", len(df))
            
            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("val_r2", val_r2)
            mlflow.log_metric("train_rmse_k_eur", train_rmse_k_eur)
            mlflow.log_metric("val_rmse_k_eur", val_rmse_k_eur)
            mlflow.log_metric("train_mae_k_eur", train_mae_k_eur)
            mlflow.log_metric("mae", val_mae_k_eur)
            
            # Sauvegarde dans le Model Registry
            model_info = mlflow.sklearn.log_model(
                sk_model=model_pipeline, 
                artifact_path="model",
                registered_model_name=MODEL_NAME
            )
            
            # --- ALIAS CHAMPION INTELLIGENT ---
            print("[MLFLOW] Évaluation de la version pour le statut 'champion'...")
            client = MlflowClient()
            model_version = model_info.registered_model_version
            
            try:
                # 1. Tentative de récupération des performances du champion en cours
                current_champion = client.get_model_version_by_alias(name=MODEL_NAME, alias="champion")
                current_champion_run = client.get_run(current_champion.run_id)
                current_champion_mae = float(current_champion_run.data.metrics.get("mae", float("inf")))
                
                print(f"[MLFLOW] Champion actuel : Version {current_champion.version} (MAE: {current_champion_mae:.2f} k€)")
                print(f"[MLFLOW] Nouveau modèle   : Version {model_version} (MAE: {val_mae_k_eur:.2f} k€)")
                
                # 2. Transition de l'alias uniquement en cas d'amélioration de la MAE
                if val_mae_k_eur < current_champion_mae:
                    client.set_registered_model_alias(
                        name=MODEL_NAME,
                        alias="champion",
                        version=str(model_version)
                    )
                    print(f"[MLFLOW] Succès : La version {model_version} montre une MAE plus faible. Nouveau 'champion' assigné.")
                else:
                    print(f"[MLFLOW] Stabilité : La version {model_version} n'améliore pas la MAE. Le champion reste la version {current_champion.version}.")
            
            except Exception:
                # Bloc de secours s'il n'existe aucun champion (initialisation du registre)
                print(f"[MLFLOW] Aucun alias 'champion' détecté. Initialisation par défaut avec la version {model_version}.")
                client.set_registered_model_alias(
                    name=MODEL_NAME,
                    alias="champion",
                    version=str(model_version)
                )
            
            print(f"Entraînement terminé avec succès ! Val R2: {val_r2:.3f} | Val MAE: {val_mae_k_eur:.0f} k€")
            return {
                "status": "success",
                "run_id": run.info.run_id,
                "model_version": model_version,
                "metrics": {
                    "r2": val_r2, 
                    "rmse": val_rmse_k_eur,
                    "mae": val_mae_k_eur
                }
            }
            
    except Exception as e:
        return {"status": "failed", "error": f"Erreur pendant l'entraînement ou MLflow : {str(e)}"}

# Bloc d'exécution principale (permet de tester le script de manière autonome)
if __name__ == "__main__":
    # Laissé sur True pour tes tests locaux et la soutenance en direct
    result = train_model(n_estimators=20, random_state=42, demo_mode=True)
    print(result)