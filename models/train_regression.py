import os
import sys
import subprocess
import json
import hashlib
import time
import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# ============================================================
# 1. PYTHON PATH DYNAMIQUE
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# ============================================================
# 2. ENGINE SUPABASE WITH NULLPOOL (Anti-saturation des connexions)
# ============================================================
try:
    from api.database import engine
    print("[TRAIN] Engine importé depuis api.database.")
except ImportError:
    SUPABASE_DB_URI = os.getenv(
        "SUPABASE_DB_URI",
        "postgresql://postgres:postgres@localhost:5432/postgres"
    )
    engine = create_engine(SUPABASE_DB_URI, poolclass=NullPool)
    print("[TRAIN] Fallback sur SUPABASE_DB_URI avec NullPool (Sécurité Connexions).")

# ============================================================
# 3. CONFIG
# ============================================================
TABLE_NAME = os.getenv("TABLE_NAME", "afd")
MODEL_NAME = os.getenv("MODEL_NAME", "RandomForestRegressor")
LOG_FULL_DATASET = True

def get_git_revision_hash() -> str:
    git_env = os.getenv("GIT_COMMIT", "").strip()
    if git_env and git_env != "indisponible":
        return git_env
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]
        ).decode("ascii").strip()
    except Exception:
        return "indisponible"

# ============================================================
# 4. FONCTION PRINCIPALE D'ENTRAÎNEMENT
# ============================================================
def train_model(
    n_estimators=800,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
    demo_mode=False
):
    print("Récupération des données depuis Supabase...")
    current_code_version = get_git_revision_hash()

    try:
        query = f'SELECT * FROM public."{TABLE_NAME}";'
        df = pd.read_sql_query(query, con=engine)
    except Exception as e:
        return {"status": "failed", "error": f"Erreur Supabase : {str(e)}"}

    if df.empty:
        return {"status": "failed", "error": f"La table '{TABLE_NAME}' est vide."}

    total_rows_database = len(df)

    if demo_mode and total_rows_database > 1000:
        print(f"[DEMO MODE] Échantillonnage à 1000 lignes.")
        df = df.sample(n=1000, random_state=random_state).reset_index(drop=True)
    else:
        print(f"[PRODUCTION MODE] Entraînement sur {total_rows_database} lignes.")

    # Normalisation des noms de colonnes
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.replace(r"[^\w]+", "_", regex=True)
    )

    if "log_engagements" not in df.columns:
        return {"status": "failed", "error": "Colonne 'log_engagements' introuvable."}

    y_log = df["log_engagements"]
    
    # Exclusion de la variable cible et du montant brut pour éviter le Data Leakage
    cols_to_exclude = [
        "log_engagements", 
        "engagements_k_eur_", 
        "engagements_k_eur", 
        "id", 
        "created_at", 
        "index"
    ]
    
    feature_cols = [c for c in df.columns if c not in cols_to_exclude]
    X = df[feature_cols]

    X_train, X_val, y_train_log, y_val_log = train_test_split(
        X, y_log, test_size=0.2, random_state=random_state
    )

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numerical_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                    max_features="sqrt",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # ============================================================
    # 5. MLflow Tracking
    # ============================================================
    try:
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        
        mlflow.set_experiment("Agence_Developpement_Regression")

        with mlflow.start_run() as run:
            mlflow.set_tag("git_commit", current_code_version)
            mlflow.set_tag("mode", "demo" if demo_mode else "full")

            # Enregistrement des paramètres sous forme de colonnes MLflow
            mlflow.log_param("n_estimators", int(n_estimators))
            mlflow.log_param("random_state", int(random_state))
            mlflow.log_param("min_samples_leaf", int(min_samples_leaf))

            def hash_dataframe(df_local: pd.DataFrame) -> str:
                return hashlib.md5(pd.util.hash_pandas_object(df_local, index=True).values).hexdigest()

            dataset_version = hash_dataframe(df)
            mlflow.log_param("dataset_version", dataset_version)
            mlflow.log_param("dataset_rows", int(df.shape[0]))
            mlflow.log_param("dataset_cols", int(df.shape[1]))
            mlflow.set_tag("dataset_name", f"{TABLE_NAME}_train")

            os.makedirs("dataset_artifact", exist_ok=True)
            meta = {
                "name": f"{TABLE_NAME}_train",
                "version": dataset_version,
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "schema": list(df.columns)
            }
            meta_path = os.path.join("dataset_artifact", "dataset_metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            mlflow.log_artifact(meta_path, artifact_path="dataset")

            if LOG_FULL_DATASET:
                try:
                    df_sample_ui = df.head(100) 
                    mlflow_dataset = mlflow.data.from_pandas(
                        df=df_sample_ui,
                        targets="log_engagements",
                        name=f"{TABLE_NAME}_prod_snapshot"
                    )
                    mlflow.log_input(mlflow_dataset, context="training", tags={
                        "real_rows_count": str(df.shape[0]),
                        "dataset_hash": dataset_version
                    })
                except Exception as e:
                    print(f"[MLFLOW] Échec du log UI du dataset : {e}")

            # Entraînement du modèle
            model_pipeline.fit(X_train, y_train_log)

            # Prédictions d'évaluation
            y_train_pred_log = model_pipeline.predict(X_train)
            y_val_pred_log = model_pipeline.predict(X_val)

            # --- CALCUL DES MÉTRIQUES ---
            r2_log_train = r2_score(y_train_log, y_train_pred_log)
            r2_log_val = r2_score(y_val_log, y_val_pred_log)
            
            # Reconversion sur l'échelle réelle (Euros)
            y_val_real = np.expm1(y_val_log)
            y_val_pred_real = np.expm1(y_val_pred_log)

            val_mae_k_eur = mean_absolute_error(y_val_real, y_val_pred_real) / 1000.0
            val_rmse_k_eur = mean_squared_error(y_val_real, y_val_pred_real, squared=False) / 1000.0

            # AJOUT POUR LA DÉMO : Calcul du Biais Central
            biais_reel = np.mean(y_val_real - y_val_pred_real)
            biais_k_eur = biais_reel / 1000.0

            # --- LOGGING DES COMPARAISONS SUR MLFLOW ---
            mlflow.log_metric("R2_log_train", float(r2_log_train))
            mlflow.log_metric("R2_log_val", float(r2_log_val))
            mlflow.log_metric("val_mae_k_eur", float(val_mae_k_eur))
            mlflow.log_metric("val_rmse_k_eur", float(val_rmse_k_eur))
            mlflow.log_metric("val_bias_k_eur", float(biais_k_eur)) # Logging du biais

            print(f"[METRICS] R2 Train (Log): {r2_log_train:.4f} | R2 Validation (Log): {r2_log_val:.4f}")
            print(f"[METRICS] Biais: {biais_k_eur:.2f} k€ | MAE: {val_mae_k_eur:.2f} k€ | RMSE: {val_rmse_k_eur:.2f} k€")
            sys.stdout.flush()

            # Enregistrement de l'artefact du modèle
            try:
                model_info = mlflow.sklearn.log_model(
                    sk_model=model_pipeline,
                    artifact_path="model",
                    serialization_format="cloudpickle"
                )
            except TypeError:
                model_info = mlflow.sklearn.log_model(
                    sk_model=model_pipeline,
                    artifact_path="model"
                )

            # Gestion de l'alias Champion basé sur la métrique MAE
            client = MlflowClient()
            model_version = None
            try:
                model_uri = getattr(model_info, "model_uri", None)
                if not model_uri:
                    model_uri = f"runs:/{run.info.run_id}/model"
                
                try:
                    client.get_registered_model(MODEL_NAME)
                except Exception:
                    print(f"[MLFLOW] Le modèle enregistré '{MODEL_NAME}' n'existe pas. Création...")
                    client.create_registered_model(MODEL_NAME)

                mv = client.create_model_version(name=MODEL_NAME, source=model_uri, run_id=run.info.run_id)
                model_version = mv.version
            except Exception as e:
                print(f"[MLFLOW WARNING] Échec de la création de la version : {e}")
                try:
                    all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
                    for v in all_versions:
                        if getattr(v, "run_id", None) == run.info.run_id:
                            model_version = v.version
                            break
                except Exception:
                    pass

            try:
                if model_version is not None:
                    try:
                        current_champion = client.get_model_version_by_alias(name=MODEL_NAME, alias="champion")
                        current_champion_run = client.get_run(current_champion.run_id)
                        current_champion_mae = float(current_champion_run.data.metrics.get("val_mae_k_eur", float("inf")))

                        if val_mae_k_eur < current_champion_mae:
                            client.set_registered_model_alias(name=MODEL_NAME, alias="champion", version=str(model_version))
                            print(f"[MLFLOW] Nouveau champion validé : version {model_version}")
                        else:
                            print(f"[MLFLOW] Champion actuel conservé : version {current_champion.version}")
                    except Exception:
                        client.set_registered_model_alias(name=MODEL_NAME, alias="champion", version=str(model_version))
                        print(f"[MLFLOW] Premier modèle enregistré comme champion : version {model_version}")
            except Exception as e:
                print("[MLFLOW WARNING] Erreur lors de l'alias champion :", e)

            # HISTORISATION SÉCURISÉE DES MÉTRIQUES DANS SUPABASE POUR GRAFANA
            try:
                with engine.begin() as conn:
                    conn.execute(
                        text("""
                        INSERT INTO public.model_metrics 
                        (created_at, model_version, run_id, r2_score, mae_k_eur, rmse_k_eur, bias_k_eur)
                        VALUES (NOW(), :version, :run, :r2, :mae, :rmse, :bias)
                        """),
                        {
                            "version": str(model_version),
                            "run": str(run.info.run_id),
                            "r2": float(r2_log_val),
                            "mae": float(val_mae_k_eur),
                            "rmse": float(val_rmse_k_eur),
                            "bias": float(biais_k_eur) # 🚨 Insertion du biais !
                        }
                    )
                print("[SQL] Métriques enregistrées dans Supabase (table model_metrics) avec succès.")
            except Exception as sql_err:
                print(f"[SQL ERROR] Échec de l'insertion dans model_metrics : {sql_err}")

            return {
                "status": "success",
                "run_id": run.info.run_id,
                "model_version": model_version,
                "metrics": {
                    "r2_train": float(r2_log_train),
                    "r2_val": float(r2_log_val),
                    "mae_k_eur": float(val_mae_k_eur),
                    "rmse_k_eur": float(val_rmse_k_eur),
                    "bias_k_eur": float(biais_k_eur) # Retour du biais dans la réponse
                },
            }

    except Exception as e:
        return {"status": "failed", "error": str(e)}

if __name__ == "__main__":
    train_model()