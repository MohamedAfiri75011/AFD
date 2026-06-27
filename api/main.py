import datetime
import os
import secrets
import traceback
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy import create_engine, text
import mlflow
from mlflow.tracking import MlflowClient

# Imports locaux vers le dossier 'models'
from models.train_regression import train_model
from models.predict import make_prediction

# On importe la fonction directement depuis le fichier etl.py situé à la racine
from etl import run_etl

app = FastAPI(
    title="API MLOps — Spécification 5 Variables",
    description="Backend industriel connecté à MLflow et Supabase (Sécurisé par Basic Auth).",
    version="2.0.0"
)

# 🔐 CONFIGURATION DE LA SÉCURITÉ (BASIC AUTH)
security = HTTPBasic()
ADMIN_USER = "admin_agence"
ADMIN_PASSWORD = "AFD2026!"

def verification_auth(credentials: HTTPBasicCredentials = Depends(security)):
    """Vérifie de manière sécurisée les identifiants de Basic Auth."""
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USER)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiants d'accès incorrects ou manquants.",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# Configuration de la base de données de production Supabase
SUPABASE_DB_URI = "postgresql://postgres.vsusfuhifwtuxohnbmwi:Uv7K6MelZ4xMVcDS@aws-0-eu-west-1.pooler.supabase.com:5432/postgres?sslmode=require"
engine = create_engine(SUPABASE_DB_URI, pool_pre_ping=True)

# Initialisation du client MLflow
client = MlflowClient()
MODEL_NAME = "RandomForestRegressor"

# Variable globale pour stocker l'état du dernier entraînement
TRAINING_STATUS = {
    "last_run": None,
    "status": "idle",  # idle, running, success, failed
    "details": "Aucun entraînement lancé depuis le démarrage."
}

# Définition des schémas Pydantic
class PredictionInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    agence: str = Field(..., description="Agence en charge")
    secteur: str = Field(..., description="Secteur d'intervention")
    bi_multi_1: str = Field(..., alias="Bi/Multi.1", description="Type de canal")
    type_de_financement: str = Field(..., alias="Type de financement", description="Secteur public/privé/mixte")
    pays_beneficiaire: str = Field(..., alias="Pays beneficiaire", description="Pays ciblé")

class TrainInput(BaseModel):
    n_estimators: int = 100
    random_state: int = 42

@app.get("/")
def read_root():
    return {"status": "online", "message": "API MLOps opérationnelle (Configuration à 5 variables)"}

# ==============================================================================
# ENDPOINT METADATA DU MODÈLE CHAMPION
# ==============================================================================
@app.get("/model-metadata")
def get_model_metadata():
    """Récupère la version active tagguée 'champion' et ses métriques associées."""
    try:
        model_version_details = client.get_model_version_by_alias(name=MODEL_NAME, alias="champion")
        version = model_version_details.version
        run_id = model_version_details.run_id
        
        run_data = client.get_run(run_id).data
        metrics = run_data.metrics
        tags = run_data.tags
        
        return {
            "status": "success",
            "model_name": MODEL_NAME,
            "version": version,
            "run_id": run_id,
            "git_commit": tags.get("git_commit", "indisponible"),
            "metrics": {
                "val_r2": metrics.get("val_r2", 0.0),
                "mae": metrics.get("mae", 0.0),
                "val_rmse_k_eur": metrics.get("val_rmse_k_eur", 0.0)
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Impossible de récupérer les métadonnées du modèle champion : {str(e)}"
        }

# ==============================================================================
# FONCTION EN TÂCHE DE FOND + ENDPOINTS D'ENTRAÎNEMENT ASYNCHRONE
# ==============================================================================
def async_training_worker(n_estimators: int, random_state: int):
    """Worker exécuté en arrière-plan pour éviter de bloquer le serveur API."""
    try:
        res = train_model(n_estimators=n_estimators, random_state=random_state)
        if res.get("status") == "success":
            TRAINING_STATUS["status"] = "success"
            TRAINING_STATUS["details"] = f"Modèle v{res.get('model_version')} entraîné avec succès ! R²: {res['metrics']['r2']:.3f}"
        else:
            TRAINING_STATUS["status"] = "failed"
            TRAINING_STATUS["details"] = f"Échec de l'entraînement : {res.get('error')}"
    except Exception as e:
        TRAINING_STATUS["status"] = "failed"
        TRAINING_STATUS["details"] = f"Erreur critique tâche de fond : {str(e)}"

@app.post("/train")
def trigger_training(payload: TrainInput, background_tasks: BackgroundTasks, username: str = Depends(verification_auth)):
    """Déclenche le ré-entraînement en arrière-plan (anti-timeout). Protégé par Basic Auth."""
    
    if TRAINING_STATUS["status"] == "running":
        return {"status": "warning", "message": "Un entraînement est déjà en cours d'exécution."}
        
    TRAINING_STATUS["status"] = "running"
    TRAINING_STATUS["last_run"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    TRAINING_STATUS["details"] = "L'entraînement a été confié au worker FastAPI en arrière-plan..."

    # Ajout de la tâche lourde à la file d'attente de FastAPI
    background_tasks.add_task(async_training_worker, payload.n_estimators, payload.random_state)
    
    return {
        "status": "success", 
        "message": "Pipeline d'entraînement démarré avec succès en arrière-plan."
    }

@app.get("/train-status")
def get_train_status():
    """Permet au frontend de venir poller l'avancement de l'entraînement."""
    return TRAINING_STATUS

# ==============================================================================
# ENDPOINT ETL (Requis par le Scheduler)
# ==============================================================================
@app.post("/etl")
def trigger_etl(username: str = Depends(verification_auth)):
    """Exécute l'ETL. Protégé par Basic Auth."""
    try:
        print("[API] Déclenchement du script ETL...", flush=True)
        run_etl() 
        return {
            "status": "success",
            "message": "Pipeline ETL exécuté et fichier brut archivé avec succès."
        }
    except Exception as e:
        print("[API] CRASH DANS L'ETL — TRACEBACK COMPLET :", flush=True)
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=str(e))
    
# ==============================================================================
# ENDPOINT PRÉDICTION
# ==============================================================================
@app.post("/predict")
def predict(payload: PredictionInput):
    raw_data = payload.model_dump(by_alias=False)
    prediction_k_eur = make_prediction(raw_data)
    
    if prediction_k_eur is None:
        raise HTTPException(status_code=500, detail="Erreur d'exécution du modèle Champion.")

    try:
        with engine.begin() as connection:
            insert_query = text("""
                INSERT INTO public.predict_logs (
                    created_at, agence, secteur, bi_multi_1, type_de_financement, pays_beneficiaire, prediction_engagement_k_eur
                ) VALUES (
                    :created_at, :agence, :secteur, :bi_multi_1, :type_de_financement, :pays_beneficiaire, :prediction
                );
            """)
            connection.execute(insert_query, {
                "created_at": datetime.datetime.now(datetime.timezone.utc),
                "agence": raw_data["agence"],
                "secteur": raw_data["secteur"],
                "bi_multi_1": raw_data["bi_multi_1"],
                "type_de_financement": raw_data["type_de_financement"],
                "pays_beneficiaire": raw_data["pays_beneficiaire"],
                "prediction": float(prediction_k_eur)
            })
    except Exception as e:
        print(f"[WARN] Impossible d'écrire dans predict_logs : {str(e)}", flush=True)

    return {
        "status": "success",
        "prediction_engagement_k_eur": round(float(prediction_k_eur), 2)
    }