import datetime
import os
import json
import secrets
import traceback
import logging
import time
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status, Request, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge, Counter, Summary

from dotenv import load_dotenv
load_dotenv()

# Initialisation du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Imports locaux vers votre structure reelle de fichiers
from models.train_regression import train_model
from models.predict import make_prediction
from etl import run_etl

app = FastAPI(
    title="API MLOps - Specification Globale (49 Variables)",
    description="Backend industriel connecte a MLflow et Supabase avec instrumentation Prometheus.",
    version="3.3.0"
)

# ==============================================================================
# CONFIGURATION DE LA SECURITE (BASIC AUTH DEPUIS LE .ENV)
# ==============================================================================
ecurity = HTTPBasic()
ADMIN_USER = os.getenv("ADMIN_USER", "admin_agence")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")  # Plus de mot de passe par défaut ici !

def verification_auth(credentials: HTTPBasicCredentials = Depends(security)):
    """Verifie de maniere securisee les identifiants de Basic Auth."""
    # Si la variable n'est pas définie dans l'environnement, on refuse par sécurité
    if not ADMIN_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Configuration de sécurité manquante sur le serveur.",
        )
        
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USER)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiants d'acces incorrects ou manquants.",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# Connexion Supabase sécurisée
SUPABASE_DB_URI = os.getenv("SUPABASE_DB_URI")

if not SUPABASE_DB_URI:
    # En local, l'API lèvera cette erreur si le fichier .env est mal lu
    # En CI (GitHub Actions), notre variable dummy prendra le relais sans encombre
    raise ValueError(
        "Erreur : La variable d'environnement SUPABASE_DB_URI n'est pas définie !"
    )

engine = create_engine(SUPABASE_DB_URI, pool_size=2, max_overflow=0, pool_recycle=300)

# MLflow Tracking
MLFLOW_TRACKING_SERVER = os.getenv("MLFLOW_TRACKING_SERVER", "http://mlflow_server:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_SERVER)
client = MlflowClient()
MODEL_NAME = "RandomForestRegressor"

# ==============================================================================
# INSTRUMENTATION PROMETHEUS
# ==============================================================================
model_rmse_score = Gauge("model_rmse_score", "Root Mean Squared Error du modele de regression")
model_mae_score = Gauge("model_mae_score", "Mean Absolute Error du modele de regression")
model_r2_score = Gauge("model_r2_score", "R2 Score du modele de regression")
model_drift_value = Gauge("model_drift_value", "Valeur numerique brute du score de derive des donnees")
evidently_data_drift_detected_status = Gauge("evidently_data_drift_detected_status", "Statut global du Data Drift (1 = Alerte, 0 = OK)")

PREDICTION_METRIC = Summary('model_prediction_k_eur', 'Valeur predite par le Random Forest (en k EUR)')
COUNTRY_METRIC = Counter('model_input_country_total', 'Volume de predictions demandees par pays', ['pays'])

Instrumentator().instrument(app).expose(app)

# Etat de suivi du re-entrainement
TRAINING_STATUS = {
    "last_run": None,
    "status": "idle",  
    "details": "Aucun entrainement lance depuis le demarrage."
}

# Definition du schema Pydantic
class TrainInput(BaseModel):
    n_estimators: int = 100
    random_state: int = 42

@app.get("/")
def read_root():
    return {
        "status": "online", 
        "message": "API MLOps operationnelle", 
        "training_status": "busy" if TRAINING_STATUS["status"] == "running" else "idle"
    }

# ==============================================================================
# ENDPOINT METADATA DU MODELE CHAMPION
# ==============================================================================
@app.get("/model-metadata")
def get_model_metadata():
    try:
        champion_version = client.get_model_version_by_alias(MODEL_NAME, "champion")
        if champion_version:
            version_final = champion_version.version
            run_id = champion_version.run_id
            run_data = client.get_run(run_id).data
            
            r2_final = float(run_data.metrics.get("R2_log_val", run_data.metrics.get("r2_val", 0.0)))
            mae_final = float(run_data.metrics.get("val_mae_k_eur", run_data.metrics.get("mae_k_eur", 0.0)))
            rmse_final = float(run_data.metrics.get("val_rmse_k_eur", run_data.metrics.get("rmse_val", 0.0)))
            
            model_r2_score.set(r2_final)
            model_mae_score.set(mae_final)
            model_rmse_score.set(rmse_final)
        else:
            version_final, r2_final, mae_final, rmse_final = "Aucun", 0.0, 0.0, 0.0

        return {"status": "success", "version": version_final, "r2": r2_final, "mae": mae_final, "rmse": rmse_final}
    except Exception as e:
        logger.error(f"Erreur MLflow Metadata : {str(e)}")
        return {"status": "success", "version": "Non entraine", "r2": 0.0, "mae": 0.0, "rmse": 0.0}

# ==============================================================================
# PIPELINE D'ENTRAINEMENT ASYNCHRONE
# ==============================================================================
def async_training_worker(n_estimators: int, random_state: int):
    try:
        logger.info(f"[CT] Lancement de l'entrainement avec {n_estimators} arbres sur MLflow...")
        res = train_model(n_estimators=n_estimators, random_state=random_state)
        if res.get("status") == "success":
            metrics = res.get("metrics", {})
            
            TRAINING_STATUS["status"] = "success"
            TRAINING_STATUS["details"] = (
                f"Modele v{res.get('model_version')} entraine avec succes ! "
                f"R2: {metrics.get('r2_val', 0):.3f} | MAE: {metrics.get('mae_k_eur', 0):.1f} kEUR"
            )
            
            model_r2_score.set(metrics.get('r2_val', 0.0))
            model_mae_score.set(metrics.get('mae_k_eur', 0.0))
            model_rmse_score.set(metrics.get('rmse_val', metrics.get('rmse_k_eur', 0.0)))
            logger.info("[CT] Entrainement termine. Nouveau modele pousse sur MLflow.")
        else:
            TRAINING_STATUS["status"] = "failed"
            TRAINING_STATUS["details"] = f"Echec de l'entrainement : {res.get('error')}"
    except Exception as e:
        TRAINING_STATUS["status"] = "failed"
        TRAINING_STATUS["details"] = f"Erreur critique tache de fond : {str(e)}"

@app.post("/train")
def trigger_training(payload: TrainInput, background_tasks: BackgroundTasks, username: str = Depends(verification_auth)):
    if TRAINING_STATUS["status"] == "running":
        return {"status": "warning", "message": "Un entrainement est deja en cours d'execution."}
        
    TRAINING_STATUS["status"] = "running"
    TRAINING_STATUS["last_run"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    TRAINING_STATUS["details"] = "L'entrainement a ete delegue au worker FastAPI en arriere-plan..."

    background_tasks.add_task(async_training_worker, payload.n_estimators, payload.random_state)
    
    return {
        "status": "success", 
        "message": f"Pipeline d'entrainement demarre en arriere-plan ({payload.n_estimators} arbres)."
    }

@app.get("/train-status")
def get_train_status():
    return TRAINING_STATUS

# ==============================================================================
# WEBHOOK GRAFANA & ENDPOINTS COMPLEMENTAIRES
# ==============================================================================
@app.post("/retrain")
def grafana_webhook_retrain(payload: dict, background_tasks: BackgroundTasks):
    logger.info(f"[Webhook Grafana] Payload d'alerte recu : {json.dumps(payload)}")
    alert_status = payload.get("status", "firing")
    
    if alert_status in ["resolved", "ok"]:
        return {"status": "ignored", "message": "Alerte resolue."}
        
    if TRAINING_STATUS["status"] == "running":
        return {"status": "warning", "message": "Entrainement deja en cours."}
        
    TRAINING_STATUS["status"] = "running"
    TRAINING_STATUS["last_run"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    TRAINING_STATUS["details"] = "Re-entrainement d'urgence declenche automatiquement par le Webhook Grafana."

    background_tasks.add_task(async_training_worker, 100, 42)
    return {"status": "success", "message": "Continuous Training active via le Webhook."}

@app.post("/etl")
def trigger_etl(username: str = Depends(verification_auth)):
    try:
        run_etl() 
        return {"status": "success", "message": "Pipeline ETL execute avec succes."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/drift")
def trigger_drift_analysis(background_tasks: BackgroundTasks, username: str = Depends(verification_auth)):
    try:
        from models.compute_drift import calculate_data_drift
    except ImportError:
        from compute_drift import calculate_data_drift
        
    result = calculate_data_drift()
    if result["status"] == "failed":
        raise HTTPException(status_code=500, detail=result["error"])

    score_drift = result.get("global_drift", 0.0)
    seuil_critique = 0.01
    retraining_triggered = False

    model_drift_value.set(score_drift)

    if score_drift > seuil_critique:
        logger.warning(f"[ALERTE] Drift massif detecte : {score_drift*100:.1f}%.")
        evidently_data_drift_detected_status.set(1)
        background_tasks.add_task(async_training_worker, 100, 42)
        retraining_triggered = True
    else:
        evidently_data_drift_detected_status.set(0)
    
    result["action_taken"] = "Automated retraining started" if retraining_triggered else "No action needed."
    return result

# ==============================================================================
# ENDPOINT PREDICTION
# ==============================================================================
@app.post("/predict")
def predict(payload: dict, request: Request):
    start_time = time.time()
    status_code = "success"
    val_float = 0.0
    
    data_dict = payload.get("data", payload.get("inputs", payload))

    try:
        champion_version = client.get_model_version_by_alias(MODEL_NAME, "champion")
        v_model = str(champion_version.version)
        r_id = str(champion_version.run_id)
    except Exception:
        v_model, r_id = "unknown", "unknown"

    try:
        pays_saisi = data_dict.get("Pays beneficiaire", data_dict.get("pays", "Inconnu"))
        COUNTRY_METRIC.labels(pays=pays_saisi).inc()

        val_float = make_prediction(data_dict)
        PREDICTION_METRIC.observe(val_float)
    except Exception as e:
        status_code = "error"
        logger.error(f"Echec global de l'inference : {str(e)}")

    latency_ms = int((time.time() - start_time) * 1000)

    user_agent = request.headers.get("x-client-user-agent") or request.headers.get("user-agent", "Inconnu")
    ip_address = request.headers.get("x-client-ip") or (request.client.host if request.client else "127.0.0.1")

    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO public.predict_logs (
                    created_at, 
                    prediction_engagement_k_eur, 
                    inputs, 
                    model_version, 
                    run_id, 
                    latency_ms, 
                    user_agent, 
                    ip_address, 
                    status
                )
                VALUES (
                    NOW(), :pred, :inputs, :model_version, :run_id, :latency_ms, :user_agent, :ip_address, :status
                );
            """), {
                "pred": val_float if status_code == "success" else None,
                "inputs": json.dumps(data_dict),
                "model_version": v_model,
                "run_id": r_id,
                "latency_ms": latency_ms,
                "user_agent": user_agent,
                "ip_address": ip_address,
                "status": status_code
            })
        logger.info("[SQL] Historisation complete reussie avec metadonnees client.")
    except Exception as sql_err:
        logger.error(f"[SQL ERROR] : {str(sql_err)}")

    if status_code == "error":
        raise HTTPException(status_code=500, detail="Erreur interne du modele lors du calcul.")

    return {"status": "success", "prediction": val_float, "latency_ms": latency_ms}

@app.post("/trigger-drift")
def trigger_drift(username: str = Depends(verification_auth)):
    model_rmse_score.set(155.0)  
    model_mae_score.set(92.0)
    model_r2_score.set(-2.5)     
    model_drift_value.set(0.68)
    evidently_data_drift_detected_status.set(1) 
    return {"status": "alert_fired", "operator": username}

# ==============================================================================
# HISTORIQUE DES PREDICTIONS (POUR STREAMLIT)
# ==============================================================================
@app.get("/prediction-history")
def get_prediction_history(limit: int = 15):
    """Recupere l'integralite des colonnes de la table predict_logs depuis Supabase."""
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT id, created_at, prediction_engagement_k_eur, inputs, model_version, run_id, latency_ms, user_agent, ip_address, status 
                FROM public.predict_logs 
                ORDER BY created_at DESC 
                LIMIT :limit
            """)
            result = conn.execute(query, {"limit": limit})
            
            logs = []
            for row in result:
                logs.append({
                    "id": row[0],
                    "created_at": str(row[1]),
                    "prediction_engagement_k_eur": row[2],
                    "inputs": row[3],
                    "model_version": row[4],
                    "run_id": row[5],
                    "latency_ms": row[6],
                    "user_agent": row[7],
                    "ip_address": row[8],
                    "status": row[9]
                })
            return logs
    except Exception as e:
        logger.error(f"[SQL ERROR] Historique complet : {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur lors de la lecture de la base de donnees.")