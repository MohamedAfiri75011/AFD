import time
import requests
import schedule

# URL de l'API FastAPI à l'intérieur du réseau Docker
API_URL_ETL = "http://agence_api:8000/etl"
API_URL_TRAIN = "http://agence_api:8000/train"

# 🔐 Identifiants configurés sur l'API
AUTH_IDENTIFIANTS = ("admin_agence", "AFD2026!")

def run_automated_etl():
    print("[SCHEDULER] Lancement automatique de l'ETL...", flush=True)
    try:
        # 🚀 Ajout de l'argument 'auth' pour s'authentifier
        response = requests.post(API_URL_ETL, auth=AUTH_IDENTIFIANTS, timeout=600)
        if response.status_code == 200:
            print(f"[SCHEDULER] ETL exécuté avec succès : {response.json()}", flush=True)
        else:
            print(f"[SCHEDULER] Échec de l'ETL (Code {response.status_code}) : {response.text}", flush=True)
    except Exception as e:
        print(f"[SCHEDULER] Erreur de connexion à l'API pendant l'ETL : {e}", flush=True)

def run_automated_training():
    print("[SCHEDULER] Lancement automatique de l'entraînement...", flush=True)
    try:
        payload = {
            "n_estimators": 100,
            "random_state": 42
        }
        
        # 🚀 Ajout de l'argument 'auth' pour s'authentifier
        response = requests.post(API_URL_TRAIN, json=payload, auth=AUTH_IDENTIFIANTS, timeout=600)
        
        if response.status_code == 200:
            print(f"[SCHEDULER] Entraînement réussi : {response.json()}", flush=True)
        else:
            print(f"[SCHEDULER] Échec de l'entraînement (Code {response.status_code}) : {response.text}", flush=True)
    except Exception as e:
        print(f"[SCHEDULER] Erreur de connexion à l'API pendant l'entraînement : {e}", flush=True)

# --- PLANIFICATION DES TÂCHES ---
schedule.every(2).minutes.do(run_automated_etl)
schedule.every(2).minutes.do(run_automated_training)

print("[SCHEDULER] Le planificateur a démarré avec succès et attend ses tâches...", flush=True)

while True:
    schedule.run_pending()
    time.sleep(1)

# schedule.every().day.at("02:00").do(run_automated_etl)
# schedule.every().day.at("02:00").do(run_automated_training

