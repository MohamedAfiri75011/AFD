import time
import requests
import schedule

# URL de l'API FastAPI à l'intérieur du réseau Docker
API_URL_ETL = "http://fastapi_server:8000/etl"
API_URL_TRAIN = "http://fastapi_server:8000/train"
API_URL_DRIFT = "http://fastapi_server:8000/drift"

# Identifiants configurés sur l'API
AUTH_IDENTIFIANTS = ("admin_agence", "AFD2026!")

def run_automated_etl():
    print("[SCHEDULER] Lancement automatique de l'ETL...", flush=True)
    try:
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
        response = requests.post(API_URL_TRAIN, json=payload, auth=AUTH_IDENTIFIANTS, timeout=600)
        
        if response.status_code == 200:
            print(f"[SCHEDULER] Entraînement réussi : {response.json()}", flush=True)
        else:
            print(f"[SCHEDULER] Échec de l'entraînement (Code {response.status_code}) : {response.text}", flush=True)
    except Exception as e:
        print(f"[SCHEDULER] Erreur de connexion à l'API pendant l'entraînement : {e}", flush=True)

# Détection du Drift
def run_automated_drift():
    print("[SCHEDULER] Lancement automatique de l'analyse de dérive (Drift)...", flush=True)
    try:
        response = requests.post(API_URL_DRIFT, auth=AUTH_IDENTIFIANTS, timeout=600)
        
        if response.status_code == 200:
            result = response.json()
            score = result.get('global_drift', 0.0) * 100
            action = result.get('action_taken', 'Aucune action requise.')
            print(f"[SCHEDULER] Analyse terminée. Score de dérive : {score:.2f}%. Action : {action}", flush=True)
        else:
            print(f"[SCHEDULER] Échec de l'analyse de dérive (Code {response.status_code}) : {response.text}", flush=True)
    except Exception as e:
        print(f"[SCHEDULER] Erreur de connexion à l'API pendant l'analyse de dérive : {e}", flush=True)


# --- PLANIFICATION DES TÂCHES ---
# On décale les exécutions pour ne pas surcharger le serveur
schedule.every().day.at("01:00").do(run_automated_drift)   # 1h00 : On vérifie l'état des données du jour
schedule.every().day.at("02:00").do(run_automated_etl)     # 2h00 : On intègre les nouvelles données
schedule.every().day.at("03:00").do(run_automated_training) # 3h00 : On force un entraînement planifié

print("[SCHEDULER] Le planificateur a démarré avec succès et attend ses tâches...", flush=True)

while True:
    schedule.run_pending()
    time.sleep(1)