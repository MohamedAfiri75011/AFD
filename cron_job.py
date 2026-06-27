import os
import time
import requests
import schedule

# URL de ton API FastAPI
API_URL = "http://api:8000"

# Chemin vers le nouveau fichier CSV brut a integrer
CSV_SOURCE_PATH = "data/collect/nouveaux_engagements.csv"


def run_automated_pipeline():
    print(f"[INFO] [{time.strftime('%Y-%m-%d %H:%M:%S')}] Demarrage du pipeline planifie...")

    # --- 1. COLLECTE ET ETL ---
    if not os.path.exists(CSV_SOURCE_PATH):
        print(f"[WARNING] Fichier source introuvable a l'emplacement : {CSV_SOURCE_PATH}")
        print("Fin du pipeline (Pas de nouvelles donnees a traiter).")
        return

    print("[INFO] Envoi du CSV brut a l'API pour nettoyage et injection Supabase...")
    with open(CSV_SOURCE_PATH, "rb") as file:
        files = {"file": (os.path.basename(CSV_SOURCE_PATH), file, "text/csv")}
        response_etl = requests.post(f"{API_URL}/data/collect", files=files)

    if response_etl.status_code != 200:
        print(f"[ERROR] Echec de l'ETL : {response_etl.json()}")
        return

    print(f"[SUCCESS] ETL Reussi : {response_etl.json()['message']}")

    # --- 2. ENTRAINEMENT DU MODELE ---
    print("[INFO] Declenchement de l'entrainement asynchrone...")
    response_train = requests.post(f"{API_URL}/train")

    if response_train.status_code != 200:
        print(f"[ERROR] Echec du lancement de l'entrainement : {response_train.json()}")
        return

    print("[INFO] Entrainement lance en arriere-plan. Surveillance du statut...")

    # Boucle de verification du statut (Polling)
    while True:
        time.sleep(10)  # Attendre 10 secondes entre chaque verification
        status_response = requests.get(f"{API_URL}/train/status")
        status_data = status_response.json()

        if not status_data["running"]:
            last_result = status_data["last_result"]
            if last_result and "error" in last_result:
                print(f"[ERROR] L'entrainement a echoue en arriere-plan : {last_result['error']}")
            else:
                print("[SUCCESS] Entrainement termine ! Le modele a ete evalue et potentiellement promu.")
                try:
                    os.remove(CSV_SOURCE_PATH)
                    print("[CLEANUP] Fichier CSV brut source nettoye.")
                except Exception:
                    pass
            break
        else:
            print("[WAIT] Entrainement toujours en cours...")


# --- CONFIGURATION DE LA PLANIFICATION ---

# Exemple 1 : Executer tous les jours a minuit pile (Configuration de production)
schedule.every().day.at("00:00").do(run_automated_pipeline)

# Exemple 2 : Décommente la ligne ci-dessous si tu veux tester toutes les 10 minutes
# schedule.every(10).minutes.do(run_automated_pipeline)

print("[START] Le planificateur Python est actif. En attente de l'heure d'execution...")

# Boucle infinie pour maintenir le script en eveil permanent
while True:
    schedule.run_pending()
    time.sleep(1)  # Pause d'une seconde pour ne pas surcharger le processeur