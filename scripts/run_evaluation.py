import os
import sys
import requests
from dotenv import load_dotenv

# 1. Chargement des variables d'environnement depuis le fichier .env
load_dotenv()

ADMIN_USER = os.getenv("ADMIN_USER")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# CORRECTION : L'endpoint réel configuré dans ton main.py est /drift
URL_API = "http://localhost:8000/drift"

# Sécurité : On valide immédiatement que le fichier .env est lu correctement
if not ADMIN_USER or not ADMIN_PASSWORD:
    print("❌ [Erreur Critique] Impossible de charger les identifiants de sécurité.")
    print("👉 Assurez-vous que le fichier '.env' est présent à la racine et contient ADMIN_USER et ADMIN_PASSWORD.")
    sys.exit(1)

def main():
    print("🚀 [Monitoring] Démarrage du script d'évaluation des dérives MLOps...")
    
    try:
        print(f"📡 Déclenchement de l'analyse sur l'API ({URL_API}) via Basic Auth...")
        
        # CORRECTION : On envoie une requête POST authentifiée sans corps JSON,
        # puisque l'API compare directement les tables Supabase afd et predict_logs.
        response = requests.post(
            URL_API,
            auth=(ADMIN_USER, ADMIN_PASSWORD),
            timeout=30
        )
        
        # 3. Traitement du retour de l'API
        if response.status_code == 200:
            print("✅ [Succès] L'API a traité l'analyse de dérive avec succès.")
            drift_res = response.json()
            
            global_drift = drift_res.get("global_drift", 0.0)
            run_id = drift_res.get("run_id", "N/A")
            action = drift_res.get("action_taken", "Aucune action requise.")
            
            print("\n📈 --- Rapport d'Observabilité (Evidently AI) ---")
            print(f"🔹 Statut : {drift_res.get('status', 'success')}")
            print(f"🔹 Score de Drift Global : {global_drift * 100:.2f} %")
            print(f"🔹 Run ID d'Analyse      : {run_id}")
            print(f"🔹 Décision Automatique  : {action}")
            print("--------------------------------------------------")
            
            if global_drift > 0.30:
                print("⚠️  Alerte : Le seuil de 30% est dépassé, le ré-entraînement a démarré en arrière-plan !")
        else:
            print(f"❌ [Erreur API] Le serveur a renvoyé un code {response.status_code}")
            print(f"Détail : {response.text}")
            sys.exit(1)
            
    except requests.exceptions.ConnectionError:
        print("❌ [Erreur Connexion] Impossible de joindre l'API. Vérifiez que votre conteneur tourne sur le port 8000.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ [Erreur Inattendue] : {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()