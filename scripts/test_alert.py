import time
import requests

# URL de ton API locale
URL = "http://localhost:8000/predict"

# On envoie un dictionnaire vide ou erroné pour forcer le bloc 'except Exception' à lever une erreur 500
bad_payload = {"data": {"Donnee_Invalide": "Crash_Me"}}

print("🚀 Lancement du spam de requêtes invalides pour lever l'alerte...")

# On bombarde l'API pendant quelques secondes pour faire grimper le 'rate' dans Prometheus
for i in range(20):
    try:
        response = requests.post(URL, json=bad_payload)
        print(f"Requête {i+1}/20 : Statut reçu = {response.status_code}")
    except Exception as e:
        print(f"Erreur de connexion : {e}")
    time.sleep(0.5)

print("\n🏁 Bombardement terminé. Dans l'onglet Alerting dans Grafana, l'alerte va passer au rouge d'ici quelques secondes !")