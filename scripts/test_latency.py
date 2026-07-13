import time
import requests

URL = "http://localhost:8000/predict"

# On ajoute notre paramètre secret "force_latency" : True
payload_lent = {"data": {"Pays beneficiaire": "France", "force_latency": True}}

print("🐌 Début du test de latence. L'API va ramer artificiellement...")

# On envoie une vingtaine de requêtes ralenties pour que Prometheus calcule une moyenne haute
for i in range(25):
    start = time.time()
    response = requests.post(URL, json=payload_lent)
    duration = time.time() - start
    print(f"Requête {i+1}/25 : Temps de réponse = {duration:.2f}s")

print("\nTest terminé. Grafana va détecter que la moyenne dépasse 0.5s et lever l'alerte.")